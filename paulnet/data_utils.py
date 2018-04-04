from __future__ import print_function
from random import randrange
import random
from six.moves import cPickle as pickle
import numpy as np
import os
from scipy.misc import imread
import platform

def grad_check_scalar(f, x, epsilon=1e-7):
    """
    Implement the numerical gradient for a function with a single scalar.

    Arguments:
    f -- a function that takes a scalar value
    x -- a real-valued scalar to evaluate the gradient
    epsilon -- tiny shift to the input to compute the approximated gradient

    Returns:
    the approximated numerical gradient
    """
    xplus = x + epsilon
    xminus = x - epsilon
    f_plus = f(xplus)
    f_minus = f(xminus)
    grad = (f_plus - f_minus) / (2 * epsilon)
    return grad


def grad_check(f, x, epsilon=1e-7):
    """
    Implements the numerical gradient for a function with a vector input.

    Arguments:
    f -- a function that takes a vector argument
    x -- input datapoint, of shape (input size, 1)
    epsilon -- tiny shift to the input to compute approximated gradient

    Returns:
    the approximated numerical gradient
    """
    # Set-up variables
    xshape = x.shape
    input_size = x.size
    grad = np.zeros((input_size,))
    x = x.ravel()

    # Compute grad
    for i in range(input_size):
        # Compute f_plus[i]
        oldval = x[i]
        x[i] = oldval + epsilon
        f_plus =  f(x.reshape(xshape))

        # Compute f_minus[i]
        x[i] = oldval - epsilon
        f_minus = f(x.reshape(xshape))

        # Restore
        x[i] = oldval

        # Compute gradapprox[i]
        grad[i] = (f_plus - f_minus) / (2 * epsilon)
    return grad.reshape(xshape)


def compare_grads(analytic_grad, num_grad):
    """
    Compares the relative difference between the numerical gradient and
    approximated gradient.

    Arguments:
    analytic_grad -- analytically evaluated grad
    num_grad -- numerically approximated grad

    Returns:
    the relative difference between both gradients.
    """
    numerator = np.linalg.norm(analytic_grad - num_grad)
    denominator = np.linalg.norm(analytic_grad) + np.linalg.norm(num_grad)
    return numerator / denominator


def grad_check_sparse(f, x, analytic_grad, num_checks=10, seed=42, epsilon=1e-5):
    """
    Sample a few random elements and only return the relative distance
    between the numerical and analyitical gradient.

    Arguments:
    f -- a function that takes a vector argument
    x -- input ndarray datapoint
    analytic_grad -- analytically evaluated grad
    num_checks -- number of coordinates to evaluate
    epsilon -- tiny shift to the input to compute approximated gradient
    seed -- indicate seed for randomness control

    Returns: nothing
    prints the relative difference between gradients for the sampled values.
    """
    random.seed(seed)
    rel_errors = np.empty(num_checks)
    for i in range(num_checks):
        ix = tuple([randrange(m) for m in x.shape])

        oldval = x[ix]
        x[ix] = oldval + epsilon        # increment by epsilon
        f_pos = f(x)                    # evaluate f(x + epsilon)
        x[ix] = oldval - epsilon        # increment by epsilon
        f_minus = f(x)                  # evaluate f(x - epsilon)
        x[ix] = oldval                  # reset

        grad_numerical = (f_pos - f_minus) / (2 * epsilon)
        grad_analytic = analytic_grad[ix]
        rel_error = abs(grad_numerical - grad_analytic) / (abs(grad_numerical) + abs(grad_analytic))
        rel_errors[i] = rel_error
    return rel_errors

def load_pickle(f):
    version = platform.python_version_tuple()
    if version[0] == '2':
        return  pickle.load(f)
    elif version[0] == '3':
        return  pickle.load(f, encoding='latin1')
    raise ValueError("invalid python version: {}".format(version))

def load_CIFAR_batch(filename):
  """ load single batch of cifar """
  with open(filename, 'rb') as f:
    datadict = load_pickle(f)
    X = datadict['data']
    Y = datadict['labels']
    X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("float")
    Y = np.array(Y)
    return X, Y

def load_CIFAR10(ROOT):
  """ load all of cifar """
  xs = []
  ys = []
  for b in range(1,6):
    f = os.path.join(ROOT, 'data_batch_%d' % (b, ))
    X, Y = load_CIFAR_batch(f)
    xs.append(X)
    ys.append(Y)
  Xtr = np.concatenate(xs)
  Ytr = np.concatenate(ys)
  del X, Y
  Xte, Yte = load_CIFAR_batch(os.path.join(ROOT, 'test_batch'))
  return Xtr, Ytr, Xte, Yte


def get_CIFAR10_data(num_training=49000, num_validation=1000, num_test=1000, num_dev=500):
    """
    Load the CIFAR-10 dataset from disk and perform preprocessing to prepare
    it for the linear classifier. These are the same steps as we used for the
    SVM, but condensed to a single function.
    """
    # Load the raw CIFAR-10 data
    cifar10_dir = '../datasets/cifar-10-batches-py'
    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

    # subsample the data
    # Validation set
    mask = list(range(num_training, num_training + num_validation))
    X_val = X_train[mask]
    y_val = y_train[mask]
    # Training set
    mask = list(range(num_training))
    X_train = X_train[mask]
    y_train = y_train[mask]
    # Test set
    mask = list(range(num_test))
    X_test = X_test[mask]
    y_test = y_test[mask]
    # Dev data set: just for debugging purposes, it overlaps with the training set,
    # but has a smaller size.
    mask = np.random.choice(num_training, num_dev, replace=False)
    X_dev = X_train[mask]
    y_dev = y_train[mask]

    # Preprocessing: reshape the image data into rows
    X_train = np.reshape(X_train, (X_train.shape[0], -1))
    X_val = np.reshape(X_val, (X_val.shape[0], -1))
    X_test = np.reshape(X_test, (X_test.shape[0], -1))
    X_dev = np.reshape(X_dev, (X_dev.shape[0], -1))

    # Normalize the data: subtract the mean image
    mean_image = np.mean(X_train, axis = 0)
    X_train -= mean_image
    X_val -= mean_image
    X_test -= mean_image
    X_dev -= mean_image

    return X_train, y_train, X_val, y_val, X_test, y_test, X_dev, y_dev

def load_models(models_dir):
  """
  Load saved models from disk. This will attempt to unpickle all files in a
  directory; any files that give errors on unpickling (such as README.txt) will
  be skipped.

  Inputs:
  - models_dir: String giving the path to a directory containing model files.
    Each model file is a pickled dictionary with a 'model' field.

  Returns:
  A dictionary mapping model file names to models.
  """
  models = {}
  for model_file in os.listdir(models_dir):
    with open(os.path.join(models_dir, model_file), 'rb') as f:
      try:
        models[model_file] = load_pickle(f)['model']
      except pickle.UnpicklingError:
        continue
  return models
