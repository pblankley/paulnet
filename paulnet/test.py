import numpy as np
from random import randrange

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


def grad_check_sparse(f, x, analytic_grad, num_checks=10, epsilon=1e-7):
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
        print('numerical: %f analytic: %f, relative error: %e' % (grad_numerical, grad_analytic, rel_error))
