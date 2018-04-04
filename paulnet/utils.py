# Utils
import numpy as np
from sklearn.preprocessing import OneHotEncoder

class Batcher(object):
    """ Iterable that iterates over batches of the specified size until it exhausts the dataset. 
    NOTE: if you pass a batch size not a even divisor to the dataset size, the final batch will be the 
        size of the remainder.
    """
    def __init__(self,x,y, batch_size=64):
        self.x = x
        self.y = y if len(y.shape)>1 else y.reshape(-1,1)
        self.bs = batch_size
        # Always shuffle

    def __iter__(self):
        batch = []
        for idx in np.random.permutation(range(self.x.shape[0])):
            batch.append(idx)
            if len(batch)==self.bs:
                yield (self.x[batch, :], self.y[batch,:])
                batch = []
        if len(batch)>0:
            yield (self.x[batch, :], self.y[batch,:])

    def __len__(self):
        return np.ceil(len(self.y)/self.bs)

class DataLoader(object):
    """ Still in process """
    def __init__(self,x,y, batch_size=64, shuffle=True):
        self.x = x
        self.y = y if len(y.shape)>1 else y.reshape(-1,1)
        self.bs = batch_size

    def __iter__(self):
        raise NotImplementedError()

    def __len__(self):
        raise NotImplementedError()
        
class no_reg(object):
    """ Class to specify no regularization in a layer """
    @classmethod
    def f(cls,weights,reg_param):
        return 0.0
    @classmethod
    def b(cls,weights,reg_param):
        return 0.0

class l2_reg(object):
    """ Class to specify l2 regularization in a layer """
    @classmethod
    def f(cls,weights,reg_param):
        return 0.5*reg_param*np.linalg.norm(weights,ord=2)**2
    @classmethod
    def b(cls,weights,reg_param):
        return reg_param*weights

class frob_reg(object):
    """ Class to specify frobinious norm regularization in a layer """
    @classmethod
    def f(cls,weights,reg_param):
        return 0.5*reg_param*np.linalg.norm(weights)**2
    @classmethod
    def b(cls,weights,reg_param):
        return reg_param*weights

class Eval(object):
    @classmethod
    def accuracy(cls, y_pred, y_true):
        return np.mean(np.argmax(y_pred,axis=1)==np.argmax(y_true,axis=1))

def accuracy(model,x,y_true,convert=False):
    """ Returns the classification accuracy
    ------
    Args: model; a Network class
          x; a nxm numpy array 
          y; either a one hot vector or a length n array with the correct class 
          convert; bool, if True convert the one hot to the class,
            if False assume already in class form.
    ------
    Returns; float, the accuracy of the model with the passed data.
    """
    if convert:
        y_true = np.argmax(y_true, axis=1)
    y_pred = model.predict(x)
    return np.mean(y_pred == y_true)

def encode_labels(y):
    """ Encodes from categorical to one hot vector
    ------
    Args: y, numpy array of length n with c classes 
    ------
    Returns; numpy array, matrix of size nxc
    """
    enc = OneHotEncoder()
    enc.fit(y.reshape(-1,1))
    y_enc = enc.transform(y.reshape(-1,1)).toarray()
    return y_enc

def rel_error(x, y):
    """ returns relative error """
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))
