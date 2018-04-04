# Utils
import numpy as np
from sklearn.preprocessing import OneHotEncoder

class Batcher(object):
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
    def __init__(self,x,y, batch_size=64, shuffle=True):
        self.x = x
        self.y = y if len(y.shape)>1 else y.reshape(-1,1)
        self.bs = batch_size

    def __iter__(self):
        raise NotImplementedError()

    def __len__(self):
        raise NotImplementedError()
        
class no_reg(object):
    @classmethod
    def f(cls,weights,reg_param):
        return 0.0
    @classmethod
    def b(cls,weights,reg_param):
        return 0.0

class l2_reg(object):
    @classmethod
    def f(cls,weights,reg_param):
        return 0.5*reg_param*np.linalg.norm(weights,ord=2)**2
    @classmethod
    def b(cls,weights,reg_param):
        return reg_param*weights

class frob_reg(object):
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
    """ Returns the classification accuravy """
    if convert:
        y_true = np.argmax(y_true, axis=1)
    y_pred = model.predict(x)
    return np.mean(y_pred == y_true)

def encode_labels(y):
    """ Encodes from categorical to one hot vector """
    enc = OneHotEncoder()
    enc.fit(y.reshape(-1,1))
    y_enc = enc.transform(y.reshape(-1,1)).toarray()
    return y_enc

def rel_error(x, y):
    """ returns relative error """
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))
