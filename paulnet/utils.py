# Utils
import numpy as np

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
    if convert:
        y_true = np.argmax(y_true, axis=1)
    y_pred = model.predict(x)
    return np.mean(y_pred == y_true)
