# Loss functions
import numpy as np

class Loss(object):
    def __init__(self):
        self.inpt = None

    def forward(self,x):
        raise NotImplementedError()

    def backward(self,grad):
        raise NotImplementedError()

    def __call__(self,x,y):
        return self.forward(x,y)

class CrossEntropy(Loss):
    def __init__(self):
        self.model_params = None
        self.reg = None

    def forward(self,x,yt):
        self.y_true = yt
        self.nobs = yt.shape[0]
        self.inpt = x - np.max(x, axis=1).reshape(-1,1)
        return -1/self.nobs*np.sum(yt*(self.inpt - np.log(np.sum(np.exp(self.inpt), axis=1, keepdims=True))))

    def backward(self):
        sm_out = np.exp(self.inpt)/np.exp(self.inpt).sum(axis=1, keepdims=True)
        return (sm_out-self.y_true)/self.nobs
