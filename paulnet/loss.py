# Loss functions
import numpy as np

class Loss(object):
    """ Base class of all losses """
    def __init__(self):
        self.inpt = None

    def forward(self,x):
        raise NotImplementedError()

    def backward(self,grad):
        raise NotImplementedError()

    def __call__(self,x,y):
        return self.forward(x,y)

class CrossEntropy(Loss):
    """ Cross entropy loss """
    def __init__(self):
        self.model_params = None
        self.reg = None

    def forward(self,x,yt):
        """ The forward method that, given the previous layers output, will return the loss
        ------
        Args: x, numpy array nxm
              yt, numpy array, nxc the true y value (must be a one hot) 
        ------
        Returns: float, the value of the loss
        """
        self.y_true = yt
        self.nobs = yt.shape[0]
        self.inpt = x - np.max(x, axis=1).reshape(-1,1)
        return -1/self.nobs*np.sum(yt*(self.inpt - np.log(np.sum(np.exp(self.inpt), axis=1, keepdims=True))))

    def backward(self):
        """ Calculates and returns the gradient of the loss function. """
        sm_out = np.exp(self.inpt)/np.exp(self.inpt).sum(axis=1, keepdims=True)
        return (sm_out-self.y_true)/self.nobs
