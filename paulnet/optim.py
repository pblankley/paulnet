# Optimizers
import numpy as np

class Optimizer(object):
    def step(self):
        raise NotImplementedError()

class SGD(Optimizer):
    """ Classic SGD. This optimizer defaults to the no momentum version, but you can 
    speficy momentum if you would like.
    --------
    Args: lr, float the learning rate 
          lr_decay; float, the coefficient used to decay the learning rate 
            i.e. lr_(t) = lr_decay * lr_(t-1)
          beta; momentum parameter; the larger this is the more you 'remember' 
            previous steps (gradients)
    """
    def __init__(self,lr,lr_decay=1.0,beta=0.0):
        self.lr = lr
        self.lr_decay = lr_decay
        self.beta = beta
        self.velo = []

    def decay(self):
        self.lr = self.lr*self.lr_decay

    def step(self, net):
        if len(self.velo)!=len(net.grads):
            self.velo = [0]*len(net.grads)
        for i in range(len(net.grads)):

            self.velo[i] = self.beta*self.velo[i] + (1-self.beta)*net.grads[i]
            net.params[i] -= self.lr * self.velo[i]

class RMSProp(Optimizer):
    """ RMSProp. This optimizer calibrates the learning rate as it goes using 
    the square of the gradient with the beta parameter governing the strength of the 
    'memory' of the square of the gradient 
    --------
    Args: lr, float the learning rate 
          lr_decay; float, the coefficient used to decay the learning rate 
            i.e. lr_(t) = lr_decay * lr_(t-1)
          beta; momentum parameter; the larger this is the more you 'remember' 
            previous steps (squared gradients)
          eps; float, the minute value added to make sure we never divide by zero in the update.
    """
    def __init__(self,lr,lr_decay=1.0,beta=0.999,eps=1e-8):
        self.lr = lr
        self.lr_decay = lr_decay
        self.beta = beta
        self.eps = eps
        self.moment = []

    def decay(self):
        self.lr = self.lr*self.lr_decay

    def step(self, net):
        if len(self.moment)!=len(net.grads):
            self.moment = [0]*len(net.grads)
        for i in range(len(net.grads)):

            self.moment[i] = self.beta*self.moment[i] + (1-self.beta)*(net.grads[i]**2)
            update =  (1/(np.sqrt(self.moment[i])+self.eps)) * net.grads[i]
            net.params[i] -= self.lr * update

class Adam(Optimizer):
    """ Adam optimizer. This optimizer calibrates the learning rate as it goes using 
    the square of the gradient with the beta parameter governing the strength of the 
    'memory' of the square of the gradient. It usually represents an improvement to 
    RMSProp (which does the above) because it adds gradient (non-squared) information 
    to the update and includes a stabilizing normalization step.
    --------
    Args: lr, float the learning rate 
          lr_decay; float, the coefficient used to decay the learning rate 
            i.e. lr_(t) = lr_decay * lr_(t-1)
          beta1; momentum parameter (gradient); the larger this is the more you 'remember' 
            previous steps (gradients)
          beta2; momentum parameter (squared gradient); the larger this is the more you 'remember' 
            previous steps (sqaured gradients)
          eps; float, the minute value added to make sure we never divide by zero in the update.
    """
    def __init__(self,lr,lr_decay=1.0,beta1=0.999,beta2=0.999,eps=1e-8):
        self.lr = lr
        self.lr_decay = lr_decay
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.velo = []
        self.moment = []

    def decay(self):
        self.lr = self.lr*self.lr_decay

    def step(self, net):
        if len(self.moment)!=len(net.grads):
            self.moment = [0]*len(net.grads)
        if len(self.velo)!=len(net.grads):
            self.velo = [0]*len(net.grads)
        for i in range(len(net.grads)):

            self.velo[i] = self.beta1*self.velo[i] + (1-self.beta1)*net.grads[i]
            self.moment[i] = self.beta2*self.moment[i] + (1-self.beta2)*(net.grads[i]**2)
            velo_hat = self.velo[i] / (1-self.beta1)
            mom_hat = self.moment[i] / (1-self.beta2)
            update =  velo_hat / (np.sqrt(mom_hat)+self.eps)
            net.params[i] -= self.lr * update
