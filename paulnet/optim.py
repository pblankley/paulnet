# Optimizers
import numpy as np

class Optimizer(object):
    def step(self):
        raise NotImplementedError()

class SGD(Optimizer):
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
