# Neural Network
import numpy as np
import utils

class Network(object):
    def __init__(self, layers, loss, optimizer):
        self.layers = layers
        self.loss_func = loss
        self.optimizer = optimizer
        self.depth = len(self.layers)
        self.reg_funcs = [lay.reg for lay in layers]

    def forward(self,x):
        self.reg_fwd = []
        for i,(layer,rf) in enumerate(zip(self.layers, self.reg_funcs)):
            x = layer(x)
            if not layer.act:
                self.reg_fwd.append(rf.f(layer.params['W'], layer.reg_param))
        return x

    def loss(self,x,y):
        x = self.forward(x)
        return self.loss_func(x,y)+np.sum(self.reg_fwd)

    def backward(self):
        self.params,self.grads = [],[]
        grad = self.loss_func.backward()
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
            if not layer.act:
                self.params.extend([layer.params['W'],layer.params['b']])
                self.grads.extend([layer.grads['dW'],layer.grads['db']])
        return grad

    def predict(self,x):
        return np.argmax(self.forward(x),axis=1)

    def train(self,x,y,epochs=100,batch_size=64,val_sets =(),one_hot=True,verbose=False):
        losses,val_acc,tr_acc = [],[],[]
        for e in range(epochs):
            batch_sampler = utils.Batcher(x,y,batch_size)
            for batch_x, batch_y in batch_sampler:

                loss_val = self.loss(batch_x, batch_y)
                self.backward()
                self.optimizer.step(self)

            if verbose:
                if e % int(epochs/10)==0:
                    print(f'loss at epoch {e}: {loss_val}')
            if len(val_sets)>0:
                val_acc.append(utils.accuracy(self, val_sets[0], val_sets[1], convert=one_hot))
            losses.append(loss_val)
            tr_acc.append(utils.accuracy(self, batch_x, batch_y, convert=one_hot))
        if len(val_sets)>0:
            return losses, tr_acc, val_acc
        return losses, tr_acc
