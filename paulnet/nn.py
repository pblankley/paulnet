# Neural Network
import numpy as np
import data as data_util
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
        # print(self.reg_fwd)
        return self.loss_func(x,y)+np.sum(self.reg_fwd)

    def backward(self):
        self.params,self.grads = [],[]
        grad = self.loss_func.backward()
        for layer in reversed(self.layers):
            # print(grad)
            grad = layer.backward(grad)
            if not layer.act:
                self.params.extend([layer.params['W'],layer.params['b']])
                self.grads.extend([layer.grads['dW'],layer.grads['db']])
        return grad

    def predict(self,x):
        return np.argmax(self.forward(x),axis=1)

    def pavlos_train(self,x,y,num_iterations=100,batch_size=64,val_sets =(),one_hot=True,verbose=False):
        # Change below after assignment TODO
        costs, val_acc,tr_acc = [],[],[]
        for it in range(num_iterations):
            batch_idx = np.random.choice(range(len(y)),size=batch_size)
            batch_x, batch_y = x[batch_idx,:],  y[batch_idx,:]

            loss_val = self.loss(batch_x, batch_y)
            grads = self.backward()
            self.optimizer.step(self)


            if it%100==0:
                if verbose:
                    print(f'cost at iteration {it}: {loss_val}')
                self.optimizer.decay()
                if len(val_sets)>0:
                    val_acc.append(utils.accuracy(self,val_sets[0], val_sets[1], convert=one_hot))
                    tr_acc.append(utils.accuracy(self,x, y, convert=one_hot))
            costs.append(loss_val)
        if len(val_sets)>0:
            return costs, val_acc, tr_acc
        return costs

    def train(self,x,y,epochs=100,batch_size=64,val_sets =(),one_hot=True,verbose=False):
        losses,val_acc = [],[]
        for e in range(epochs):
            batch_sampler = data_util.Batcher(x,y,batch_size)
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
        if len(val_sets)>0:
            return losses, val_acc
        return losses
