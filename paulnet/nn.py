# Neural Network
import numpy as np
import utils

class Network(object):
    """ Simple neural network class to take an arbitrary number of layers.
    -------
    Args: layers; list of layer objects
          loss; the loss function to use for this neural net
          optimizer; the optimizer to use for this neural net
    """
    def __init__(self, layers, loss, optimizer):
        self.layers = layers
        self.loss_func = loss
        self.optimizer = optimizer
        self.depth = len(self.layers)
        self.reg_funcs = [lay.reg for lay in layers]

    def forward(self,x):
        """ This method calculates the result of the network. Here we just loop
        through the layers calculating the output and passing it to the next layer
        We also calculate any regularization associated with a layer and add that
        to a collector to be added later to the loss
        -------
        Args: x; numpy array of dimension nxm where n in the number of points and
                    m is the number of features
        -------
        Returns: numpy array of shape denoted by the final layer
        """
        self.reg_fwd = []
        for i,(layer,rf) in enumerate(zip(self.layers, self.reg_funcs)):
            x = layer(x)
            if not layer.act:
                self.reg_fwd.append(rf.f(layer.params['W'], layer.reg_param))
        return x

    def loss(self,x,y):
        """ This method calls forward internally and calculates the loss of the network
        ------
        Args: x; numpy array of dimension nxm where n in the number of points and
                    m is the number of features
              y; numpy array of size nxc where n is the number of points and c is
                    the number of classes - one in the case of regression.
        ------
        Returns: float; the loss of the network
        """
        x = self.forward(x)
        return self.loss_func(x,y)+np.sum(self.reg_fwd)

    def backward(self):
        """ This method back propogates the gradient from the loss and updates the 
        values of the parameters. 
        ------
        Args: None
        ------
        Returns: the gradient of the most-backward layer
        ------
        NOTE: You must call the loss function before you call this function  or this will not work.
        """
        self.params,self.grads = [],[]
        grad = self.loss_func.backward()
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
            if not layer.act:
                self.params.extend([layer.params['W'],layer.params['b']])
                self.grads.extend([layer.grads['dW'],layer.grads['db']])
        return grad

    def predict(self,x):
        """ This just takes the argmax to get the prediction for a classification problem. """
        return np.argmax(self.forward(x),axis=1)

    def train(self,x,y,epochs=100,batch_size=64,val_sets =(),one_hot=True,verbose=False):
        """ trains the network for the specified number of epochs. 
        -------
        Args: x; numpy array of dimension nxm where n in the number of points and
                    m is the number of features
              y; numpy array of size nxc wher n is the number of points and c is the 
                    number of classes (classes=1 for regression)
              epochs; int, the number of times the network runs through the dataset passed to it.
              batch_size; int, the size of the batch to run through the network (usually a power of 2)
              val_sels; tuple with the x and y similar to the input for validation accuracy (defaults to empty)
              one_hot; bool, if true the y values are alrady one hot encoded.
              verbose; bool, just what it sounds like 
        -------
        Returns: tuples with the losses and training accuracy it each batch, or tuple with those two 
                    and the validation accuracy if validation sets passed in the args
        """
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
