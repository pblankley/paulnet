# Neural Net framework
import numpy as np
import utils
np.random.seed(42)

class Layer(object):
    def __init__(self):
        self.params = dict()
        self.grads = dict()

    def forward(self, x):
        raise NotImplementedError()

    def backward(self, grad):
        raise NotImplementedError()

    def __call__(self,x):
        return self.forward(x)

class Linear(Layer):
    """ Standard linear layer """
    def __init__(self, inpt_dim, out_dim, reg='None',reg_param=0.0, dropout=0.0, use_bias=True, init_type='normal',init_vals=()):
        super(Linear,self).__init__()
        self.inpt_dim = inpt_dim
        self.out_dim = out_dim
        self.act = False
        if reg not in {'None','l2', 'frob'}:
            raise NotImplementedError()
        self.reg = self.init_reg(reg)
        self.reg_param = reg_param
        self.dropout = dropout
        self.use_bias = use_bias

        if len(init_vals)!=0:
            assert(init_vals[0].shape==(self.inpt_dim, self.out_dim))
            assert(init_vals[1].shape==(self.out_dim,))
            self.params['W'] = init_vals[0]
            self.params['b'] = init_vals[1]
        else:
            valid_types = {'normal','zero'}
            if init_type not in valid_types:
                raise ValueError('Specify a valid initialization type')

            if init_type=='normal':
                self.params['W'] = np.random.randn(self.inpt_dim, self.out_dim) * 1e-1
                self.params['b'] = np.random.randn(self.out_dim) * 1e-1
            else:
                self.params['W'] = np.zeros((self.inpt_dim,self.out_dim))
                self.params['b'] = np.zeros((self.out_dim))

    def forward(self, x):
        self.inpt = x
        self.dr_mat = np.random.binomial(n=1,p=(1-self.dropout),size=(self.inpt.shape[0],self.params['W'].shape[1]))
        # return (1/(1-self.dropout))*x@self.params['W']*self.dr_mat+self.params['b']
        return x@self.params['W']+self.params['b']

    def backward(self, grad):
        if self.use_bias:
            self.grads['db'] = np.sum(grad,axis=0)
        # self.grads['dW'] = (1/(1-self.dropout))*self.inpt.T @ grad + self.reg.b(self.params['W'], self.reg_param)
        self.grads['dW'] = self.inpt.T @ grad + self.reg.b(self.params['W'], self.reg_param)
        # print(grad.shape, self.params['W'].T.shape, self.params['b'].shape)
        return grad @ self.params['W'].T

    def init_reg(self, rtype):
        if rtype=='None':
            return utils.no_reg
        elif rtype=='l2':
            return utils.l2_reg
        elif rtype=='frob':
            return utils.frob_reg
        else:
            raise NotImplementedError()

class Relu(Layer):
    """ Traditional ReLu layer """
    def __init__(self):
        self.act = True
        self.inpt = None
        self.reg = utils.no_reg

    def forward(self, x):
        self.inpt = x
        return np.maximum(np.zeros(x.shape),x)

    def backward(self, grad):
        grad[self.inpt < 0] = 0
        return grad


class Softmax(Layer):
    def __init__(self):
        self.act = True
        self.inpt = None
        self.reg = utils.no_reg

    def forward(self, x):
        self.inpt = x - np.max(x, axis=1).reshape(-1,1)
        return np.exp(self.inpt)/np.exp(self.inpt).sum(axis=1, keepdims=True)

    def backward(self, grad):
        sm_out = np.exp(self.inpt)/np.exp(self.inpt).sum(axis=1, keepdims=True)
        return np.diag(sm_out) - sm_out @ sm_out.T

class Log_softmax(Layer):
    def __init__(self):
        self.act = True
        self.inpt = None
        self.reg = utils.no_reg

    def forward(self, x):
        self.inpt = x - np.max(x, axis=1).reshape(-1,1)
        return np.log(np.exp(self.inpt)/np.exp(self.inpt).sum(axis=1, keepdims=True))

    def backward(self, grad):
        return np.zeros(self.inpt.shape)

class Softplus(Layer):
    """ SoftPlus layer """
    def __init__(self):
        self.act = True
        self.inpt = None

    def forward(self, x):
        self.inpt = x
        return np.log(1.0 + np.exp(x))

    def backward(self, grad):
        sp_grad = 1/(1+np.exp(-self.inpt))
        return grad + sp_grad
