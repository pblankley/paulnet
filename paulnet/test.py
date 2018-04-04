import numpy as np
import utils as ut
import nn
import layers
import loss as ls
import optim
import data_utils as dutil


# Global variables
X_train, y_train, X_val, y_val, X_test, y_test, X_dev, y_dev = dutil.get_CIFAR10_data()
n = X_train.shape[1]         # features dimension
c = 10                       # number of classes in the database
Y_dev_enc = ut.encode_labels(y_dev)

def test_CrossEntropyLoss():
    np.random.seed(1)
    W = np.random.randn(c,n) * 0.0001
    b = np.random.randn(c,1) * 0.0001
    layer_lin = layers.Linear(n,c,init_vals=(W.T,b.ravel()))
    loss_func = ls.CrossEntropy()
    net = nn.Network([layer_lin], loss_func, optimizer=None)
    my_loss = net.loss(X_dev, Y_dev_enc)
    assert(np.isclose(my_loss,-np.log(.1),atol=1e-2))

def test_CrossEntropy_Linear_Grad():
    np.random.seed(1)
    W = np.random.randn(c,n) * 0.0001
    b = np.random.randn(c,1) * 0.0001
    layer_lin = layers.Linear(n,c, reg='l2',reg_param=0.05,init_vals=(W.T,b.ravel()))
    loss_func = ls.CrossEntropy()
    net = nn.Network([layer_lin], loss_func, optimizer=None)
    net_loss = net.loss(X_dev,Y_dev_enc)
    ngrad = net.backward()

    # Define functions to pass to helper
    def loss_func_W(ww):
        layer_lin = layers.Linear(n,c, reg='l2',reg_param=0.05,init_vals=(ww.T,b.ravel()))
        loss_func = ls.CrossEntropy()
        net = nn.Network([layer_lin], loss_func, optimizer=None)
        return net.loss(X_dev,Y_dev_enc)
    def loss_func_b(bb):
        layer_lin = layers.Linear(n,c, reg='l2',reg_param=0.05,init_vals=(W.T,bb.ravel()))
        loss_func = ls.CrossEntropy()
        net = nn.Network([layer_lin], loss_func, optimizer=None)
        return net.loss(X_dev,Y_dev_enc)
    # Actually run the test
    rel_err_weight = dutil.grad_check_sparse(loss_func_W, W, net.grads[0].T, 10,seed=42)
    rel_err_bias = dutil.grad_check_sparse(loss_func_b, b.ravel(), net.grads[1], 10,seed=42)
    assert(np.allclose(rel_err_weight,np.zeros(rel_err_weight.shape),atol=1e-4))
    assert(np.allclose(rel_err_bias,np.zeros(rel_err_bias.shape),atol=1e-4))

####### Helpers ############
def init_2layer_net(input_size, hidden_size, output_size, std=1e-4):
    params = {}
    params['W1'] = std * np.random.randn(hidden_size, input_size)
    params['b1'] = np.zeros((hidden_size, 1))
    params['W2'] = std * np.random.randn(output_size, hidden_size)
    params['b2'] = np.zeros((output_size, 1))
    return params

def init_toy_model():
    np.random.seed(0)
    input_size = 4
    hidden_size = 10
    num_classes = 3
    return init_2layer_net(input_size, hidden_size, num_classes, std=1e-1)

def init_toy_data():
    np.random.seed(1)
    num_inputs = 5
    input_size = 4
    X = 10 * np.random.randn(num_inputs, input_size)
    y = np.array([0, 1, 2, 2, 1])
    return X, y

#########################################

def test_2layer_net():
    params = init_toy_model()
    X, y = init_toy_data()
    Y_enc = ut.encode_labels(y)
    # Make the net
    layer_1 = layers.Linear(*params['W1'].T.shape,reg='frob',reg_param=0.05,init_vals=(params['W1'].T,params['b1'].ravel()))
    act_1 = layers.Relu()
    layer_2 = layers.Linear(*params['W2'].T.shape,reg='frob',reg_param=0.05,init_vals=(params['W2'].T,params['b2'].ravel()))
    net_2 = nn.Network([layer_1,act_1,layer_2], ls.CrossEntropy(), optim.SGD(lr=1e-5))
    scores = net_2.forward(X)
    correct_scores = np.asarray([[-1.07260209,  0.05083871, -0.87253915],
                                 [-2.02778743, -0.10832494, -1.52641362],
                                 [-0.74225908,  0.15259725, -0.39578548],
                                 [-0.38172726,  0.10835902, -0.17328274],
                                 [-0.64417314, -0.18886813, -0.41106892]])
    diff = np.sum(np.abs(scores - correct_scores))
    assert(np.isclose(diff,0.0,atol=1e-6))
    loss = net_2.loss(X,Y_enc)
    correct_loss = 1.071696123862817
    assert(np.isclose(loss,correct_loss,atol=1e-8))

def test_2layer_grad():
    params = init_toy_model()
    X, y = init_toy_data()
    Y_enc = ut.encode_labels(y)
    # Make the net
    layer_1 = layers.Linear(*params['W1'].T.shape,reg='frob',reg_param=0.05,init_vals=(params['W1'].T,params['b1'].ravel()))
    act_1 = layers.Relu()
    layer_2 = layers.Linear(*params['W2'].T.shape,reg='frob',reg_param=0.05,init_vals=(params['W2'].T,params['b2'].ravel()))
    net_2 = nn.Network([layer_1,act_1,layer_2], ls.CrossEntropy(), optim.SGD(lr=1e-5))
    loss = net_2.loss(X,Y_enc)
    net_2.backward()

    def f_change_param(param_name, U):
        if param_name==3:
            net_2.layers[0].params['b'] = U
        if param_name==2:
            net_2.layers[0].params['W'] = U
        if param_name==1:
            net_2.layers[2].params['b'] = U
        if param_name==0:
            net_2.layers[2].params['W'] = U
        return net_2.loss(X, Y_enc)

    rel_errs = np.empty(4)
    for param_name in range(4):
        f = lambda U: f_change_param(param_name, U)
        if param_name==3:
            pass_pars = net_2.layers[0].params['b']
        if param_name==2:
            pass_pars = net_2.layers[0].params['W']
        if param_name==1:
            pass_pars = net_2.layers[2].params['b']
        if param_name==0:
            pass_pars = net_2.layers[2].params['W']
        param_grad_num = dutil.grad_check(f, pass_pars, epsilon=1e-5)
        rel_errs[param_name] = ut.rel_error(param_grad_num, net_2.grads[param_name])
    assert(np.allclose(rel_errs,np.zeros(4),atol=1e-7))

test_CrossEntropyLoss()
test_CrossEntropy_Linear_Grad()
test_2layer_net()
test_2layer_grad()
