import numpy as np
from util import *
# do not include any more libraries here!
# do not put any code outside of functions!

############################## Q 2.1 ##############################
# initialize b to 0 vector
# b should be a 1D array, not a 2D array with a singleton dimension
# we will do XW + b.
# X be [Examples, Dimensions]
def initialize_weights(in_size,out_size,params,name=''):
    b = np.zeros((out_size,))
    limit = (6./(in_size+out_size))**(0.5)
    W = np.random.uniform(-limit,limit,size=(in_size,out_size))

    params['W' + name] = W
    params['b' + name] = b

############################## Q 2.2.1 ##############################
# x is a matrix
# a sigmoid activation function
def sigmoid(x):
    res = 1./(1.+np.exp(-x))
    return res

############################## Q 2.2.1 ##############################
def forward(X,params,name='',activation=sigmoid):
    """
    Do a forward pass

    Keyword arguments:
    X -- input vector [Examples x D]
    params -- a dictionary containing parameters
    name -- name of the layer
    activation -- the activation function (default is sigmoid)
    """
    pre_act, post_act = None, None
    # get the layer parameters
    W = params['W' + name]
    b = params['b' + name]

    pre_act = X@W+b
    post_act = activation(pre_act)

    # store the pre-activation and post-activation values
    # these will be important in backprop
    params['cache_' + name] = (X, pre_act, post_act)

    return post_act

############################## Q 2.2.2  ##############################
# x is [examples,classes]
# softmax should be done for each row
def softmax(x):
    res = np.zeros(x.shape)

    if x.ndim == 1:
        c = -np.max(x)
        s = np.exp(x+c)
        sum = np.sum(s)
        res = s/sum
    else:
        rows = x.shape[0]
        for i in range(rows):
            c = -np.max(x[i,:])
            s = np.exp(x[i,:]+c)
            sum = np.sum(s)
            res[i,:] = s/sum

    return res

############################## Q 2.2.3 ##############################
# compute total loss and accuracy
# y is size [examples,classes]
# probs is size [examples,classes]
def compute_loss_and_acc(y, probs):
    loss = 0.
    acc = 0.

    N,C = y.shape
    for i in range(N):
        L = -np.dot(y[i,:],np.log(probs[i,:]))
        loss += L

        pred = np.argmax(probs[i,:])
        actual = np.argmax(y[i,:])
        correct = int(pred == actual)
        acc += correct/N
    return loss, acc

############################## Q 2.3 ##############################
# we give this to you
# because you proved it
# it's a function of post_act
def sigmoid_deriv(post_act):
    res = post_act*(1.0-post_act)
    return res

def backwards(delta,params,name='',activation_deriv=sigmoid_deriv):
    """
    Do a backwards pass

    Keyword arguments:
    delta -- errors to backprop
    params -- a dictionary containing parameters
    name -- name of the layer
    activation_deriv -- the derivative of the activation_func
    """

    # everything you may need for this layer
    W = params['W' + name]
    b = params['b' + name]
    X, pre_act, post_act = params['cache_' + name]

    # do the derivative through activation first
    da = activation_deriv(post_act)

    # then compute the derivative W,b, and X
    grad_X = (delta*da).dot(W.T)
    grad_W = (X.T).dot(delta*da)
    grad_b = np.sum((delta*da).T,axis=1)

    # store the gradients
    params['grad_W' + name] = grad_W
    params['grad_b' + name] = grad_b
    return grad_X

############################## Q 2.4 ##############################
# split x and y into random batches
# return a list of [(batch1_x,batch1_y)...]
def get_random_batches(x,y,batch_size):
    batches = []
    while len(x) > 0:
        size = min(len(x),batch_size)
        batch_x = np.zeros((size,x.shape[1]))
        batch_y = np.zeros((size,y.shape[1]))
        for i in range(size):
            random_index = np.random.randint(0, len(x))
            batch_x[i,:] = x[random_index,:]
            batch_y[i,:] = y[random_index,:]
            x = np.delete(x, random_index, 0)
            y = np.delete(y, random_index, 0)
        batches.append((batch_x,batch_y))

    return batches
