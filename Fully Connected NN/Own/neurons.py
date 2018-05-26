import numpy as np 

def affine_forward(x, w, b):   # Computes the forward pass for an affine (fully-connected) layer.
    #getting the shape of the inout in order to change it 
    N = x.shape[0] 
    D = np.prod(x.shape[1:])
    x2 = np.reshape(x, (N, D))
    #perform the multiplication
    out = np.dot(x2, w) + b
    cache = (x, w, b)
    return out, cache

def affine_backward(dout, cache): # switcher
    x, w, b = cache
    dx = np.dot(dout, w.T).reshape(x.shape)
    dw = np.dot(x.reshape(x.shape[0], np.prod(x.shape[1:])).T, dout)
    db = np.sum(dout, axis=0)
    return dx, dw, db

#Leaky relu activation function forward and backward paths
def relu_forward(x):
    out = np.maximum(0.1*x, x)
    cache = x
    return out, cache

def relu_backward(dout, cache):
    x = cache
    dx = np.array(dout, copy=True)
    dx[x <= 0.1*x] = 0
    return dx


# I used the model provided by the link below to implement batch normalization forward and backward baths
    # https://kratzert.github.io/2016/02/12/understanding-the-gradient-flow-through-the-batch-normalization-layer.html
    
def batchnorm_forward(x, gamma, beta, bn_param):
    mode = bn_param['mode']
    eps = bn_param.get('eps', 1e-5)
    momentum = bn_param.get('momentum', 0.9)
    N, D = x.shape
    running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))
    out, cache = None, None
    if mode == 'train':
        # Mean
        mu = np.mean(x, axis=0)
        # Variance
        var = 1 / float(N) * np.sum((x - mu) ** 2, axis=0)
        # Normalized Data
        x_hat = (x - mu) / np.sqrt(var + eps)
        # Scale and Shift
        y = gamma * x_hat + beta
        out = y
        # Make the record of means and variances in running parameters
        running_mean = momentum * running_mean + (1 - momentum) * mu
        running_var = momentum * running_var + (1 - momentum) * var
        cache = (x_hat, mu, var, eps, gamma, beta, x)

    elif mode == 'test':
        # Normalized Data
        x_hat = (x - running_mean) / np.sqrt(running_var + eps)
        # Scale and Shift
        y = gamma * x_hat + beta
        out = y
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)
    bn_param['running_mean'] = running_mean
    bn_param['running_var'] = running_var
    return out, cache


def batchnorm_backward(dout, cache):
    x_hat, mu, var, eps, gamma, beta, x = cache
    N, D = dout.shape
    dbeta = np.sum(dout, axis=0)
    dgamma = np.sum(dout * x_hat, axis=0)
    dx_hat = dout * gamma
    dxmu1 = dx_hat * 1 / np.sqrt(var + eps)
    divar = np.sum(dx_hat * (x - mu), axis=0)
    dvar = divar * -1 / 2 * (var + eps) ** (-3/2)
    dsq = 1 / N * np.ones((N, D)) * dvar
    dxmu2 = 2 * (x - mu) * dsq
    dx1 = dxmu1 + dxmu2
    dmu = -1 * np.sum(dxmu1 + dxmu2, axis=0)
    dx2 = 1 / N * np.ones((N, D)) * dmu
    dx = dx1 + dx2
    return dx, dgamma, dbeta

# loss function
def softmax_loss(x, y):
    shifted_logits = x - np.max(x, axis=1, keepdims=True)
    Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
    log_probs = shifted_logits - np.log(Z)
    probs = np.exp(log_probs)
    N = x.shape[0]
    loss = -np.sum(log_probs[np.arange(N), y]) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx


#connnecting all the layers together
#connection the affine forward to relu forward 
def affine_relu_forward(x, w, b):
    a, fc_cache = affine_forward(x, w, b)
    out, relu_cache = relu_forward(a)
    cache = (fc_cache, relu_cache)
    return out, cache
#connecting the affine to relu backward
def affine_relu_backward(dout, cache):
    fc_cache, relu_cache = cache
    da = relu_backward(dout, relu_cache)
    dx, dw, db = affine_backward(da, fc_cache)
    return dx, dw, db
#connecting the affine to batch normlization and then to relu in the forward path 
def affine_bn_relu_forward(x, w, b, gamma, beta, bn_param):
        out1, fc_cache = affine_forward(x, w, b)
        out2, bn_cache = batchnorm_forward(out1, gamma, beta, bn_param)
        out3, relu_cache = relu_forward(out2)
        cache = (fc_cache, bn_cache, relu_cache)
        return out3, cache
#connecting the affine to batch normlization and then to relu in the backward path
def affine_bn_relu_backward(dout, cache):
        fc_cache, bn_cache, relu_cache = cache
        d1 = relu_backward(dout, relu_cache)
        d2, dgamma, dbeta = batchnorm_backward(d1, bn_cache)
        d3, dw, db = affine_backward(d2, fc_cache)
        return d3, dw, db, dgamma, dbeta
