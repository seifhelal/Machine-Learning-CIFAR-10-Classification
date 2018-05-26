from Optim import adam 
import numpy as np
class hidden_layer(object):
    def __init__(self, inp, neurons):
        self.weights = np.random.randn(inp, neurons)/np.sqrt(inp/2.0)
        self.bias = np.zeros(neurons)
        self.w_opt = adam(self.weights)
        self.b_opt = adam(self.bias)
    def name (self):
        print("Hidden layer")
    def forward(self, X):
        self.inp = X
        h = X.reshape(X.shape[0], self.weights.shape[0]).dot(self.weights) + self.bias
        return h
    
    def backward(self, d):
        X = self.inp
        self.dW = X.reshape(X.shape[0], self.weights.shape[0]).T.dot(d)
        self.db = np.sum(d, axis = 0, keepdims=True)
        DH = d.dot(self.weights.T).reshape(X.shape)
        
        self.weights = self.w_opt.update(self.weights, self.dW)
        self.bias = self.b_opt.update(self.bias, self.db)
        return DH        

