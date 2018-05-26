import numpy as np
class adam(object):
    def __init__(self, inp):
        self.learning_rate = 1e-3
        self.beta1 = 0.9
        self.beta2 = 0.999 
        self.eps =  1e-8
        self.m = np.zeros_like(inp)
        self.v= np.zeros_like(inp)
        self.t = 0
        
    
    def update(self, x, dx):
        next_x = None
        self.t += 1
        self.m = self.beta1 * self.m
        temp = (1 - self.beta1) * dx
        self.m = self.m + temp
  
        self.v = self.beta2 * self.v + (1 - self.beta2) * (dx**2)

        # bias correction:
        mb = self.m / (1 - self.beta1**self.t)
        vb = self.v / (1 - self.beta2**self.t)

        next_x = -self.learning_rate * mb / (np.sqrt(vb) + self.eps) + x
        
        return next_x 

