import numpy as np

from .layer import Layer


class Linear(Layer):
    
    def __init__(self, input_size, output_size):
        epsilon = np.sqrt(2/input_size)
        # initialize the weights matrices with random values
        self.w = np.random.randn(output_size, input_size)*epsilon
        # initialize the bias matrices with zero values
        self.b = np.zeros((output_size, 1))
        
    def forward(self, x):
        # cache input data
        self.cache["x"] = np.copy(x)
        y = np.dot(self.w, x) + self.b
        
        return y

    def backward(self, dy, alpha):        
        m = dy.shape[1]
        # back-propagation
        dx = np.dot(self.w.T, dy)
        #update w and b
        dw = 1/m*np.dot(dy, self.cache["x"].T)
        db = np.mean(dy, axis=1, keepdims=True)
        self.w -= alpha*dw
        self.b -= alpha*db
        
        return dx
