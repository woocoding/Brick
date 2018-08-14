import numpy as np

from .layer import Layer


class Linear(Layer):

    def __init__(self, input_size, output_size):
        epsilon = np.sqrt(1/input_size)
        # initialize the weights matrices with random values
        self.w = np.random.randn(output_size, input_size)*epsilon
        # initialize the bias matrices with zero values
        self.b = np.zeros((output_size, 1))

    def forward(self, x):
        # cache input data
        self.cache["x"] = np.copy(x)
        y = np.dot(self.w, x) + self.b

        return y

    def backward(self, dy, alpha, **kw):
        m = dy.shape[1]
        # back-propagation
        dx = np.dot(self.w.T, dy)
        # update w and b
        dw = 1/m*np.dot(dy, self.cache["x"].T)
        reg = kw.get("reg")
        lambd = kw.get("lambd")
        if reg == "L2":
            dw += lambd*self.w
        elif reg == "L1":
            dw += lambd*np.where(self.w>0, 1, -1)
            
        db = np.mean(dy, axis=1, keepdims=True)
        self.w -= alpha*dw
        self.b -= alpha*db

        return dx
