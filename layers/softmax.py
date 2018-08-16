import numpy as np

from .layer import Layer


class Softmax(Layer):
    
    def forward(self, x, **kw):
        x = x - np.max(x, axis=0)
        x_exp = np.exp(x)
        # denominator
        base = np.sum(x_exp, axis=0, keepdims=True)
        y = x_exp / base
        self.cache["y"] = np.copy(y)
        return y

    def backward(self, dy, **kw):
        
        dx = dy*self.cache["y"] + self.cache["y"]
        return dx