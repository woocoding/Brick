import numpy as np

from .layer import Layer


class Tanh(Layer):

    def forward(self, x, **kw):
        y = np.tanh(x)
        # cache g(x)
        self.cache["y"] = np.copy(y)
        return y

    def backward(self, dy, **kw):
        y = self.cache["y"]
        return (1 - y**2)*dy
