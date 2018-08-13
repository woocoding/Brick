import numpy as np

from .layer import Layer


class Sigmoid(Layer):

    def forward(self, x):
        y = 1 / (1 + np.exp(-x))
        # cache g(x)
        self.cache["y"] = np.copy(y)
        return y

    def backward(self, dy, *args, **kw):
        y = self.cache["y"]
        return y*(1-y)*dy
