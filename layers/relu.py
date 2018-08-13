import numpy as np

from .layer import Layer


class ReLU(Layer):

    def forward(self, x):
        x[x < 0] = 0
        y = x
        # cache g(x)
        self.cache["y"] = np.copy(y)
        return y

    def backward(self, dy, *args, **kw):
        y = self.cache["y"]
        return (y > 0)*dy
