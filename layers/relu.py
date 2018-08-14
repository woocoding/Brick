import numpy as np

from .layer import Layer


class ReLU(Layer):

    def forward(self, x):
        y = np.maximum(0,x)
        # cache g(x)
        self.cache["y"] = np.copy(y)
        return y

    def backward(self, dy, **kw):
        y = self.cache["y"]
        dy = np.multiply(dy, np.int64(y > 0))
        return dy
