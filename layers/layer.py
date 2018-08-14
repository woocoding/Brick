from collections import OrderedDict

import numpy as np

from ..loss import LossFunc
from .misc import regulization


class Layer(object):

    def __new__(cls, *args, **kw):

        instance = super().__new__(cls)
        instance.__dict__["_layers"] = OrderedDict()
        instance.__dict__["cache"] = {}
        instance.__dict__["_criterion"] = None
        return instance

    def forward(self, x, **kw):

        for name in self._layers.keys():
            x = self._layers[name].forward(x, **kw)
        return x

    def backward(self, dy, **kw):

        keys = list(self._layers.keys())
        keys.reverse()
        for name in keys:
            dy = self._layers[name].backward(dy, **kw)
        return dy

    @property
    def criterion(self):

        if self._criterion == None:
            raise AttributeError("criterion is None")
        else:
            return self._criterion

    @criterion.setter
    def criterion(self, value):
        
        if isinstance(value, LossFunc):
            self._criterion = value
        else:
            raise ValueError("Must assign a LossFunc instance to criterion")

    def optimize(self, x, y, **kw):
        # forward propagation
        yhat = self.forward(x)
        loss, dyhat = self.criterion(yhat, y)
        reg = kw.get("reg")
        if reg != None:
            reg_loss = regulization(reg, self._layers)
            loss += reg_loss

        alpha = kw.get("alpha")
        if alpha == None:
            raise ValueError("Learning rate (alpha) is None")
        # backward propagation
        self.backward(dyhat, **kw)
        return yhat, loss 


    def __setattr__(self, name, value):

        if isinstance(value, Layer):
            self._layers[name] = value
        else:
            object.__setattr__(self, name, value)

    def __call__(self, x):

        return self.forward(x)
