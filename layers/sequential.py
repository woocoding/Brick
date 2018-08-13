import warnings

import numpy as np

from .layer import Layer


class Sequantial(Layer):

    def __new__(cls, *args):
        instance = super().__new__(cls, *args)
        for i, value in enumerate(args):
            name = "layer" + str(i)
            instance.__setattr__(name, value)
        return instance
    
    def add_layer(self, name, value):
        if name in self._layers:
            warnings.warn(f"Layer:{name} will be replaced")
        if isinstance(value, Layer):
            self._layers[name] = value