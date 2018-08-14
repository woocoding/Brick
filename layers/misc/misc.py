import numpy as np


def regulization(reg, layers):

    def loop_layers(ufunc, layers):
        """ 遍历所有layer
        ufunc: 取决于什么正则化方式 
        """
        loss = 0
        for l in layers.values():
            if hasattr(l, "w"):
                loss += np.sum(ufunc(l.w))
            if len(l._layers) != 0:
                loss += loop_layers(ufunc, l._layers)
        
        return loss
    
    if reg == "L2":
        ufunc = np.square
        loss = 0.5*loop_layers(ufunc, layers)
    elif reg == "L1":
        ufunc = np.abs
        loss = loop_layers(ufunc, layers)

    return loss
