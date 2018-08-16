import numpy as np


class LossFunc(object):

    def __init__(self, reg=None, lambd=None):
        self.reg = reg
        self.lambd = lambd

    def expression(self, yhat, y):
        raise NotImplementedError

    def derivative(self, yhat, y):
        raise NotImplementedError

    def __call__(self, yhat, y, layers):
        loss = self.expression(yhat, y)
        # regularization term
        add_loss, reg_func = regularization(self.reg, self.lambd, layers)
        loss += add_loss
        dyhat = self.derivative(yhat, y)
        return loss, dyhat, reg_func


class CrossEntropyLoss(LossFunc):

    def expression(self, yhat, y):
        n = y.shape[0]
        eps = 1e-256
        if n == 1:
            loss = y*np.log(yhat+eps) + (1-y)*np.log(1-yhat+eps)
        else:
            loss = np.sum(y*np.log(yhat+eps), axis=0)
        loss = -np.mean(loss)
        return np.squeeze(loss)

    def derivative(self, yhat, y):
        n = y.shape[0]
        eps = 1e-256
        if n == 1:
            dyhat = (yhat - y) / ((yhat*(1-yhat)) + eps)
        else:
            dyhat = - y / (yhat+eps)
        
        return dyhat


class MSELoss(LossFunc):

    def expression(self, yhat, y):
        loss = np.mean(np.sum(0.5*((yhat - y)**2), axis=0))
        return np.squeeze(loss)

    def derivative(self, yhat, y):
        dyhat = yhat - y
        return dyhat

def regularization(reg, lambd, layers):
    """
    Retrun:
        loss: regularization loss
        reg_func: 用于更新w
    """
    if reg == None:
        return 0, None

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
        reg_func = lambda w: lambd*w
    elif reg == "L1":
        ufunc = np.abs
        loss = loop_layers(ufunc, layers)
        reg_func = lambda w: lambd*np.where(w>=0, 1, -1)

    return loss, reg_func