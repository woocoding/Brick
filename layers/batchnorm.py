import numpy as np

from .layer import Layer
from ..utils import EWMA


class BatchNorm(Layer):
    
    def __init__(self, input_size, momentum=0.9, eps=1e-05):
        self.eps =eps
        shape = (input_size, 1)
        self.gamma = np.random.rand(*shape)
        self.beta = np.zeros(shape)
        self.mu_ewma = EWMA(momentum, shape)
        self.var_ewma = EWMA(momentum, shape)

    def forward(self, x, training):
        
        if training:
            mu = np.mean(x, axis=1, keepdims=True)
            var = np.var(x, axis=1, keepdims=True)

            assert mu.shape == (x.shape[0], 1)
            assert var.shape == (x.shape[0], 1)
            # EWMA
            self.mu_ewma(mu)
            self.mu_ewma(var)
            x_norm = (x - mu) / np.sqrt(var + self.eps)
            x_scaled = self.gamma*x_norm + self.beta
            # Cache
            self.cache["mu"] = mu
            self.cache["var"] = var
            self.cache["x"] = x
            self.cache["x_norm"] = x_norm
        else:
            m = x.shape[1]
            mu = self.mu_ewma.e
            var = self.var_ewma.e
            var *= m/(m-1) 
            x_norm = (x - mu) / np.sqrt(var + self.eps)
            x_scaled = self.gamma*x_norm + self.beta

        return x_scaled

    def backward(self, dy, alpha, **kw):
        
        eps = self.eps
        mu = self.cache["mu"]
        var = self.cache["var"]
        x = self.cache["x"]
        x_norm = self.cache["x_norm"]
        m = dy.shape[1]
        
        dbeta = np.sum(dy, axis=1, keepdims=True)
        dgamma = np.sum(dy*x_norm, axis=1, keepdims=True) 
        dx_norm = dy*self.gamma
        dx_self = dx_norm*(1/np.sqrt(var+eps))
        dmu = np.sum(dx_norm*(-1/np.sqrt(var+eps)), axis=1, keepdims=True)
        dvar = np.sum(dx_norm*0.5*(x-mu)*np.power(var+self.eps, -1.5), axis=1, keepdims=True)

        # Upadate gamma and beta
        self.gamma -= alpha*dgamma 
        self.beta -= alpha*dbeta 

        dx = dx_self + dmu*np.ones_like(dy)/m + dvar*2/m*(x-mu)

        return dx
        


