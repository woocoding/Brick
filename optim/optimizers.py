import numpy as np


class EWMA(object):
    
    def __init__(self, beta, shape):
        self.beta = beta
        self.e = np.zeros(shape)
        # record the times of calculating EWMA
        self.t = 0

    def __call__(self, y, corrected=False):
        self.e = self.beta*self.e + (1-self.beta)*y
        self.t += 1
        if corrected:
            e = self.e / (1-self.beta**self.t)
        else:
            e = np.copy(self.e)
        return e

class Optimizer(object):

    def bind_new_instance(self, shape):
        raise NotImplementedError

    def update_grad(self, dw, db):
        raise NotImplementedError

    def __call__(self, *arg, **kw):
        return self.update_grad(*arg, **kw)


class Momentum(Optimizer):

    def __init__(self, beta):
        self.beta = beta

    def bind_new_instance(self, dw_shape, db_shape):
        """
        shape is tuple 
        """
        isinstance = Momentum(self.beta)
        isinstance.ewma_dw = EWMA(self.beta, dw_shape)
        isinstance.ewma_db = EWMA(self.beta, db_shape)
        return isinstance

    def update_grad(self, dw, db):
        v_dw = self.ewma_dw(dw)
        v_db = self.ewma_db(db)
        return v_dw, v_db


class RMSprob(Optimizer):

    def __init__(self, beta):
        self.beta = beta

    def bind_new_instance(self, dw_shape, db_shape):
        """
        shape is tuple 
        """
        isinstance = RMSprob(self.beta)
        isinstance.ewma_dw2 = EWMA(self.beta, dw_shape)
        isinstance.ewma_db2 = EWMA(self.beta, db_shape)
        return isinstance

    def update_grad(self, dw, db):
        s_dw = self.ewma_dw2(dw**2)
        s_db = self.ewma_db2(db**2)
        eps = 1e-8
        dw /= (np.sqrt(s_dw) + eps)
        db /= (np.sqrt(s_db) + eps)
        return dw, db


class Adam(Optimizer):

    def __init__(self, betas):
        self.betas = betas

    def bind_new_instance(self, dw_shape, db_shape):
        """
        shape is tuple 
        """
        isinstance = Adam(self.betas)
        isinstance.ewma_dw = EWMA(self.betas[0], dw_shape)
        isinstance.ewma_db = EWMA(self.betas[0], db_shape)
        isinstance.ewma_dw2 = EWMA(self.betas[1], dw_shape)
        isinstance.ewma_db2 = EWMA(self.betas[1], db_shape)
        return isinstance

    def update_grad(self, dw, db):

        v_dw = self.ewma_dw(dw, corrected=True)
        v_db = self.ewma_db(db, corrected=True)
        s_dw = self.ewma_dw2(dw**2, corrected=True)
        s_db = self.ewma_db2(db**2, corrected=True)
        eps = 1e-16

        return v_dw/(np.sqrt(s_dw) + eps), v_db/(np.sqrt(s_db) + eps)
