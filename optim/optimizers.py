import numpy as np


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
        isinstance.v_dw = np.zeros(dw_shape)
        isinstance.v_db = np.zeros(db_shape)
        return isinstance

    def calc_past(self, dw, db):
        self.v_dw = self.beta * self.v_dw + (1 - self.beta)*dw
        self.v_db = self.beta * self.v_db + (1 - self.beta)*db

    def update_grad(self, dw, db):
        self.calc_past(dw, db)
        return self.v_dw.copy(), self.v_db.copy()


class RMSprob(Optimizer):

    def __init__(self, beta):
        self.beta = beta

    def bind_new_instance(self, dw_shape, db_shape):
        """
        shape is tuple 
        """
        isinstance = RMSprob(self.beta)
        isinstance.s_dw = np.zeros(dw_shape)
        isinstance.s_db = np.zeros(db_shape)
        return isinstance

    def calc_past(self, dw, db):
        self.s_dw = self.beta * self.s_dw + (1 - self.beta)*np.square(dw)
        self.s_db = self.beta * self.s_db + (1 - self.beta)*np.square(db)

    def update_grad(self, dw, db):
        self.calc_past(dw, db)
        dw /= np.sqrt(self.s_dw) + 1e-8
        db /= np.sqrt(self.s_db) + 1e-8
        return dw, db


class Adam(Optimizer):

    def __init__(self, betas):
        self.betas = betas

    def bind_new_instance(self, dw_shape, db_shape):
        """
        shape is tuple 
        """
        isinstance = Adam(self.betas)
        isinstance.mom = Momentum(
            self.betas[0]).bind_new_instance(dw_shape, db_shape)
        isinstance.rms = RMSprob(
            self.betas[1]).bind_new_instance(dw_shape, db_shape)
        return isinstance

    def update_grad(self, dw, db):

        self.mom.calc_past(dw, db)
        self.rms.calc_past(dw, db)
        v_dw, v_db = self.mom.v_dw.copy(), self.mom.v_db.copy()
        s_dw, s_db = self.rms.s_dw.copy(), self.rms.s_db.copy()
        v_dw /= (1 - self.betas[0])
        v_db /= (1 - self.betas[0])
        s_dw /= (1 - self.betas[1])
        s_db /= (1 - self.betas[1])

        assert v_dw.shape == dw.shape
        assert v_db.shape == db.shape
        assert s_dw.shape == dw.shape
        assert s_db.shape == db.shape

        return v_dw/(np.sqrt(s_dw) + 1e-16), v_db/(np.sqrt(s_db) + 1e-16)
