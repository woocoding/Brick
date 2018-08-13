import numpy as np


class LossFunc(object):

    def expression(self, yhat, y):
        raise NotImplementedError

    def derivative(self, yhat, y):
        raise NotImplementedError

    def __call__(self, yhat, y):
        loss = self.expression(yhat, y)
        dyhat = self.derivative(yhat, y)
        return loss, dyhat


class CrossEntropyLoss(LossFunc):

    def expression(self, yhat, y):
        loss = np.sum(y*np.log(yhat) + (1-y)*np.log(1-yhat), axis=0)
        loss = -np.mean(loss)
        return loss

    def derivative(self, yhat, y):
        dyhat = (yhat - y) / ((yhat*(1-yhat)) + 1e-8)
        return dyhat


class MSELoss(LossFunc):

    def expression(self, yhat, y):
        loss = np.mean(np.sum(0.5*((yhat - y)**2), axis=0))
        return loss

    def derivative(self, yhat, y):
        dyhat = yhat - y
        return dyhat
