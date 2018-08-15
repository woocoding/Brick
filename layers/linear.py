import numpy as np

from .layer import Layer


class Linear(Layer):

    def __init__(self, input_size, output_size):
        epsilon = np.sqrt(1/input_size)
        # initialize the weights matrices with random values
        self.w = np.random.randn(output_size, input_size)*epsilon
        # initialize the bias matrices with zero values
        self.b = np.zeros((output_size, 1))

    def forward(self, x):
        # cache input data
        self.cache["x"] = np.copy(x)
        y = np.dot(self.w, x) + self.b

        return y

    def backward(self, dy, alpha, **kw):
        m = dy.shape[1]
        # back-propagation
        dx = np.dot(self.w.T, dy)
        # update w and b
        dw = 1/m*np.dot(dy, self.cache["x"].T) 
        db = np.mean(dy, axis=1, keepdims=True)

        reg_func = kw.get("reg_func")
        # if reg_fucn is not None, w is regularized
        if reg_func:
            dw += reg_func(self.w)

        if kw.get("optimizer") and not self.__dict__.get("optimizer"):
            optimizer = kw.get("optimizer")
            self.optimizer = optimizer.bind_new_instance(dw.shape, db.shape)
        dw, db = self.optimizer(dw, db)

        self.w -= alpha*dw
        self.b -= alpha*db

        return dx

    def __getattr__(self, name):

        if name == "optimizer":
            def f(x, y):
                return x, y
            return f
        else:
            raise AttributeError("")
