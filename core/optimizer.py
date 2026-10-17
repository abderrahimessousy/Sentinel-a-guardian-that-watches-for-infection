"""Optimizers used to update model parameters."""
import numpy as np


class Optimizer:
    """Base class: step(params, grads) updates params in place."""

    def step(self, params, grads):
        raise NotImplementedError


class SGD(Optimizer):
    def __init__(self, lr=0.1):
        self.lr = lr

    def step(self, params, grads):
        for name in params:
            params[name] -= self.lr * grads[name]


class Adam(Optimizer):
    def __init__(self, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = {}
        self.v = {}
        self.t = 0

    def step(self, params, grads):
        self.t += 1
        for name in params:
            if name not in self.m:
                self.m[name] = np.zeros_like(params[name])
                self.v[name] = np.zeros_like(params[name])
            self.m[name] = self.beta1 * self.m[name] + (1 - self.beta1) * grads[name]
            self.v[name] = self.beta2 * self.v[name] + (1 - self.beta2) * grads[name] ** 2
            m_hat = self.m[name] / (1 - self.beta1 ** self.t)
            v_hat = self.v[name] / (1 - self.beta2 ** self.t)
            params[name] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
