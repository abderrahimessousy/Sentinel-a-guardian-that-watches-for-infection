"""Logistic regression implemented with NumPy."""
import numpy as np

from core.model import Model


class LogisticRegression(Model):
    def __init__(self):
        super().__init__()
        self.params = None

    def fit(self, X, y, optimizer, epochs=200):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n, d = X.shape
        self.params = {"w": np.zeros(d), "b": np.zeros(1)}
        self.loss_history = []
        for _ in range(epochs):
            p = self._sigmoid(X @ self.params["w"] + self.params["b"])
            grads = {"w": X.T @ (p - y) / n, "b": np.array([np.mean(p - y)])}
            optimizer.step(self.params, grads)
            self.loss_history.append(self._bce(y, p))
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return self._sigmoid(X @ self.params["w"] + self.params["b"])
