"""Small neural network (1 hidden layer, ReLU, sigmoid output) in NumPy."""
import numpy as np

from core.model import Model


class NeuralNetwork(Model):
    def __init__(self, hidden_size=16, seed=42):
        super().__init__()
        self.hidden_size = hidden_size
        self.seed = seed
        self.params = None

    def _init_params(self, d):
        rng = np.random.default_rng(self.seed)
        self.params = {
            "W1": rng.normal(0, np.sqrt(2.0 / d), (d, self.hidden_size)),
            "b1": np.zeros(self.hidden_size),
            "W2": rng.normal(0, np.sqrt(2.0 / self.hidden_size), (self.hidden_size, 1)),
            "b2": np.zeros(1),
        }

    def _forward(self, X):
        z1 = X @ self.params["W1"] + self.params["b1"]
        a1 = np.maximum(0, z1)
        z2 = a1 @ self.params["W2"] + self.params["b2"]
        return z1, a1, self._sigmoid(z2).ravel()

    def fit(self, X, y, optimizer, epochs=200):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n, d = X.shape
        self._init_params(d)
        self.loss_history = []
        for _ in range(epochs):
            z1, a1, p = self._forward(X)
            dz2 = (p - y).reshape(-1, 1) / n
            grads = {
                "W2": a1.T @ dz2,
                "b2": dz2.sum(axis=0),
            }
            da1 = dz2 @ self.params["W2"].T
            dz1 = da1 * (z1 > 0)
            grads["W1"] = X.T @ dz1
            grads["b1"] = dz1.sum(axis=0)
            optimizer.step(self.params, grads)
            self.loss_history.append(self._bce(y, p))
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return self._forward(X)[2]
