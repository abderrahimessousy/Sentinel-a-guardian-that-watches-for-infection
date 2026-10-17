"""Abstract base class shared by every model of the project."""
from abc import ABC, abstractmethod

import numpy as np


class Model(ABC):
    """Common interface: any model must implement fit() and predict().

    predict() returns probabilities in [0, 1], so that the clinical
    threshold (0.5) has a meaning for every model.
    """

    def __init__(self):
        self.loss_history = []

    @abstractmethod
    def fit(self, X, y, optimizer, epochs=200):
        """Train the model on (X, y)."""

    @abstractmethod
    def predict(self, X):
        """Return the probability of infection for each row of X."""

    def predict_label(self, X, threshold=0.5):
        return (self.predict(X) >= threshold).astype(int)

    @staticmethod
    def _sigmoid(z):
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    @staticmethod
    def _bce(y, p):
        eps = 1e-12
        p = np.clip(p, eps, 1 - eps)
        return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
