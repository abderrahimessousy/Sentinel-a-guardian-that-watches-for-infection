"""Trainer: the ONLY place where models are trained."""
from core.optimizer import Adam


class Trainer:
    def __init__(self, model, optimizer=None, epochs=300):
        self.model = model
        self.optimizer = optimizer or Adam(lr=0.05)
        self.epochs = epochs

    def train(self, X_train, y_train, verbose=True):
        self.model.fit(X_train, y_train, self.optimizer, self.epochs)
        if verbose:
            h = self.model.loss_history
            print(f"  [{type(self.model).__name__}] loss: {h[0]:.4f} -> {h[-1]:.4f}")
        return self.model
