"""Dataset class: loads patient_data.csv and prepares train/test splits."""
import csv

import numpy as np

from utils.preprocessing import StandardScaler, train_test_split


class PatientDataset:
    TARGET = "infected"

    def __init__(self, path):
        self.path = path
        self.feature_names = []
        self.X = None
        self.y = None
        self.scaler = StandardScaler()
        self._load()

    def _load(self):
        with open(self.path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            self.feature_names = [c for c in reader.fieldnames if c != self.TARGET]
            rows = list(reader)
        self.X = np.array([[float(r[c]) for c in self.feature_names] for r in rows])
        self.y = np.array([int(r[self.TARGET]) for r in rows])

    def __len__(self):
        return len(self.y)

    def split(self, test_size=0.2, seed=42):
        """Return scaled X_train, X_test, y_train, y_test (scaler fit on train only)."""
        X_tr, X_te, y_tr, y_te = train_test_split(self.X, self.y, test_size, seed)
        X_tr = self.scaler.fit_transform(X_tr)
        X_te = self.scaler.transform(X_te)
        return X_tr, X_te, y_tr, y_te
