"""Exercise: ClinicalPredictor (prediction only, no training)."""
import numpy as np


class ClinicalPredictor:
    """Uses an already-trained model to produce a diagnosis.

    Works with ANY model that has a predict() method (duck typing /
    polymorphism): LogisticRegression, NeuralNetwork, scikit-learn, ...
    """

    def __init__(self, model, threshold=0.5, scaler=None, feature_names=None):
        if not hasattr(model, "predict"):
            raise TypeError("The model must have a predict() method.")
        self.model = model
        self.threshold = threshold
        self.scaler = scaler              # optional: same scaler as in training
        self.feature_names = feature_names  # optional: to accept dict input

    def _to_array(self, patient_data):
        if isinstance(patient_data, dict):
            if self.feature_names is None:
                raise ValueError("feature_names is required for dict input.")
            patient_data = [patient_data[name] for name in self.feature_names]
        x = np.asarray(patient_data, dtype=float).reshape(1, -1)
        if self.scaler is not None:
            x = self.scaler.transform(x)
        return x

    def probability(self, patient_data):
        prediction = self.model.predict(self._to_array(patient_data))
        return float(np.asarray(prediction).ravel()[0])

    def diagnose(self, patient_data):
        """Return 'Infecté' if prediction >= threshold, else 'Sain'."""
        return "Infecté" if self.probability(patient_data) >= self.threshold else "Sain"
