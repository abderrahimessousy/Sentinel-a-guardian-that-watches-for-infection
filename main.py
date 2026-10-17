"""virus_diag - full pipeline.

Usage:
    python generate_data.py        # create data/patient_data.csv
    python main.py                 # train, evaluate, demo diagnosis
    python main.py --cli           # interactive clinical interface
    python main.py --api           # start the HTTP API
"""
import os
import sys

from app.api import run_server
from app.interface_clinique import run_interface
from core.clinical_predictor import ClinicalPredictor
from core.dataset import PatientDataset
from core.logistic_regression import LogisticRegression
from core.neural_network import NeuralNetwork
from core.optimizer import Adam
from pipeline.evaluator import Evaluator
from pipeline.trainer import Trainer

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "patient_data.csv")


def main():
    dataset = PatientDataset(DATA)
    X_train, X_test, y_train, y_test = dataset.split()
    print(f"{len(dataset)} patients | {len(dataset.feature_names)} features | "
          f"{len(y_train)} train / {len(y_test)} test\n")

    print("Training (Trainer):")
    models = {
        "LogisticRegression": Trainer(LogisticRegression(), Adam(lr=0.05), 300).train(X_train, y_train),
        "NeuralNetwork": Trainer(NeuralNetwork(hidden_size=16), Adam(lr=0.02), 400).train(X_train, y_train),
    }

    print("\nEvaluation (Evaluator):")
    evaluator = Evaluator()
    for name, model in models.items():
        evaluator.report(name, evaluator.evaluate(model, X_test, y_test))

    print("\nDiagnosis (ClinicalPredictor) - same class, different models:")
    patient = {"age": 62, "temperature": 38.6, "cough": 1, "fatigue": 1,
               "oxygen_saturation": 92.5, "loss_smell": 1, "contact_infected": 1, "heart_rate": 98}
    healthy = {"age": 25, "temperature": 36.8, "cough": 0, "fatigue": 0,
               "oxygen_saturation": 98.5, "loss_smell": 0, "contact_infected": 0, "heart_rate": 70}
    predictors = {}
    for name, model in models.items():
        predictors[name] = ClinicalPredictor(model, scaler=dataset.scaler,
                                             feature_names=dataset.feature_names)
        for label, p in (("symptomatic", patient), ("healthy-looking", healthy)):
            print(f"  {name:<20} {label:<16} -> {predictors[name].diagnose(p)} "
                  f"(p={predictors[name].probability(p):.2f})")

    predictor = predictors["LogisticRegression"]
    if "--cli" in sys.argv:
        run_interface(predictor)
    elif "--api" in sys.argv:
        run_server(predictor)


if __name__ == "__main__":
    main()
