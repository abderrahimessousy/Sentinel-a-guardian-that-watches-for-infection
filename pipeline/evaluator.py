"""Evaluator: measures the quality of a trained model."""
from utils import metrics


class Evaluator:
    def __init__(self, threshold=0.5):
        self.threshold = threshold

    def evaluate(self, model, X_test, y_test):
        y_pred = (model.predict(X_test) >= self.threshold).astype(int)
        tp, tn, fp, fn = metrics.confusion_matrix(y_test, y_pred)
        return {
            "accuracy": metrics.accuracy(y_test, y_pred),
            "precision": metrics.precision(y_test, y_pred),
            "recall": metrics.recall(y_test, y_pred),
            "f1": metrics.f1_score(y_test, y_pred),
            "confusion": {"TP": tp, "TN": tn, "FP": fp, "FN": fn},
        }

    @staticmethod
    def report(name, results):
        c = results["confusion"]
        print(f"  {name:<20} acc={results['accuracy']:.3f}  prec={results['precision']:.3f}  "
              f"rec={results['recall']:.3f}  f1={results['f1']:.3f}  "
              f"(TP={c['TP']} TN={c['TN']} FP={c['FP']} FN={c['FN']})")
