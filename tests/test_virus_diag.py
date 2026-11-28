import numpy as np

from core.dataset import PatientDataset
from core.logistic_regression import LogisticRegression


def test_dataset_loading(tmp_path):
    path = tmp_path / "patients.csv"
    path.write_text(
        "age,temperature,infected\n"
        "25,36.5,0\n"
        "40,39.0,1\n",
        encoding="utf-8"
    )

    dataset = PatientDataset(path)

    assert len(dataset) == 3
    assert dataset.X.shape == (2, 2)
    assert dataset.feature_names == ["age", "temperature"]
    assert np.array_equal(dataset.y, [0, 1])


def test_logistic_regression_initialization():
    model = LogisticRegression()
    assert model.params is None


def test_prediction_shape():
    model = LogisticRegression()
    model.params = {
        "w": np.array([0.5, 0.5]),
        "b": np.array([0.0])
    }

    X = np.array([[1, 2], [3, 4]])
    predictions = model.predict(X)

    assert predictions.shape == (2,)


def test_prediction_probabilities():
    model = LogisticRegression()
    model.params = {
        "w": np.array([1.0, 1.0]),
        "b": np.array([0.0])
    }

    X = np.array([[0, 0], [1, 2], [-1, -2]])
    predictions = model.predict(X)

    assert np.all(predictions >= 0)
    assert np.all(predictions <= 1)
    assert np.isclose(predictions[0], 0.5)

