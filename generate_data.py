"""Generate a synthetic patient dataset -> data/patient_data.csv."""
import csv
import os

import numpy as np

COLUMNS = ["age", "temperature", "cough", "fatigue", "oxygen_saturation",
           "loss_smell", "contact_infected", "heart_rate", "infected"]


def generate(n=1000, seed=42):
    rng = np.random.default_rng(seed)
    age = rng.integers(5, 90, n)
    cough = rng.binomial(1, 0.4, n)
    fatigue = rng.binomial(1, 0.4, n)
    loss_smell = rng.binomial(1, 0.2, n)
    contact = rng.binomial(1, 0.3, n)
    temperature = rng.normal(37.2, 0.6, n)
    oxygen = rng.normal(97, 2, n)
    heart_rate = rng.normal(78, 10, n)

    # hidden rule: infection probability depends on symptoms and contact
    z = (-3.2 + 0.015 * (age - 45) + 1.1 * (temperature - 37.2) + 1.0 * cough
         + 0.8 * fatigue + 1.8 * loss_smell + 1.6 * contact
         - 0.35 * (oxygen - 97) + 0.02 * (heart_rate - 78))
    p = 1 / (1 + np.exp(-z))
    infected = rng.binomial(1, p)

    # realistic: infected patients tend to have fever / lower oxygen
    temperature = temperature + infected * rng.normal(0.5, 0.3, n)
    oxygen = oxygen - infected * rng.normal(1.5, 1.0, n)

    rows = zip(age, np.round(temperature, 1), cough, fatigue,
               np.round(np.clip(oxygen, 80, 100), 1), loss_smell, contact,
               np.round(heart_rate).astype(int), infected)
    return list(rows)


if __name__ == "__main__":
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "patient_data.csv")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    rows = generate()
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(COLUMNS)
        w.writerows(rows)
    print(f"{len(rows)} patients written to {path}")
