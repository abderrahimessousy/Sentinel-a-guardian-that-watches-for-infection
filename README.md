# virus_diag

Virus diagnosis project (OOP applied to AI), NumPy only.

```
virus_diag/
├── data/        patient_data.csv
├── core/        dataset.py, model.py, logistic_regression.py,
│                neural_network.py, optimizer.py, clinical_predictor.py
├── pipeline/    trainer.py, evaluator.py
├── utils/       preprocessing.py, metrics.py
├── app/         api.py, interface_clinique.py
├── generate_data.py
└── main.py
```

## Run

```bash
pip install -r requirements.txt
python generate_data.py     # creates data/patient_data.csv
python main.py              # train + evaluate + demo diagnosis
python main.py --cli        # interactive interface
python main.py --api        # HTTP API on port 8000
```

API example:

```bash
curl -X POST localhost:8000/diagnose -d '{"age":62,"temperature":38.6,"cough":1,"fatigue":1,"oxygen_saturation":92.5,"loss_smell":1,"contact_infected":1,"heart_rate":98}'
```

## Exercise answers

1. `ClinicalPredictor` (core/clinical_predictor.py) receives a trained model and
   `diagnose(patient_data)` returns "Infecté" if prediction >= 0.5, else "Sain".
2. Concept: polymorphism (duck typing) - any model with `predict()` works.
3. Prediction is separated from training (Single Responsibility): safety, lighter
   deployment, reusability, testability, reproducibility.
