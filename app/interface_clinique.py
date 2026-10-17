"""Command-line clinical interface: a doctor enters patient data."""

PROMPTS = {
    "age": "Âge (années)",
    "temperature": "Température (°C)",
    "cough": "Toux (1=oui, 0=non)",
    "fatigue": "Fatigue (1=oui, 0=non)",
    "oxygen_saturation": "Saturation en oxygène (%)",
    "loss_smell": "Perte d'odorat (1=oui, 0=non)",
    "contact_infected": "Contact avec un cas infecté (1=oui, 0=non)",
    "heart_rate": "Fréquence cardiaque (bpm)",
}


def ask_patient():
    patient = {}
    for key, label in PROMPTS.items():
        while True:
            try:
                patient[key] = float(input(f"{label} : "))
                break
            except ValueError:
                print("  Valeur invalide, réessayez.")
    return patient


def run_interface(predictor):
    print("=== Interface clinique - Diagnostic viral ===")
    while True:
        patient = ask_patient()
        prob = predictor.probability(patient)
        print(f"\n>>> Diagnostic : {predictor.diagnose(patient)}  (probabilité = {prob:.2%})\n")
        if input("Autre patient ? (o/n) : ").strip().lower() != "o":
            break
