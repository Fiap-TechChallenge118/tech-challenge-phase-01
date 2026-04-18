"""Exemplos de uso da Churn Prediction API via httpx.

Pré-requisito: API rodando em http://localhost:8000 (make run)

Executar:
    python examples/api_client.py
"""

import httpx

BASE_URL = "http://localhost:8000"

# --- Perfis de exemplo ---
HIGH_CHURN_RISK = {
    "gender": "Female",
    "Senior Citizen": "Yes",
    "partner": "No",
    "dependents": "No",
    "Phone Service": "Yes",
    "Multiple Lines": "Yes",
    "Internet Service": "Fiber optic",
    "Online Security": "No",
    "Online Backup": "No",
    "Device Protection": "No",
    "Tech Support": "No",
    "Streaming TV": "Yes",
    "Streaming Movies": "Yes",
    "contract": "Month-to-month",
    "Paperless Billing": "Yes",
    "Payment Method": "Electronic check",
    "Tenure Months": 2,
    "Monthly Charges": 95.5,
    "Total Charges": 191.0,
}

LOW_CHURN_RISK = {
    "gender": "Male",
    "Senior Citizen": "No",
    "partner": "Yes",
    "dependents": "Yes",
    "Phone Service": "Yes",
    "Multiple Lines": "No",
    "Internet Service": "DSL",
    "Online Security": "Yes",
    "Online Backup": "Yes",
    "Device Protection": "Yes",
    "Tech Support": "Yes",
    "Streaming TV": "No",
    "Streaming Movies": "No",
    "contract": "Two year",
    "Paperless Billing": "No",
    "Payment Method": "Bank transfer (automatic)",
    "Tenure Months": 60,
    "Monthly Charges": 55.0,
    "Total Charges": 3300.0,
}


def main() -> None:
    with httpx.Client(base_url=BASE_URL) as client:
        # Health check
        r = client.get("/health")
        print(f"[health] {r.json()}")

        # Predições
        for label, payload in [("alto risco", HIGH_CHURN_RISK), ("baixo risco", LOW_CHURN_RISK)]:
            r = client.post("/predict", json=payload)
            r.raise_for_status()
            result = r.json()
            print(
                f"[predict] {label} → probability={result['churn_probability']:.4f}"
                f"  prediction={result['churn_prediction']}"
            )


if __name__ == "__main__":
    main()
