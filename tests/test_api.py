"""API tests: smoke test de /health e teste de contrato do /predict."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api.main import app

VALID_PAYLOAD = {
    "gender": "Male",
    "Senior Citizen": "No",
    "partner": "Yes",
    "dependents": "No",
    "Phone Service": "Yes",
    "Multiple Lines": "No",
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
    "Monthly Charges": 70.5,
    "Total Charges": 141.0,
}


@pytest.fixture
def client(tmp_path, monkeypatch):
    """TestClient com cwd na raiz do projeto para que os artefatos sejam encontrados.

    O lifespan da API usa paths relativos (data/processed/), então o cwd precisa
    ser a raiz do projeto no momento em que o TestClient é instanciado e o lifespan roda.
    """
    monkeypatch.chdir(Path(__file__).parent.parent)
    with TestClient(app) as c:
        yield c


def test_health_returns_200(client):
    """GET /health deve retornar 200 com status ok."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_online_returns_valid_response(client):
    """POST /predict com payload válido deve retornar 200 com os campos corretos."""
    response = client.post("/predict", json=VALID_PAYLOAD)
    assert response.status_code == 200

    body = response.json()
    assert "churn_probability" in body
    assert "churn_prediction" in body
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert isinstance(body["churn_prediction"], bool)


def test_predict_online_rejects_invalid_payload(client):
    """POST /predict com payload incompleto deve retornar 422 (Unprocessable Entity)."""
    response = client.post("/predict", json={"gender": "Male"})  # faltam campos obrigatórios
    assert response.status_code == 422


def test_predict_online_rejects_invalid_categorical(client):
    """POST /predict com valor categórico inválido deve retornar 422."""
    payload = {**VALID_PAYLOAD, "gender": "Unknown"}  # "Unknown" não é valor válido
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_online_rejects_inconsistent_services(client):
    """POST /predict com dependências de serviço inconsistentes deve retornar 422."""
    # phone_service=No mas multiple_lines=Yes é inconsistente
    payload = {**VALID_PAYLOAD, "Phone Service": "No", "Multiple Lines": "Yes"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_batch_lookup_returns_503_without_db(client, monkeypatch):
    """GET /predict/batch sem scores.db deve retornar 503 (banco não encontrado)."""
    import api.main
    monkeypatch.setattr(api.main, "SCORES_DB_PATH", Path("/nonexistent/scores.db"))
    response = client.get("/predict/batch?customer_id=7590-VHVEG")
    assert response.status_code in (404, 503)
