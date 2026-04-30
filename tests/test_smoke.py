"""Smoke tests: verifica que o ChurnMLP instancia, carrega artefatos e faz predições corretamente."""

from pathlib import Path

import joblib
import pytest
import torch

from src.model import ChurnMLP

ARTIFACTS_DIR    = Path("data/processed")
MODEL_PATH       = ARTIFACTS_DIR / "model.pth"
PREPROCESSOR_PATH = ARTIFACTS_DIR / "preprocessor.pkl"

# Feature row representando um cliente de alto risco (mês-a-mês, fiber optic, pouco tempo de casa)
_SAMPLE_ROW = {
    "Gender": "Male", "Senior Citizen": "No", "Partner": "No", "Dependents": "No",
    "Phone Service": "Yes", "Multiple Lines": "No",
    "Internet Service": "Fiber optic", "Online Security": "No", "Online Backup": "No",
    "Device Protection": "No", "Tech Support": "No",
    "Streaming TV": "Yes", "Streaming Movies": "Yes",
    "Contract": "Month-to-month", "Paperless Billing": "Yes",
    "Payment Method": "Electronic check",
    "Tenure Months": 2.0, "Monthly Charges": 70.5, "Total Charges": 141.0,
}


def test_mlp_forward_pass_shape():
    """O forward pass deve retornar logits com shape (batch_size, 1); sigmoid aplicado externamente."""
    model = ChurnMLP(input_dim=46, hidden_dims=[64, 32], dropout=0.0)
    model.eval()

    x = torch.randn(16, 46)
    with torch.no_grad():
        logits = model(x)
        probs = torch.sigmoid(logits)

    assert logits.shape == (16, 1), f"Shape esperado (16, 1), obtido {logits.shape}"
    assert probs.min() >= 0.0 and probs.max() <= 1.0, "Probabilidades fora do intervalo [0, 1]"


@pytest.mark.skipif(
    not (MODEL_PATH.exists() and PREPROCESSOR_PATH.exists()),
    reason="Artefatos não encontrados — rode src/pipeline.py primeiro",
)
def test_model_loads_and_predicts():
    """Carrega os artefatos treinados do disco e faz uma predição end-to-end sem erro.

    Verifica:
    - preprocessor.pkl e model.pth carregam sem exceção
    - A predição retorna um único float em [0, 1]
    - O modelo está em modo eval (Dropout desativado — resultado determinístico)
    """
    import pandas as pd

    preprocessor = joblib.load(PREPROCESSOR_PATH)

    input_dim = preprocessor.transform(pd.DataFrame([_SAMPLE_ROW])).shape[1]
    model = ChurnMLP(input_dim=input_dim, hidden_dims=[64, 32], dropout=0.3)
    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
    model.eval()

    X = preprocessor.transform(pd.DataFrame([_SAMPLE_ROW]))
    with torch.no_grad():
        prob = float(torch.sigmoid(model(torch.tensor(X, dtype=torch.float32))).squeeze())

    assert 0.0 <= prob <= 1.0, f"Probabilidade fora de [0, 1]: {prob}"

    # Resultado determinístico: duas chamadas consecutivas devem retornar o mesmo valor
    with torch.no_grad():
        prob2 = float(torch.sigmoid(model(torch.tensor(X, dtype=torch.float32))).squeeze())
    assert prob == prob2, "Predição não é determinística em modo eval"
