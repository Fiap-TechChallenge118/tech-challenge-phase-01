"""Schema tests: valida o DataFrame de entrada com pandera em dois estágios.

1. test_raw_dataset_schema   — valida o CSV após limpeza/normalização (antes do preprocessamento)
2. test_preprocessed_features_schema — valida o array numpy gerado pelo ColumnTransformer
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pandera as pa
import pytest
from pandera import Column, DataFrameSchema

RAW_CSV_PATH = Path("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")

# ---------------------------------------------------------------------------
# Schema do CSV limpo (formato estendido, após _clean / _normalize_columns)
# Cobre todas as 19 features usadas no treinamento + target
# ---------------------------------------------------------------------------
CHURN_SCHEMA = DataFrameSchema(
    columns={
        # --- Numéricas ---
        "Tenure Months":   Column(float, pa.Check.ge(0)),
        "Monthly Charges": Column(float, pa.Check.ge(0), nullable=True),
        "Total Charges":   Column(float, pa.Check.ge(0), nullable=True),

        # --- Target ---
        "Churn Value": Column(int, pa.Check.isin([0, 1])),

        # --- Demográficas ---
        "Gender":         Column(str, pa.Check.isin(["Male", "Female"])),
        "Senior Citizen": Column(str, pa.Check.isin(["Yes", "No"])),
        "Partner":        Column(str, pa.Check.isin(["Yes", "No"])),
        "Dependents":     Column(str, pa.Check.isin(["Yes", "No"])),

        # --- Telefonia ---
        "Phone Service": Column(str, pa.Check.isin(["Yes", "No"])),
        "Multiple Lines": Column(str, pa.Check.isin(["Yes", "No", "No phone service"])),

        # --- Internet ---
        "Internet Service":  Column(str, pa.Check.isin(["DSL", "Fiber optic", "No"])),
        "Online Security":   Column(str, pa.Check.isin(["Yes", "No", "No internet service"])),
        "Online Backup":     Column(str, pa.Check.isin(["Yes", "No", "No internet service"])),
        "Device Protection": Column(str, pa.Check.isin(["Yes", "No", "No internet service"])),
        "Tech Support":      Column(str, pa.Check.isin(["Yes", "No", "No internet service"])),
        "Streaming TV":      Column(str, pa.Check.isin(["Yes", "No", "No internet service"])),
        "Streaming Movies":  Column(str, pa.Check.isin(["Yes", "No", "No internet service"])),

        # --- Contrato e cobrança ---
        "Contract":          Column(str, pa.Check.isin(["Month-to-month", "One year", "Two year"])),
        "Paperless Billing": Column(str, pa.Check.isin(["Yes", "No"])),
        "Payment Method":    Column(str, pa.Check.isin([
            "Electronic check", "Mailed check",
            "Bank transfer (automatic)", "Credit card (automatic)",
        ])),
    },
    coerce=True,
)


@pytest.mark.skipif(not RAW_CSV_PATH.exists(), reason="CSV não encontrado — rode o download do dataset primeiro")
def test_raw_dataset_schema():
    """O dataset raw (após limpeza) deve ter todas as colunas com tipos e valores válidos."""
    from src.preprocessing import _clean

    df = pd.read_csv(RAW_CSV_PATH)
    df = _clean(df)

    validated = CHURN_SCHEMA.validate(df)

    assert len(validated) >= 5000, "Dataset deve ter pelo menos 5.000 registros"
    assert validated["Churn Value"].mean() > 0.20, "Taxa de churn esperada acima de 20%"
    assert validated["Churn Value"].mean() < 0.35, "Taxa de churn esperada abaixo de 35%"


@pytest.mark.skipif(not RAW_CSV_PATH.exists(), reason="CSV não encontrado — rode o download do dataset primeiro")
def test_preprocessed_features_schema():
    """O array pós-ColumnTransformer deve ter shape correto, sem NaN e valores numéricos finitos."""
    from src.preprocessing import load_and_split

    X_train, X_test, y_train, y_test, _ = load_and_split(str(RAW_CSV_PATH))

    # Shape: 46 features (3 numéricas escaladas + 43 colunas OHE)
    assert X_train.shape[1] == X_test.shape[1], "Train e test devem ter o mesmo número de features"
    assert X_train.shape[0] > X_test.shape[0], "Train deve ter mais amostras que test"

    # Sem NaN nem infinito após imputer + scaler
    assert not np.isnan(X_train).any(), "X_train contém NaN após preprocessamento"
    assert not np.isnan(X_test).any(),  "X_test contém NaN após preprocessamento"
    assert np.isfinite(X_train).all(),  "X_train contém valores infinitos"
    assert np.isfinite(X_test).all(),   "X_test contém valores infinitos"

    # Labels binários
    assert set(np.unique(y_train)).issubset({0, 1}), "y_train contém valores fora de {0, 1}"
    assert set(np.unique(y_test)).issubset({0, 1}),  "y_test contém valores fora de {0, 1}"

    # Proporção de churn preservada pela estratificação (entre 20% e 35% em ambos os splits)
    for name, y in [("y_train", y_train), ("y_test", y_test)]:
        rate = y.mean()
        assert 0.20 < rate < 0.35, f"Taxa de churn em {name} fora do esperado: {rate:.2%}"
