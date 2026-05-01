"""Data loading and preprocessing pipeline."""

import logging

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logger = logging.getLogger(__name__)

TARGET = "Churn Value"

NUMERIC_FEATURES = ["Tenure Months", "Monthly Charges", "Total Charges"]
CATEGORICAL_FEATURES = [
    "Gender", "Senior Citizen", "Partner", "Dependents",
    "Phone Service", "Multiple Lines", "Internet Service",
    "Online Security", "Online Backup", "Device Protection",
    "Tech Support", "Streaming TV", "Streaming Movies",
    "Contract", "Paperless Billing", "Payment Method",
]


# Mapeamento da versão compacta do dataset IBM Telco (21 colunas, sem espaços)
# para o formato estendido esperado pelo projeto (33 colunas, com espaços).
_COMPACT_COLUMN_MAP = {
    "customerID": "CustomerID",
    "gender": "Gender",
    "SeniorCitizen": "Senior Citizen",
    "Partner": "Partner",
    "Dependents": "Dependents",
    "tenure": "Tenure Months",
    "PhoneService": "Phone Service",
    "MultipleLines": "Multiple Lines",
    "InternetService": "Internet Service",
    "OnlineSecurity": "Online Security",
    "OnlineBackup": "Online Backup",
    "DeviceProtection": "Device Protection",
    "TechSupport": "Tech Support",
    "StreamingTV": "Streaming TV",
    "StreamingMovies": "Streaming Movies",
    "Contract": "Contract",
    "PaperlessBilling": "Paperless Billing",
    "PaymentMethod": "Payment Method",
    "MonthlyCharges": "Monthly Charges",
    "TotalCharges": "Total Charges",
}


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza a versão compacta do CSV (21 colunas) para o formato estendido esperado.

    A versão compacta (Kaggle / IBM GitHub) usa camelCase sem espaços e uma coluna
    'Churn' com Yes/No. O projeto foi desenvolvido com o formato estendido (IBM Cloud)
    que usa espaços nos nomes e uma coluna 'Churn Value' com 0/1.
    Se o CSV já estiver no formato estendido, o DataFrame não é alterado.
    """
    if "tenure" in df.columns:
        df = df.rename(columns=_COMPACT_COLUMN_MAP)
        # SeniorCitizen vem como 0/1 inteiro na versão compacta — converter para Yes/No
        df["Senior Citizen"] = df["Senior Citizen"].map({0: "No", 1: "Yes"}).fillna(df["Senior Citizen"])
        # Churn vem como "Yes"/"No" — criar coluna numérica Churn Value (1/0)
        if "Churn" in df.columns and "Churn Value" not in df.columns:
            df["Churn Value"] = (df["Churn"] == "Yes").astype(int)
        # Também criar CustomerID se não existir (em algumas versões pode estar ausente)
        if "CustomerID" not in df.columns and "customerID" not in df.columns:
            df["CustomerID"] = range(len(df))
    return df


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    df = _normalize_columns(df)
    # Colunas de cobrança chegam como string no CSV — forçar conversão para float
    df["Total Charges"] = pd.to_numeric(df["Total Charges"], errors="coerce")
    df["Monthly Charges"] = pd.to_numeric(df["Monthly Charges"], errors="coerce")
    return df


def build_preprocessor() -> ColumnTransformer:
    """Return an unfitted ColumnTransformer for the Telco dataset."""
    numeric_pipe = Pipeline([
        # SimpleImputer preenche NaNs com a mediana — mais robusta que a média em presença de outliers
        ("imputer", SimpleImputer(strategy="median")),
        # StandardScaler centraliza (média=0) e normaliza (desvio=1)
        # sem isso, features em escalas diferentes (ex: Total Charges vs Tenure) distorcem os pesos da rede
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        # OneHotEncoder transforma categorias em colunas binárias
        # handle_unknown="ignore": categorias novas na inferência viram vetor zero, sem levantar erro
        # sparse_output=False: retorna array denso (numpy), compatível com PyTorch tensors
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    # ColumnTransformer aplica cada pipeline apenas nas colunas correspondentes
    # remainder="drop" por padrão — colunas não listadas (ex: CustomerID) são descartadas automaticamente
    return ColumnTransformer(transformers=[
        ("num", numeric_pipe, NUMERIC_FEATURES),
        ("cat", categorical_pipe, CATEGORICAL_FEATURES),
    ])


def load_and_split(
    path: str, test_size: float = 0.2, seed: int = 42
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load raw CSV, clean and return train/test splits + fitted preprocessor."""
    df = pd.read_csv(path)
    df = _clean(df)

    # stratify=y garante que a proporção de churn (~26%) seja mantida em treino e teste
    X = df[CATEGORICAL_FEATURES + NUMERIC_FEATURES]
    y = df[TARGET].values

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed
    )

    preprocessor = build_preprocessor()
    X_train = preprocessor.fit_transform(X_train_raw)  # fit apenas no treino para evitar data leakage
    X_test = preprocessor.transform(X_test_raw)

    logger.info(
        "Data split — train: %d, test: %d, churn rate: %.2f%%",
        len(y_train), len(y_test), y.mean() * 100,
    )
    return X_train, X_test, y_train, y_test, preprocessor
