"""Batch inference script — processa todos os clientes e armazena scores no SQLite.

Execução: python -m src.batch_predict
Makefile:  make batch

O script carrega o modelo treinado, roda inferência sobre a base completa e persiste
os scores em data/processed/scores.db (tabela churn_scores). O endpoint GET /predict
da API consulta esse banco, sem executar o modelo a cada request.
"""

import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
import torch

from src.model import ChurnMLP
from src.preprocessing import CATEGORICAL_FEATURES, NUMERIC_FEATURES, _clean

logger = logging.getLogger(__name__)

ARTIFACTS_DIR   = Path("data/processed")
MODEL_PATH      = ARTIFACTS_DIR / "model.pth"
PREPROCESSOR_PATH = ARTIFACTS_DIR / "preprocessor.pkl"
SCORES_DB_PATH  = ARTIFACTS_DIR / "scores.db"
THRESHOLD_PATH  = ARTIFACTS_DIR / "threshold.json"

CUSTOMER_ID_COL = "CustomerID"
FEATURE_COLS    = CATEGORICAL_FEATURES + NUMERIC_FEATURES


def run_batch_predict(data_path: str) -> int:
    """Executa inferência em lote sobre todos os clientes e salva scores no SQLite.

    Args:
        data_path: caminho para o CSV raw do dataset Telco

    Returns:
        Número de clientes processados.
    """
    preprocessor = joblib.load(PREPROCESSOR_PATH)

    threshold = 0.5
    if THRESHOLD_PATH.exists():
        with open(THRESHOLD_PATH) as f:
            threshold = json.load(f).get("threshold", 0.5)
    logger.info("Threshold carregado: %.4f", threshold)

    # Inferir input_dim a partir do preprocessor para não hardcodar
    _dummy = pd.DataFrame([{col: ("No" if col in CATEGORICAL_FEATURES else 0.0) for col in FEATURE_COLS}])
    input_dim = preprocessor.transform(_dummy).shape[1]

    model = ChurnMLP(input_dim=input_dim, hidden_dims=[64, 32], dropout=0.3)
    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
    model.eval()

    df = _clean(pd.read_csv(data_path))

    X = preprocessor.transform(df[FEATURE_COLS])

    with torch.no_grad():
        logits = model(torch.tensor(X, dtype=torch.float32)).squeeze()
        probs  = torch.sigmoid(logits).numpy()

    preds = (probs >= threshold).astype(int)
    computed_at = datetime.now(timezone.utc).isoformat()

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(SCORES_DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS churn_scores (
            customer_id       TEXT PRIMARY KEY,
            churn_probability REAL,
            churn_prediction  INTEGER,
            computed_at       TEXT
        )
    """)
    conn.executemany(
        "INSERT OR REPLACE INTO churn_scores VALUES (?, ?, ?, ?)",
        [
            (str(cid), round(float(p), 4), int(pred), computed_at)
            for cid, p, pred in zip(df[CUSTOMER_ID_COL], probs, preds)
        ],
    )
    conn.commit()
    conn.close()

    logger.info(
        "Batch predict concluído — %d clientes, threshold=%.4f → %s",
        len(df), threshold, SCORES_DB_PATH,
    )
    return len(df)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s — %(message)s")
    n = run_batch_predict("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    print(f"Scores calculados para {n} clientes -> {SCORES_DB_PATH}")
