"""End-to-end training pipeline: load → preprocess → train → evaluate → save artifacts."""

import json
import logging
from pathlib import Path

import joblib
import mlflow
import mlflow.pytorch
import numpy as np
import torch

from src.evaluate import cost_analysis, evaluate
from src.model import ChurnMLP
from src.preprocessing import load_and_split
from src.train import set_seeds, train_model

logger = logging.getLogger(__name__)

# Diretório onde o preprocessor e os pesos do modelo são persistidos para uso pela API
ARTIFACTS_DIR = Path("data/processed")

# Configuração padrão do experimento — pode ser sobrescrita via argumento em run_pipeline()
DEFAULT_CONFIG = {
    "epochs": 100,
    "batch_size": 64,
    "lr": 1e-3,
    "patience": 10,
    "hidden_dims": [64, 32],
    "dropout": 0.3,
    "seed": 42,
}


def run_pipeline(data_path: str, config: dict | None = None) -> dict:
    """Executa o pipeline completo de treino e retorna as métricas finais.

    Etapas:
        1. Fixar seeds para reprodutibilidade
        2. Carregar e dividir os dados (train/test estratificado)
        3. Instanciar e treinar o ChurnMLP com early stopping
        4. Avaliar no conjunto de teste (F1, ROC-AUC, Precision, Recall)
        5. Logar parâmetros, métricas e modelo no MLflow
        6. Salvar artefatos locais: preprocessor.pkl e model.pth

    Args:
        data_path: caminho para o CSV raw do dataset Telco
        config:    dicionário de hiperparâmetros; valores ausentes usam DEFAULT_CONFIG

    Returns:
        dict com as métricas do modelo no conjunto de teste
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    set_seeds(cfg["seed"])

    # --- 1. Dados ---
    X_train, X_test, y_train, y_test, preprocessor = load_and_split(
        data_path, seed=cfg["seed"]
    )
    input_dim = X_train.shape[1]

    mlflow.set_experiment("churn-mlp")

    with mlflow.start_run():
        # Loga todos os hiperparâmetros de uma vez para rastreabilidade completa do experimento
        mlflow.log_params({k: v for k, v in cfg.items() if k != "hidden_dims"})
        mlflow.log_param("hidden_dims", str(cfg["hidden_dims"]))

        # --- 2. Modelo e treino ---
        model = ChurnMLP(
            input_dim=input_dim,
            hidden_dims=cfg["hidden_dims"],
            dropout=cfg["dropout"],
        )
        model, _ = train_model(model, X_train, y_train, X_test, y_test, cfg)

        # --- 3. Avaliação ---
        metrics = evaluate(model, X_test, y_test)
        mlflow.log_metrics(metrics)
        logger.info("Pipeline concluído — métricas: %s", metrics)

        # Registra o modelo PyTorch como artefato MLflow
        # isso permite carregar o modelo depois com mlflow.pytorch.load_model(run_uri)
        mlflow.pytorch.log_model(model, "model")

        # --- 4. Salvar artefatos locais para a API ---
        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

        # O preprocessor (ColumnTransformer já fitado) precisa ser salvo junto com o modelo
        # pois a API precisa transformar os dados de entrada antes de passar para a rede
        preprocessor_path = ARTIFACTS_DIR / "preprocessor.pkl"
        joblib.dump(preprocessor, preprocessor_path)
        logger.info("Preprocessor salvo em %s", preprocessor_path)

        # Salva apenas os pesos (state_dict), não a arquitetura inteira —
        # mais leve e seguro; a arquitetura é reconstruída pela API usando ChurnMLP
        model_path = ARTIFACTS_DIR / "model.pth"
        torch.save(model.state_dict(), model_path)
        logger.info("Model state_dict salvo em %s", model_path)

        # --- 5. Threshold ótimo por análise de custo ---
        # Varre thresholds e seleciona o que minimiza custo total (FN=500, FP=50)
        model.eval()
        with torch.no_grad():
            probs = torch.sigmoid(
                model(torch.tensor(X_test, dtype=torch.float32))
            ).squeeze().numpy()

        thresholds = np.linspace(0.1, 0.9, 80)
        costs = [
            cost_analysis(y_test, (probs >= t).astype(int), cost_fp=50, cost_fn=500)["total_cost"]
            for t in thresholds
        ]
        best_threshold = float(thresholds[np.argmin(costs)])
        mlflow.log_param("best_threshold", best_threshold)

        threshold_path = ARTIFACTS_DIR / "threshold.json"
        with open(threshold_path, "w") as f:
            json.dump({"threshold": best_threshold}, f)
        logger.info("Threshold ótimo salvo em %s — threshold=%.4f", threshold_path, best_threshold)

    return metrics


if __name__ == "__main__":
    # Permite rodar via: python -m src.pipeline
    # O Makefile usa este entry point no target `make train`
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s — %(message)s")
    run_pipeline("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")
