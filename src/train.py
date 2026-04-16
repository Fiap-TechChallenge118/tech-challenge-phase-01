"""Training utilities for the Churn MLP model."""

import logging
import random

import mlflow
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)

SEED = 42


def set_seeds(seed: int = SEED) -> None:
    """Fix all seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


set_seeds()


def train_model(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    config: dict,
) -> tuple[nn.Module, dict]:
    """
    Treina o modelo com early stopping e loga métricas por epoch no MLflow.

    Args:
        config: dicionário com keys: epochs, batch_size, lr, patience
    Returns:
        modelo treinado e histórico de loss {"train": [...], "val": [...]}
    """
    epochs    = config.get("epochs", 50)
    batch_size = config.get("batch_size", 64)
    lr        = config.get("lr", 1e-3)
    patience  = config.get("patience", 5)

    # Converter arrays numpy em tensores float32
    X_t = torch.tensor(X_train, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_v = torch.tensor(X_val,   dtype=torch.float32)
    y_v = torch.tensor(y_val,   dtype=torch.float32).unsqueeze(1)

    loader = DataLoader(TensorDataset(X_t, y_t), batch_size=batch_size, shuffle=True)

    # pos_weight compensa o desbalanceamento: penaliza mais os erros na classe minoritária (churn)
    pos_weight = torch.tensor([(y_train == 0).sum() / (y_train == 1).sum()], dtype=torch.float32)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)  # mais estável numericamente que BCELoss + Sigmoid
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = {"train": [], "val": []}
    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None

    for epoch in range(1, epochs + 1):
        # --- Treino ---
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(X_batch)
        train_loss /= len(X_t)

        # --- Validação ---
        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_v), y_v).item()

        history["train"].append(train_loss)
        history["val"].append(val_loss)
        mlflow.log_metrics({"train_loss": train_loss, "val_loss": val_loss}, step=epoch)
        logger.info("Epoch %d/%d — train_loss: %.4f | val_loss: %.4f", epoch, epochs, train_loss, val_loss)

        # --- Early stopping: interrompe se val_loss não melhorar por `patience` epochs ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}  # salva o melhor estado
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info("Early stopping na epoch %d", epoch)
                break

    model.load_state_dict(best_state)  # restaura o modelo do melhor epoch, não do último
    return model, history
