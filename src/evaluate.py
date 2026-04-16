"""Evaluation utilities for churn prediction models."""

import logging

import numpy as np
import torch
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.model import ChurnMLP

logger = logging.getLogger(__name__)


def evaluate(
    model: ChurnMLP, X_test: np.ndarray, y_test: np.ndarray, threshold: float = 0.5
) -> dict:
    """Compute F1, ROC-AUC, Precision, Recall for a trained ChurnMLP.

    Args:
        model:     trained ChurnMLP (in eval mode after this call)
        X_test:    preprocessed test features (numpy array)
        y_test:    true binary labels (numpy array)
        threshold: decision threshold for converting probability to class label

    Returns:
        dict with keys: f1, roc_auc, precision, recall
    """
    model.eval()
    with torch.no_grad():
        probs = model(torch.tensor(X_test, dtype=torch.float32)).squeeze().numpy()

    preds = (probs >= threshold).astype(int)

    metrics = {
        "f1":        f1_score(y_test, preds),
        "roc_auc":   roc_auc_score(y_test, probs),
        "precision": precision_score(y_test, preds),
        "recall":    recall_score(y_test, preds),
    }
    logger.info("Evaluation — %s", metrics)
    return metrics


def cost_analysis(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    cost_fp: float,
    cost_fn: float,
) -> dict:
    """Calculate total business cost based on confusion matrix.

    False Negative (missed churn): customer lost — higher cost.
    False Positive (false alarm):  unnecessary retention action — lower cost.

    Args:
        cost_fp: cost per false positive (e.g. R$50 for a retention campaign)
        cost_fn: cost per false negative (e.g. R$500 for a lost customer)

    Returns:
        dict with fp_count, fn_count, total_cost
    """
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    total = fp * cost_fp + fn * cost_fn

    result = {"fp_count": fp, "fn_count": fn, "total_cost": total}
    logger.info("Cost analysis — %s", result)
    return result
