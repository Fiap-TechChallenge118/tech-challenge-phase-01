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
    """Avalia o modelo treinado no conjunto de teste e retorna as 4 métricas principais.

    Args:
        model:     ChurnMLP já treinado
        X_test:    features pré-processadas do conjunto de teste (numpy array)
        y_test:    rótulos binários reais (0 = não churn, 1 = churn)
        threshold: limiar de decisão — probabilidades acima disso viram previsão positiva (churn).
                   0.5 é o padrão, mas pode ser ajustado via cost_analysis para reduzir custos.

    Returns:
        dict com chaves: f1, roc_auc, precision, recall
    """
    # model.eval() desativa Dropout e BatchNorm no modo inferência —
    # sem isso os resultados são não-determinísticos e incorretos para avaliação.
    model.eval()
    with torch.no_grad():  # torch.no_grad() economiza memória e acelera: não precisa calcular gradientes na inferência
        probs = torch.sigmoid(model(torch.tensor(X_test, dtype=torch.float32))).squeeze().numpy()
        # .squeeze() remove a dimensão extra de saída (batch_size, 1) → (batch_size,)
        # .numpy() converte para array numpy, compatível com sklearn

    # Converte probabilidades contínuas [0,1] em rótulos binários {0, 1} usando o threshold
    preds = (probs >= threshold).astype(int)

    metrics = {
        # F1-Score: média harmônica de precision e recall — principal métrica para dados desbalanceados (churn ~26%)
        "f1":        f1_score(y_test, preds, zero_division=0),
        # ROC-AUC: mede a capacidade do modelo de separar as classes em todos os thresholds possíveis.
        # Usa as probabilidades brutas (probs), não os rótulos — independente do threshold escolhido.
        "roc_auc":   roc_auc_score(y_test, probs),
        # Precision: dos clientes classificados como churn, quantos realmente foram?
        # Precision baixa = muitos alarmes falsos (FP) → campanhas de retenção desnecessárias.
        # zero_division=0: quando nenhuma amostra é predita como positiva, precision = 0.0 (sem warning)
        "precision": precision_score(y_test, preds, zero_division=0),
        # Recall: dos clientes que realmente foram embora, quantos o modelo detectou?
        # Recall baixo = muitos churns não detectados (FN) → clientes perdidos sem intervenção.
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
    """Calcula o custo total de negócio com base nos erros do modelo.

    No contexto de churn:
    - Falso Positivo (FP): o modelo previu churn, mas o cliente ficaria.
      Custo = ação de retenção desnecessária (ex: desconto ou ligação de suporte).
    - Falso Negativo (FN): o modelo não previu churn, mas o cliente foi embora.
      Custo = receita perdida para sempre — geralmente muito maior que o FP.

    Esse trade-off justifica ajustar o threshold: abaixar o threshold aumenta o recall
    (detecta mais churns) mas aumenta os FPs — a função permite quantificar qual ponto
    minimiza o custo total para o negócio.

    Args:
        y_true:   rótulos reais
        y_pred:   rótulos previstos pelo modelo (binários, já aplicado o threshold)
        cost_fp:  custo unitário de um FP (ex: R$ 50 por campanha de retenção)
        cost_fn:  custo unitário de um FN (ex: R$ 500 por cliente perdido)

    Returns:
        dict com: fp_count, fn_count, total_cost
    """
    # Conta FPs: previu churn (1) mas era não-churn (0)
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    # Conta FNs: previu não-churn (0) mas era churn (1) — o erro mais caro
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    # Custo total = soma ponderada dos dois tipos de erro
    total = fp * cost_fp + fn * cost_fn

    result = {"fp_count": fp, "fn_count": fn, "total_cost": total}
    logger.info("Cost analysis — %s", result)
    return result
