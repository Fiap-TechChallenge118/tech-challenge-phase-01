"""MLP model definition for churn prediction."""

import numpy as np
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, ClassifierMixin


# ChurnMLP é a rede neural principal do projeto — substitui os baselines lineares (LogisticRegression)
# por uma arquitetura capaz de aprender relações não-lineares entre as features do cliente.
# Recebe features tabulares pré-processadas e retorna a probabilidade de churn (0 a 1).
class ChurnMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list[int], dropout: float = 0.3):
        """
        Args:
            input_dim:   número de features após o preprocessamento (numérico + OHE categórico)
            hidden_dims: lista com o tamanho de cada camada oculta, ex: [64, 32]
            dropout:     fração de neurônios desativados por batch durante o treino
        """
        super().__init__()
        # hidden_dims define quantas camadas ocultas existem e o tamanho de cada uma.
        # Ex: [64, 32] cria duas camadas — a primeira com 64 neurônios, a segunda com 32.
        # O laço constrói cada camada dinamicamente, conectando a saída da anterior
        # à entrada da próxima (prev_dim → dim), permitindo testar diferentes arquiteturas
        # sem alterar o código — basta mudar a lista passada no __init__.
        layers = []
        prev_dim = input_dim
        for dim in hidden_dims:
            layers += [
                nn.Linear(prev_dim, dim),
                nn.BatchNorm1d(dim),  # Estabiliza o treinamento normalizando as ativações de cada batch
                nn.ReLU(),            # Ativação não-linear — permite aprender padrões além de fronteiras lineares
                nn.Dropout(dropout),  # Desativa neurônios aleatoriamente para reduzir overfitting
            ]
            prev_dim = dim
        layers.append(nn.Linear(prev_dim, 1))  # Camada de saída: logit bruto (sigmoid aplicado externamente)
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class ChurnMLPWrapper(BaseEstimator, ClassifierMixin):
    """Wrapper sklearn-compatível para o ChurnMLP.

    Permite usar o modelo PyTorch em pipelines sklearn (ex: cross_val_score, GridSearchCV)
    e como drop-in replacement para classificadores sklearn (predict_proba, predict).
    """

    def __init__(
        self,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.3,
        threshold: float = 0.5,
    ):
        self.hidden_dims = hidden_dims or [64, 32]
        self.dropout = dropout
        self.threshold = threshold
        self.model_: ChurnMLP | None = None
        self.classes_ = np.array([0, 1])

    def fit(self, X: np.ndarray, y: np.ndarray, **fit_params):
        from src.train import train_model

        cfg = {
            "epochs":     fit_params.get("epochs", 100),
            "batch_size": fit_params.get("batch_size", 64),
            "lr":         fit_params.get("lr", 1e-3),
            "patience":   fit_params.get("patience", 10),
        }
        input_dim = X.shape[1]
        self.model_ = ChurnMLP(input_dim=input_dim, hidden_dims=self.hidden_dims, dropout=self.dropout)
        self.model_, _ = train_model(self.model_, X, y, X, y, cfg)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self.model_.eval()
        with torch.no_grad():
            probs = torch.sigmoid(
                self.model_(torch.tensor(X, dtype=torch.float32))
            ).squeeze().numpy()
        return np.column_stack([1 - probs, probs])

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= self.threshold).astype(int)
