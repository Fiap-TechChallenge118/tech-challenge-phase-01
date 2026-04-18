"""MLP model definition for churn prediction."""

import torch.nn as nn


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
        layers.append(nn.Linear(prev_dim, 1))  # Camada de saída: um único neurônio (classificação binária)
        layers.append(nn.Sigmoid())             # Converte o logit em probabilidade [0, 1]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
