"""Smoke test: verifica que o ChurnMLP instancia e executa um forward pass corretamente."""

import torch

from src.model import ChurnMLP


def test_mlp_output_shape():
    """O forward pass deve retornar shape (batch_size, 1) com valores em [0, 1]."""
    model = ChurnMLP(input_dim=46, hidden_dims=[64, 32], dropout=0.0)
    model.eval()

    x = torch.randn(16, 46)  # batch de 16 amostras com 46 features (shape real pós-preprocessamento)
    with torch.no_grad():
        out = model(x)

    assert out.shape == (16, 1), f"Shape esperado (16, 1), obtido {out.shape}"
    # Sigmoid na saída garante que todas as probabilidades ficam entre 0 e 1
    assert out.min() >= 0.0 and out.max() <= 1.0, "Saída fora do intervalo [0, 1]"
