"""Smoke test: verifica que o ChurnMLP instancia e executa um forward pass corretamente."""

import torch

from src.model import ChurnMLP


def test_mlp_output_shape():
    """O forward pass deve retornar logits com shape (batch_size, 1); sigmoid aplicado externamente."""
    model = ChurnMLP(input_dim=46, hidden_dims=[64, 32], dropout=0.0)
    model.eval()

    x = torch.randn(16, 46)  # batch de 16 amostras com 46 features (shape real pós-preprocessamento)
    with torch.no_grad():
        logits = model(x)
        probs = torch.sigmoid(logits)

    assert logits.shape == (16, 1), f"Shape esperado (16, 1), obtido {logits.shape}"
    assert probs.min() >= 0.0 and probs.max() <= 1.0, "Probabilidades fora do intervalo [0, 1]"
