"""Pytest configuration: garante que os testes rodam com o cwd na raiz do projeto."""

from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def set_project_root(monkeypatch):
    """Muda o cwd para a raiz do projeto antes de cada teste.

    A API usa paths relativos (ex: data/processed/) que só existem a partir da raiz.
    O pytest por padrão roda no cwd de onde foi chamado — esta fixture normaliza isso.
    """
    monkeypatch.chdir(Path(__file__).parent.parent)
