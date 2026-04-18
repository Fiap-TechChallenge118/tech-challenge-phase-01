# Churn MLP — Tech Challenge Fase 01

Rede Neural (MLP) para previsão de churn em operadora de telecomunicações.
Pipeline end-to-end: EDA → Baselines → MLP (PyTorch) → API (FastAPI) → Deploy (AWS).

---

## Arquitetura do Projeto

```
churn-mlp/
├── data/
│   ├── raw/                  # Dataset original (não versionado)
│   └── processed/            # Artefatos gerados (preprocessor.pkl, model.pth)
├── notebooks/
│   ├── 01_eda.ipynb          # Análise exploratória
│   ├── 02_baseline.ipynb     # DummyClassifier + LogisticRegression
│   └── 03_mlp_training.ipynb # Treinamento e avaliação da MLP
├── src/
│   ├── preprocessing.py      # ColumnTransformer, load_and_split
│   ├── model.py              # ChurnMLP (nn.Module)
│   ├── train.py              # Loop de treino + early stopping
│   ├── evaluate.py           # Métricas + análise de custo
│   └── pipeline.py           # Orquestrador end-to-end
├── api/
│   ├── main.py               # FastAPI app (/health, /predict)
│   └── schemas.py            # Pydantic models (input/output)
├── tests/
│   ├── test_smoke.py         # Forward pass da MLP
│   ├── test_schema.py        # Validação do dataset com pandera
│   └── test_api.py           # Testes dos endpoints
├── docs/
│   ├── ml_canvas.md          # ML Canvas (stakeholders, métricas, SLOs)
│   ├── model_card.md         # Model Card (limitações, vieses, métricas)
│   └── monitoring_plan.md    # Plano de monitoramento e alertas
├── Makefile                  # Atalhos: lint, test, train, run
├── Dockerfile                # Imagem para deploy
└── pyproject.toml            # Única fonte de verdade (deps + ruff + pytest)
```

---

## Pré-requisitos

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) — gerenciador de pacotes e ambientes virtuais

### Instalar o uv (caso não tenha)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env   # adiciona uv ao PATH da sessão atual
```

Para tornar permanente, adicione ao seu `~/.bashrc` ou `~/.zshrc`:

```bash
source $HOME/.local/bin/env
```

---

## Setup do Ambiente

### 1. Criar o ambiente virtual

```bash
uv venv .venv
```

### 2. Ativar o ambiente

```bash
source .venv/bin/activate
```

> Para desativar: `deactivate`

### 3. Instalar dependências

```bash
uv pip install -e ".[dev]"
```

O flag `-e` instala o projeto em modo editável (editable install).
O grupo `[dev]` inclui `pytest` e `ruff`.

### 4. Verificar instalação

```bash
python -c "import torch, sklearn, numpy, mlflow, fastapi; print('OK')"
```

---

## Como Executar

> Todos os comandos assumem que o ambiente virtual está ativo.

### Treinar o pipeline completo

```bash
make train
# equivale a: python -m src.pipeline
```

### Rodar a API

```bash
make run
# equivale a: uvicorn api.main:app --reload
```

Acesse: `http://localhost:8000/docs` (Swagger UI automático do FastAPI)

### Rodar os testes

```bash
make test
# equivale a: pytest tests/ -v
```

### Verificar qualidade do código

```bash
make lint
# equivale a: ruff check .
```

---

## MLflow

Para visualizar os experimentos registrados:

```bash
mlflow ui
```

Acesse: `http://localhost:5000`

---

## Docker

### Build

```bash
docker build -t churn-api .
```

### Run

```bash
docker run -p 8000:8000 churn-api
```

### Testar

```bash
curl http://localhost:8000/health
```

---

## Dataset

**Telco Customer Churn (IBM)** — 7.043 registros × 33 features
Localização: `data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv`

Para baixar novamente:

```bash
curl -L "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv" \
  -o data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv
```

---

## Resultados

Validação cruzada estratificada (StratifiedKFold, k=5) no conjunto de treino. Métricas reportadas como média dos folds.

| Modelo | F1 | ROC-AUC | Precision | Recall |
|---|---|---|---|---|
| DummyClassifier (most_frequent) | 0.000 | 0.500 | 0.000 | 0.000 |
| Logistic Regression | 0.639 | 0.856 | 0.532 | 0.800 |
| **MLP — PyTorch** (produção) | **0.628** | **0.840** | **0.579** | **0.687** |

O MLP apresenta precision superior (+4.7pp vs LR), reduzindo campanhas de retenção desnecessárias. O threshold padrão é 0.5 e pode ser ajustado via `THRESHOLD` em `api/main.py`.

---

## Documentação Adicional

- [ML Canvas](docs/ml_canvas.md)
- [Model Card](docs/model_card.md)
- [Plano de Monitoramento](docs/monitoring_plan.md)
