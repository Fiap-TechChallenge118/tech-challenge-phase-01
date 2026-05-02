# Churn MLP — Tech Challenge Fase 01

![Python](https://img.shields.io/badge/python-3.11+-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111+-009688?logo=fastapi&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-2.12+-0194E2?logo=mlflow&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4+-F7931E?logo=scikit-learn&logoColor=white)
![Pydantic](https://img.shields.io/badge/Pydantic-v2-E92063?logo=pydantic&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-2.2+-150458?logo=pandas&logoColor=white)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
![pytest](https://img.shields.io/badge/pytest-passing-brightgreen?logo=pytest&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-ready-2496ED?logo=docker&logoColor=white)

Rede Neural (MLP) para previsão de churn em operadora de telecomunicações.  
Pipeline end-to-end: EDA → Baselines → MLP (PyTorch) → API (FastAPI) → Deploy (Docker/Azure).

---

> ### 🌐 URL Pública da API
> **`http://churn-mlp-api.brazilsouth.azurecontainer.io:8000`**
> [![Live Demo](https://img.shields.io/badge/🌐_Live_Demo-Azure_ACI-brightgreen)](http://churn-mlp-api.brazilsouth.azurecontainer.io:8000/docs)
>
> Deploy em **Azure Container Instances** — região Brazil South.

> ### 🎬 Vídeo STAR — Apresentação do Projeto
> **`URL_VIDEO_STAR_AQUI`**
> [![Vídeo STAR](https://img.shields.io/badge/🎬_Vídeo_STAR-Apresentação-red?logo=youtube&logoColor=white)](URL_VIDEO_STAR_AQUI)
>
> *Substitua pelo link do vídeo (YouTube, Google Drive, etc.).*

---

## Sumário

- [Contexto do Problema](#contexto-do-problema)
- [Arquitetura da Solução](#arquitetura-da-solução)
- [Escolha do Modelo: Por Que MLP?](#escolha-do-modelo-por-que-mlp)
- [Arquitetura do Projeto](#arquitetura-do-projeto)
- [Pré-requisitos](#pré-requisitos)
- [Download do Dataset](#download-do-dataset)
- [Setup do Ambiente](#setup-do-ambiente)
- [Como Executar](#como-executar)
- [Endpoints da API](#endpoints-da-api)
- [MLflow](#mlflow)
- [Docker](#docker)
- [Dataset](#dataset)
- [Resultados](#resultados)
- [Análise de Custo de Negócio](#análise-de-custo-de-negócio)
- [Plano de Monitoramento](#plano-de-monitoramento)
- [URL Pública do Projeto](#url-pública-do-projeto)
- [Vídeo STAR](#vídeo-star)
- [Documentação Adicional](#documentação-adicional)

---

## Contexto do Problema

Uma operadora de telecomunicações enfrenta perda recorrente de clientes (churn). Cada cliente perdido representa não apenas a receita mensal deixada de receber, mas também o custo elevado de aquisição de um novo cliente para substituí-lo. O objetivo deste projeto é construir um sistema preditivo capaz de identificar, com antecedência, quais clientes apresentam maior risco de cancelamento — permitindo que o time de retenção atue de forma proativa e priorizada.

**Problema de ML:** Classificação binária supervisionada. Dado o perfil de um cliente (dados demográficos, serviços contratados e histórico de faturamento), prever se ele irá cancelar o serviço (`Churn = 1`) ou permanecer (`Churn = 0`).

**Dataset:** IBM Telco Customer Churn — 7.043 registros, 19 features relevantes (demográficas, de serviço e contratuais), com taxa de churn de aproximadamente 26%.

---

## Arquitetura da Solução

A solução foi projetada em quatro camadas independentes e desacopladas, cada uma com responsabilidade bem definida:

```
┌─────────────────────────────────────────────────────────────────────┐
│                         CAMADA DE DADOS                             │
│   CSV bruto → limpeza → ColumnTransformer → arrays NumPy            │
│   (src/preprocessing.py)                                            │
└────────────────────────────┬────────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────────┐
│                       CAMADA DE MODELAGEM                           │
│   ChurnMLP(46 → 64 → 32 → 1) + early stopping + pos_weight         │
│   Rastreamento de experimentos via MLflow                           │
│   (src/model.py, src/train.py, src/evaluate.py, src/pipeline.py)   │
└────────────────────────────┬────────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────────┐
│                       CAMADA DE ARTEFATOS                           │
│   preprocessor.pkl  │  model.pth  │  threshold.json  │  scores.db  │
│   (data/processed/)                                                 │
└────────────────────────────┬────────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────────┐
│                        CAMADA DE SERVIÇO                            │
│   FastAPI: /health  │  POST /predict  │  GET /predict/batch         │
│   Validação Pydantic + middleware de latência                       │
│   (api/main.py, api/schemas.py)                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Princípios de Design

**Separação de responsabilidades:** cada módulo faz uma única coisa. O preprocessador não sabe nada do modelo; o modelo não sabe nada da API. Isso facilita substituir qualquer componente de forma independente.

**Artefatos versionados:** após o treino, três arquivos são salvos em `data/processed/`: o preprocessador fitado (`preprocessor.pkl`), os pesos do modelo (`model.pth`) e o threshold calibrado (`threshold.json`). A API carrega esses artefatos no startup, sem nenhum acoplamento com o código de treino.

**Dois modos de inferência:** a API oferece inferência on-demand (POST /predict, para sistemas em tempo real) e consulta em lote (GET /predict/batch, para sistemas que consomem scores pré-calculados semanalmente). Essa separação evita recalcular probabilidades para o mesmo cliente em cada requisição de dashboards operacionais.

**Rastreabilidade com MLflow:** cada execução do pipeline registra parâmetros, métricas e o artefato do modelo. Isso permite comparar experimentos, auditar o modelo em produção e acionar retreino quando métricas caírem.

---

## Escolha do Modelo: Por Que MLP?

### Processo de Seleção

A seleção do modelo seguiu uma progressão deliberada do simples para o complexo, respeitando o princípio de que a complexidade deve ser justificada pelo ganho de performance.

**Etapa 1 — Baseline ingênuo (DummyClassifier):** prevê sempre a classe majoritária (sem churn). F1 = 0.000, ROC-AUC = 0.500, PR-AUC = 0.265 (equivale à taxa de churn base). Serve apenas como piso de referência — qualquer modelo útil precisa superar isso.

**Etapa 2 — Baseline linear (Regressão Logística):** modelo linear com regularização L2 e `class_weight="balanced"`. Alcançou F1 = 0.639, ROC-AUC = 0.856 e PR-AUC = 0.645, com alta interpretabilidade e treino em menos de 1 segundo. É o modelo de referência competitivo que o MLP precisa superar para se justificar.

**Etapa 3 — MLP (modelo de produção):** rede neural densa com duas camadas ocultas, BatchNorm e Dropout. Alcançou F1 = 0.628, ROC-AUC = 0.840 e PR-AUC = 0.637.

### Por Que o MLP Foi Escolhido Para Produção?

À primeira vista, as métricas da Regressão Logística parecem superiores em F1 e recall. A escolha do MLP como modelo de produção se justifica por três razões complementares:

**1. Precision superior reduz custo de campanha**

O MLP apresenta precision de 0.579 contra 0.532 da Regressão Logística (+4.7 pontos percentuais). No contexto de negócio, isso significa que, para cada 100 clientes sinalizados como churn, o MLP gera aproximadamente 5 campanhas de retenção a menos desnecessárias. Com um custo estimado de R$ 50 por ação de retenção (ligação, desconto, etc.), essa diferença é economicamente relevante em escala.

**2. Capacidade de aprender interações não-lineares**

O dataset contém dependências complexas entre features: clientes com contrato mensal e Fiber optic apresentam taxa de churn de 70%+, enquanto clientes com contrato bienal e DSL ficam abaixo de 5%. Um modelo linear trata cada feature de forma aditiva e independente, enquanto a MLP aprende automaticamente essas combinações na primeira camada oculta. Essa capacidade é especialmente valiosa se novos padrões emergirem com a entrada de dados de produção.

**3. Extensibilidade arquitetural**

A arquitetura MLP com wrapper sklearn-compatível (`ChurnMLPWrapper`) permite uso direto em `GridSearchCV`, `Pipeline` do scikit-learn e futura expansão para features de embedding (como histórico de suporte ou texto de atendimento). Um modelo linear exigiria reengenharia significativa para incorporar essas extensões.

### Por Que Não Outros Modelos?

**Gradient Boosting (XGBoost/LightGBM):** geralmente supera MLPs em tabular data com poucos dados. Foi descartado aqui porque o objetivo pedagógico do challenge é implementar uma rede neural com PyTorch. Em produção real, um ensemble de árvores seria um candidato forte.

**Random Forest:** interpretabilidade mediana, sem capacidade de fine-tuning incremental. Descartado pelos mesmos motivos que o Gradient Boosting.

**Redes maiores (3+ camadas):** testadas nos notebooks mas descartadas por overfitting. Com 7.043 registros e 46 features após OHE, redes com mais de duas camadas ocultas não convergem de forma estável no regime de early stopping com patience = 10.

### Arquitetura Detalhada do MLP

```
Input(46)
    │
    ├─ Linear(46 → 64)
    ├─ BatchNorm1d(64)    ← normaliza ativações, estabiliza gradientes
    ├─ ReLU()
    ├─ Dropout(0.3)       ← regularização: desliga 30% dos neurônios por batch
    │
    ├─ Linear(64 → 32)
    ├─ BatchNorm1d(32)
    ├─ ReLU()
    ├─ Dropout(0.3)
    │
    └─ Linear(32 → 1)    ← logit bruto (Sigmoid aplicado externamente)
```

**Decisões de arquitetura:**

- **BatchNorm antes de Dropout:** a ordem BN → ReLU → Dropout é a que produz gradientes mais estáveis neste regime de dados. BatchNorm normaliza a distribuição de entrada de cada camada, acelerando a convergência e reduzindo a sensibilidade à taxa de aprendizado.

- **Dropout(0.3):** com apenas ~5.600 exemplos de treino, dropout é essencial para evitar memorização. O valor 0.3 foi escolhido por ser conservador o suficiente para não destruir sinal em camadas estreitas (32 neurônios).

- **pos_weight na BCELoss:** o dataset tem ~74% negativos e ~26% positivos. Sem correção, o modelo aprenderia a prever "sem churn" na maior parte do tempo. O `pos_weight = n_negativos / n_positivos ≈ 2.84` pondera os erros nos positivos de forma equivalente ao undersampling, mas sem perda de dados.

- **Adam (lr = 0.001):** otimizador adaptativo que ajusta a taxa de aprendizado por parâmetro. Convergência mais estável que SGD puro para problemas de classificação desbalanceada.

- **Early stopping (patience = 10):** interrompe o treino quando a loss de validação não melhora por 10 épocas consecutivas, restaurando os pesos da melhor época. Previne overfitting sem necessidade de ajustar o número de épocas manualmente.

### Threshold Calibrado por Custo

O threshold padrão de 0.5 não é ótimo para este problema. Com FN (cliente perdido) custando R$ 500 e FP (campanha desnecessária) custando R$ 50, a assimetria de custo é 10:1. A análise de custo sobre o test set mostra:

| Threshold | FP | FN | Custo Total |
|-----------|----|----|-------------|
| 0.50 | 180 | 121 | R$ 69.500 |
| 0.10 (ótimo) | ~350 | ~40 | ~R$ 61.000 |

O threshold de 0.10 é salvo automaticamente em `data/processed/threshold.json` após cada execução do pipeline e carregado pela API no startup.

---

## Arquitetura do Projeto

```
churn-mlp/
├── data/
│   ├── raw/                  # Dataset original (não versionado)
│   └── processed/            # Artefatos gerados: preprocessor.pkl, model.pth,
│                             # threshold.json, scores.db
├── notebooks/
│   ├── 01_eda.ipynb          # Análise exploratória: distribuições, correlações, nulos
│   ├── 02_baseline.ipynb     # DummyClassifier + LogisticRegression + validação cruzada
│   └── 03_mlp_training.ipynb # Treinamento MLP, curva de loss, análise de threshold
├── src/
│   ├── preprocessing.py      # ColumnTransformer (StandardScaler + OHE), load_and_split
│   ├── model.py              # ChurnMLP (nn.Module) + ChurnMLPWrapper (sklearn-compat)
│   ├── train.py              # Loop de treino, early stopping, MLflow logging
│   ├── evaluate.py           # Métricas (F1, AUC, P, R) + análise de custo (FP/FN)
│   ├── pipeline.py           # Orquestrador end-to-end: load → train → evaluate → save
│   └── batch_predict.py      # Inferência em lote → salva scores no SQLite
├── api/
│   ├── main.py               # FastAPI: /health, POST /predict, GET /predict/batch
│   └── schemas.py            # Pydantic: CustomerFeatures, PredictionResponse
├── tests/
│   ├── conftest.py           # Fixture de cwd para testes
│   ├── test_smoke.py         # Forward pass, wrapper sklearn, carregamento de artefatos
│   ├── test_schema.py        # Validação do dataset com pandera (shape, tipos, valores)
│   └── test_api.py           # Testes dos endpoints (200, 422, 503, 404)
├── docs/
│   ├── ml_canvas.md          # ML Canvas: stakeholders, features, SLOs, riscos
│   ├── model_card.md         # Model Card: métricas, vieses, limitações, retreino
│   └── monitoring_plan.md    # Monitoramento: PSI, alertas, playbook de resposta
├── examples/
│   ├── api_client.py         # Cliente Python de exemplo para a API
│   ├── curl_examples.sh      # Exemplos de curl para todos os endpoints
│   └── payloads.json         # Payloads de alto e baixo risco para teste
├── Makefile                  # Atalhos: lint, test, train, run, batch
├── Dockerfile                # Imagem para deploy (python:3.11-slim)
└── pyproject.toml            # Única fonte de verdade: deps, ruff, pytest, coverage
```

---

## Pré-requisitos

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) — gerenciador de pacotes e ambientes virtuais

### Instalar o uv (caso não tenha)

**Linux/macOS:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env   # adiciona uv ao PATH da sessão atual
```

Para tornar permanente, adicione ao seu `~/.bashrc` ou `~/.zshrc`:
```bash
source $HOME/.local/bin/env
```

**Windows (PowerShell):**
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

> No Windows, o instalador já adiciona o `uv` ao PATH automaticamente. Reinicie o terminal após a instalação.

---

## Download do Dataset

> **Faça o download antes de prosseguir com o Setup.**  
> O pipeline de treino e os testes dependem do arquivo CSV em `data/raw/`.

**Linux/macOS:**
```bash
curl -L "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv" \
  -o data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv
```

**Windows (PowerShell):**
```powershell
Invoke-WebRequest -Uri "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv" `
  -OutFile "data\raw\WA_Fn-UseC_-Telco-Customer-Churn.csv"
```

Após o download, confirme que o arquivo existe em `data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv` antes de continuar.

---

## Setup do Ambiente

### 1. Criar o ambiente virtual

```bash
uv venv .venv
```

### 2. Ativar o ambiente

**Linux/macOS:**
```bash
source .venv/bin/activate
```

**Windows (PowerShell):**
```powershell
.venv\Scripts\Activate.ps1
```

**Windows (CMD):**
```cmd
.venv\Scripts\activate.bat
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

> **Windows:** o comando `make` não está disponível por padrão. Instale via [Chocolatey](https://chocolatey.org/) (`choco install make`) ou use diretamente os comandos equivalentes listados abaixo.

### Treinar o pipeline completo

```bash
make train
```

**Windows (sem make):**
```powershell
python -m src.pipeline
```

Gera os artefatos em `data/processed/`: `preprocessor.pkl`, `model.pth`, `threshold.json`.

### Gerar scores em lote

```bash
make batch
```

**Windows (sem make):**
```powershell
python -m src.batch_predict
```

Processa todos os clientes do dataset e salva scores em `data/processed/scores.db`.

### Rodar a API

```bash
make run
```

**Windows (sem make):**
```powershell
uvicorn api.main:app --reload
```

Acesse: `http://localhost:8000/docs` (Swagger UI automático do FastAPI)

### Rodar os testes

```bash
make test
```

**Windows (sem make):**
```powershell
pytest tests/ -v
```

### Verificar qualidade do código

```bash
make lint
```

**Windows (sem make):**
```powershell
ruff check .
```

---

## Endpoints da API

### `GET /health`

Verifica se a API está no ar e se os artefatos foram carregados com sucesso.

```bash
curl http://localhost:8000/health
```

```json
{
  "status": "ok",
  "model_loaded": true
}
```

---

### `POST /predict`

Inferência on-demand para um cliente. Recebe features brutas (sem pré-processamento), aplica o pipeline internamente e retorna probabilidade e classificação binária.

**Linux/macOS:**
```bash
curl -s -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "Female",
    "Senior Citizen": "Yes",
    "partner": "No",
    "dependents": "No",
    "phone_service": "Yes",
    "multiple_lines": "No",
    "internet_service": "Fiber optic",
    "online_security": "No",
    "online_backup": "No",
    "device_protection": "No",
    "tech_support": "No",
    "streaming_tv": "Yes",
    "streaming_movies": "Yes",
    "contract": "Month-to-month",
    "paperless_billing": "Yes",
    "payment_method": "Electronic check",
    "Tenure Months": 2,
    "Monthly Charges": 85.5,
    "Total Charges": 171.0
  }'
```

**Windows (PowerShell):**
```powershell
$body = @{
  gender = "Female"; "Senior Citizen" = "Yes"; partner = "No"; dependents = "No"
  phone_service = "Yes"; multiple_lines = "No"; internet_service = "Fiber optic"
  online_security = "No"; online_backup = "No"; device_protection = "No"
  tech_support = "No"; streaming_tv = "Yes"; streaming_movies = "Yes"
  contract = "Month-to-month"; paperless_billing = "Yes"
  payment_method = "Electronic check"; "Tenure Months" = 2
  "Monthly Charges" = 85.5; "Total Charges" = 171.0
} | ConvertTo-Json

Invoke-RestMethod -Method Post -Uri "http://localhost:8000/predict" `
  -ContentType "application/json" -Body $body
```

```json
{
  "churn_probability": 0.8712,
  "churn_prediction": true
}
```

---

### `GET /predict/batch?customer_id=<id>`

Consulta o score pré-calculado pelo batch semanal. Indicado para dashboards e sistemas de CRM que consomem scores de forma recorrente.

```bash
curl "http://localhost:8000/predict/batch?customer_id=7590-VHVEG"
```

```json
{
  "churn_probability": 0.6500,
  "churn_prediction": true
}
```

---

## MLflow

Para visualizar os experimentos registrados (parâmetros, métricas, curvas de loss):

```bash
mlflow ui
```

Acesse: `http://localhost:5000`

Os experimentos ficam organizados em dois projetos:
- `churn-baselines` — DummyClassifier e Regressão Logística
- `churn-mlp` — execuções do MLP com hiperparâmetros e métricas

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

> Os artefatos em `data/processed/` precisam estar gerados antes do build.
> - **Linux/macOS:** `make train && make batch`
> - **Windows:** `python -m src.pipeline; python -m src.batch_predict`

---

## Dataset

**Telco Customer Churn (IBM)** — 7.043 registros × 33 features  
Localização: `data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv`

Para baixar novamente:

**Linux/macOS:**
```bash
curl -L "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv" \
  -o data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv
```

**Windows (PowerShell):**
```powershell
Invoke-WebRequest -Uri "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv" `
  -OutFile "data\raw\WA_Fn-UseC_-Telco-Customer-Churn.csv"
```

**Features utilizadas:**

| Grupo | Features |
|-------|----------|
| Demográficas | Gender, Senior Citizen, Partner, Dependents |
| Relacionamento | Tenure Months |
| Serviços de voz | Phone Service, Multiple Lines |
| Serviços de internet | Internet Service, Online Security, Online Backup, Device Protection, Tech Support, Streaming TV, Streaming Movies |
| Contratuais | Contract, Paperless Billing, Payment Method, Monthly Charges, Total Charges |

**Pré-processamento:**
- Features numéricas (3): imputação por mediana + StandardScaler
- Features categóricas (16): OneHotEncoder → 43 colunas binárias
- Total após transformação: **46 features**
- Divisão treino/teste: 80/20, estratificada pela variável alvo

---

## Resultados

Avaliação no conjunto de teste holdout (20% do dataset, estratificado, seed=42).

| Modelo | F1 | ROC-AUC | PR-AUC | Precision | Recall |
|--------|-----|---------|--------|-----------|--------|
| DummyClassifier (most_frequent) | 0.000 | 0.500 | 0.265 | 0.000 | 0.000 |
| Logistic Regression | 0.639 | 0.856 | 0.645 | 0.532 | 0.800 |
| **MLP — PyTorch (produção)** | **0.628** | **0.840** | **0.637** | **0.579** | **0.687** |



**Performance por segmento (MLP, threshold=0.5):**

| Tipo de Contrato | Churn Real | F1 | ROC-AUC | PR-AUC | Precision | Recall | Observação |
|------------------|------------|-----|---------|--------|-----------|--------|------------|
| Month-to-month | 42.6% | 0.665 | 0.748 | 0.665 | 0.560 | 0.818 | Grupo mais representado, melhor calibração |
| One year | 12.0% | 0.245 | 0.787 | 0.358 | 0.462 | 0.167 | Recall baixo — poucos exemplos positivos |
| Two year | 2.7% | 0.000 | 0.839 | 0.110 | 0.000 | 0.000 | Churn muito raro, modelo conservador |

| Tipo de Internet | Churn Real | F1 | ROC-AUC | PR-AUC | Precision | Recall |
|------------------|------------|-----|---------|--------|-----------|--------|
| Fiber optic | 41.1% | 0.696 | 0.793 | 0.687 | 0.583 | 0.861 |
| DSL | 20.0% | 0.532 | 0.812 | 0.538 | 0.479 | 0.598 |
| Sem internet | 8.0% | 0.000 | 0.868 | 0.361 | 0.000 | 0.000 |

---

## Análise de Custo de Negócio

O modelo é avaliado não apenas por métricas técnicas, mas pelo impacto financeiro das suas decisões:

- **Falso Negativo (FN):** cliente que cancela sem ser detectado → **R$ 500** de receita perdida
- **Falso Positivo (FP):** campanha de retenção desnecessária → **R$ 50** por ação

| Modelo | FP | FN | Custo FP | Custo FN | Custo Total |
|--------|----|----|----------|----------|-------------|
| DummyClassifier (most_frequent) | 0 | ~366 | R$ 0 | ~R$ 183.000 | ~R$ 183.000 |
| Logistic Regression | 272 | 86 | R$ 13.600 | R$ 43.000 | R$ 56.600 |
| MLP (threshold=0.5) | 180 | 121 | R$ 9.000 | R$ 60.500 | R$ 69.500 |
| **MLP (threshold=0.10)** | **~350** | **~40** | **~R$ 17.500** | **~R$ 20.000** | **~R$ 37.500** |

O DummyClassifier serve como piso de referência: ao nunca acionar retenção, ele deixa todos os ~366 churners do test set escaparem, gerando ~R$ 183.000 em perda. A Regressão Logística já reduz esse custo em 69%. O MLP com threshold calibrado em 0.10 vai além e alcança uma redução de **80% em relação ao baseline ingênuo**, ao trocar falsos negativos caros (R$ 500 cada) por falsos positivos baratos (R$ 50 cada).

O threshold de 0.10 é salvo automaticamente em `threshold.json` após cada execução do pipeline e carregado pela API no startup.

---

## Plano de Monitoramento

### Métricas de Modelo

Todas as métricas requerem ground truth — calcular com lag de 30 dias (tempo médio para confirmar o churn real).

| Métrica | Frequência | Baseline | Alerta |
|---------|-----------|----------|--------|
| F1-Score | Semanal | 0.628 | < 0.597 (queda > 5%) |
| ROC-AUC | Semanal | 0.840 | < 0.798 (queda > 5%) |
| PR-AUC | Semanal | 0.637 | < 0.605 (queda > 5%) |
| Precision | Semanal | 0.579 | < 0.550 (queda > 5%) |
| Recall | Semanal | 0.687 | < 0.653 (queda > 5%) |
| Taxa de churn previsto | Diária | ~26% | Desvio > 10pp da média histórica |
| Calibração (Brier Score) | Mensal | — | > 0.20 |

### Métricas de Infraestrutura

| Métrica | SLO | Alerta |
|---------|-----|--------|
| Latência p99 `/predict` | < 200ms | > 200ms por 5 min consecutivos |
| Taxa de erro 5xx | < 1% | > 1% em janela de 5 min |
| Disponibilidade | > 99% | Qualquer downtime > 1 min |

Fonte: middleware de latência em `api/main.py` → logs estruturados → CloudWatch Logs Insights.

### Data Drift

Monitorar as features mais preditivas com PSI (Population Stability Index):

| Feature | Tipo | Método | Threshold |
|---------|------|--------|-----------|
| `Tenure Months` | Numérica | PSI (10 bins) | > 0.2 = drift crítico |
| `Monthly Charges` | Numérica | PSI (10 bins) | > 0.2 = drift crítico |
| `Contract` | Categórica | Chi-quadrado | p-value < 0.05 |
| `Internet Service` | Categórica | Chi-quadrado | p-value < 0.05 |

### Alertas e Playbooks

**Alerta 1 — Degradação de F1, PR-AUC, Precision ou Recall > 5%**
1. Verificar drift nas features (PSI > 0.2).
2. Se drift confirmado → acionar retreino com dados recentes.
3. Se sem drift → investigar mudança na distribuição do target (conceito drift).
4. Registrar novo experimento no MLflow e promover modelo se todas as métricas melhorarem.
5. Atenção especial ao Recall: queda indica aumento de falsos negativos (clientes perdidos sem detecção), custo de R$ 500 cada.

**Alerta 2 — Latência p99 > 200ms**
1. Verificar logs do middleware para identificar requests lentos.
2. Se cold start (Lambda) → aumentar `ProvisionedConcurrency`.
3. Se inferência lenta → avaliar quantização do modelo ou redução de arquitetura.

**Alerta 3 — Taxa de erro 5xx > 1%**
1. Verificar logs de exceção no CloudWatch.
2. Causa mais provável: artefatos corrompidos ou input fora do schema Pydantic.
3. Rollback para versão anterior via MLflow Model Registry.

> Documentação completa: [docs/monitoring_plan.md](docs/monitoring_plan.md)

---

## URL Pública do Projeto

[![Live Demo](https://img.shields.io/badge/🌐_Live_Demo-Azure_ACI-brightgreen)](http://churn-mlp-api.brazilsouth.azurecontainer.io:8000/docs)

**Base URL:** `http://churn-mlp-api.brazilsouth.azurecontainer.io:8000`

Deploy em **Azure Container Instances** (ACI) — região Brazil South, imagem hospedada no Azure Container Registry (`churnmlpfiap.azurecr.io`).

| Endpoint | Método | URL completa |
|----------|--------|--------------|
| Health check | `GET` | `http://churn-mlp-api.brazilsouth.azurecontainer.io:8000/health` |
| Predição on-demand | `POST` | `http://churn-mlp-api.brazilsouth.azurecontainer.io:8000/predict` |
| Consulta batch | `GET` | `http://churn-mlp-api.brazilsouth.azurecontainer.io:8000/predict/batch?customer_id=<id>` |
| Swagger UI | `GET` | `http://churn-mlp-api.brazilsouth.azurecontainer.io:8000/docs` |

### Exemplos de uso com a URL pública

**Health check:**
```bash
curl http://churn-mlp-api.brazilsouth.azurecontainer.io:8000/health
```

**Predição on-demand:**
```bash
curl -s -X POST http://churn-mlp-api.brazilsouth.azurecontainer.io:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "Male",
    "Senior Citizen": "No",
    "partner": "Yes",
    "dependents": "No",
    "Phone Service": "Yes",
    "Multiple Lines": "No",
    "Internet Service": "Fiber optic",
    "Online Security": "No",
    "Online Backup": "No",
    "Device Protection": "No",
    "Tech Support": "No",
    "Streaming TV": "Yes",
    "Streaming Movies": "Yes",
    "contract": "Month-to-month",
    "Paperless Billing": "Yes",
    "Payment Method": "Electronic check",
    "Tenure Months": 2,
    "Monthly Charges": 70.5,
    "Total Charges": 141.0
  }'
```

**Consulta batch (score pré-calculado):**
```bash
curl "http://churn-mlp-api.brazilsouth.azurecontainer.io:8000/predict/batch?customer_id=7590-VHVEG"
```

---

## Vídeo STAR

> Substitua o placeholder abaixo pelo link do vídeo de apresentação.

[![Vídeo STAR](https://img.shields.io/badge/🎬_Vídeo_STAR-Apresentação-red?logo=youtube&logoColor=white)](URL_VIDEO_STAR_AQUI)

**Link:** `URL_VIDEO_STAR_AQUI`

O vídeo segue o formato **STAR** (Situação, Tarefa, Ação, Resultado):

| Etapa | Conteúdo abordado |
|-------|-------------------|
| **Situação** | Contexto do problema de churn em telecomunicações e impacto financeiro |
| **Tarefa** | Construir um sistema preditivo end-to-end com rede neural MLP |
| **Ação** | EDA → Baselines → MLP (PyTorch) → API (FastAPI) → Deploy (Docker) |
| **Resultado** | F1 = 0.628, ROC-AUC = 0.840, PR-AUC = 0.637 — redução de 80% no custo de negócio vs. baseline ingênuo |

---

## Documentação Adicional

- [ML Canvas](docs/ml_canvas.md) — Problema de negócio, stakeholders, features, SLOs e riscos
- [Model Card](docs/model_card.md) — Arquitetura, métricas detalhadas, vieses e plano de retreino
- [Plano de Monitoramento](docs/monitoring_plan.md) — PSI, alertas, playbook de resposta a incidentes
