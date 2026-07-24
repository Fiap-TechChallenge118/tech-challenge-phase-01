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
![Terraform](https://img.shields.io/badge/Terraform-1.7+-7B42BC?logo=terraform&logoColor=white)
![AWS](https://img.shields.io/badge/AWS-ECS_Fargate-FF9900?logo=amazon-aws&logoColor=white)

Rede Neural (MLP) para previsão de churn em operadora de telecomunicações.
Pipeline end-to-end: EDA → Baselines → MLP (PyTorch) → API (FastAPI) → Deploy (AWS ECS Fargate).

---

## Sumário

- [Contexto do Problema](#contexto-do-problema)
- [Arquitetura da Solução](#arquitetura-da-solução)
- [Escolha do Modelo: Por Que MLP?](#escolha-do-modelo-por-que-mlp)
- [Arquitetura do Projeto](#arquitetura-do-projeto)
- [Pré-requisitos](#pré-requisitos)
- [Setup do Ambiente](#setup-do-ambiente)
- [Comandos do Projeto](#comandos-do-projeto)
- [Desenvolvimento Local](#desenvolvimento-local)
- [Endpoints da API](#endpoints-da-api)
- [MLflow](#mlflow)
- [Docker](#docker)
- [Deploy AWS](#deploy-aws)
- [Dataset](#dataset)
- [Resultados](#resultados)
- [Análise de Custo de Negócio](#análise-de-custo-de-negócio)
- [Troubleshooting](#troubleshooting)
- [Documentação Adicional](#documentação-adicional)

---

## Contexto do Problema

Uma operadora de telecomunicações enfrenta perda recorrente de clientes (churn). Cada cliente perdido representa não apenas a receita mensal deixada de receber, mas também o custo elevado de aquisição de um novo cliente para substituí-lo. O objetivo deste projeto é construir um sistema preditivo capaz de identificar, com antecedência, quais clientes apresentam maior risco de cancelamento — permitindo que o time de retenção atue de forma proativa e priorizada.

**Problema de ML:** Classificação binária supervisionada. Dado o perfil de um cliente (dados demográficos, serviços contratados e histórico de faturamento), prever se ele irá cancelar o serviço (`Churn = 1`) ou permanecer (`Churn = 0`).

**Dataset:** IBM Telco Customer Churn — 7.043 registros, 19 features relevantes, com taxa de churn de aproximadamente 26%.

---

## Arquitetura da Solução

A solução foi projetada em quatro camadas independentes e desacopladas:

**Separação de responsabilidades:** cada módulo faz uma única coisa. O preprocessador não sabe nada do modelo; o modelo não sabe nada da API.

**Artefatos versionados:** após o treino, três arquivos são salvos em `data/processed/`: o preprocessador fitado (`preprocessor.pkl`), os pesos do modelo (`model.pth`) e o threshold calibrado (`threshold.json`). A API carrega esses artefatos no startup.

**Dois modos de inferência:** inferência on-demand (`POST /predict`) e consulta em lote (`GET /predict/batch`). Essa separação evita recalcular probabilidades para o mesmo cliente em cada requisição.

**Rastreabilidade com MLflow:** cada execução do pipeline registra parâmetros, métricas e o artefato do modelo.

---

## Escolha do Modelo: Por Que MLP?

### Processo de Seleção

**Etapa 1 — Baseline ingênuo (DummyClassifier):** prevê sempre a classe majoritária. F1 = 0.000, ROC-AUC = 0.500. Serve apenas como piso de referência.

**Etapa 2 — Baseline linear (Regressão Logística):** F1 = 0.639, ROC-AUC = 0.856, PR-AUC = 0.645. É o modelo de referência competitivo que o MLP precisa superar para se justificar.

**Etapa 3 — MLP (modelo de produção):** F1 = 0.628, ROC-AUC = 0.840, PR-AUC = 0.637.

### Por Que o MLP Foi Escolhido Para Produção?

**1. Precision superior reduz custo de campanha**

O MLP apresenta precision de 0.579 contra 0.532 da Regressão Logística (+4.7pp). Para cada 100 clientes sinalizados como churn, o MLP gera aproximadamente 5 campanhas de retenção a menos desnecessárias.

**2. Capacidade de aprender interações não-lineares**

Clientes com contrato mensal e Fiber optic apresentam taxa de churn de 70%+, enquanto clientes com contrato bienal e DSL ficam abaixo de 5%. A MLP aprende automaticamente essas combinações.

**3. Extensibilidade arquitetural**

O wrapper sklearn-compatível (`ChurnMLPWrapper`) permite uso direto em `GridSearchCV` e futura expansão para features de embedding.

### Arquitetura Detalhada do MLP

```
Input(46)
    │
    ├─ Linear(46 → 64)
    ├─ BatchNorm1d(64)
    ├─ ReLU()
    ├─ Dropout(0.3)
    │
    ├─ Linear(64 → 32)
    ├─ BatchNorm1d(32)
    ├─ ReLU()
    ├─ Dropout(0.3)
    │
    └─ Linear(32 → 1)    ← logit bruto (Sigmoid aplicado externamente)
```

- **BatchNorm antes de Dropout:** normaliza a distribuição de entrada, acelerando a convergência.
- **Dropout(0.3):** com ~5.600 exemplos de treino, evita memorização.
- **pos_weight na BCELoss:** compensa o desbalanceamento (~74% negativos, ~26% positivos).
- **Early stopping (patience=10):** restaura os pesos da melhor época.

### Threshold Calibrado por Custo

| Threshold | FP | FN | Custo Total |
|-----------|----|----|-------------|
| 0.50 | 180 | 121 | R$ 69.500 |
| 0.10 (ótimo) | ~350 | ~40 | ~R$ 61.000 |

O threshold de 0.10 é salvo automaticamente em `data/processed/threshold.json`.

---

## Arquitetura do Projeto

```
churn-mlp/
├── data/
│   ├── raw/                  # Dataset original (não versionado)
│   └── processed/            # Artefatos: preprocessor.pkl, model.pth, threshold.json, scores.db
├── notebooks/
│   ├── 01_eda.ipynb          # Análise exploratória
│   ├── 02_baseline.ipynb     # DummyClassifier + LogisticRegression
│   └── 03_mlp_training.ipynb # Treinamento MLP, curva de loss, análise de threshold
├── src/
│   ├── preprocessing.py      # ColumnTransformer, load_and_split
│   ├── model.py              # ChurnMLP (nn.Module) + ChurnMLPWrapper (sklearn)
│   ├── train.py              # Loop de treino + early stopping
│   ├── evaluate.py           # Métricas + análise de custo FP/FN
│   ├── pipeline.py           # Orquestrador end-to-end (treino + artefatos + S3)
│   └── batch_predict.py      # Predição em lote → scores.db
├── api/
│   ├── main.py               # FastAPI app (/health, /predict, /predict/batch)
│   └── schemas.py            # Pydantic models (validação de input + output)
├── tests/
│   ├── conftest.py           # Fixture de cwd para testes
│   ├── test_smoke.py         # Forward pass, wrapper sklearn, carregamento de artefatos
│   ├── test_schema.py        # Validação do dataset com pandera (shape, tipos, valores)
│   └── test_api.py           # Testes dos endpoints (200, 422, 503, 404)
├── docs/
│   ├── ml_canvas.md          # ML Canvas
│   ├── model_card.md         # Model Card
│   └── monitoring_plan.md    # Plano de monitoramento
├── infra/                    # Terraform (VPC, ECR, ECS Fargate, ALB, S3)
├── examples/
│   ├── payloads.json         # Payloads de exemplo (alto e baixo risco)
│   └── curl_examples.sh      # Exemplos de chamada com curl
├── scripts/
│   ├── download_dataset.sh   # Download automático do dataset
│   └── cleanup_aws.sh        # Limpeza de recursos AWS orphãos
├── Makefile                  # Todos os comandos do projeto
├── Dockerfile                # Imagem para deploy
└── pyproject.toml            # Única fonte de verdade (deps + ruff + pytest)
```

---

## Pré-requisitos

| Ferramenta | Versão mínima | Instalação |
|---|---|---|
| Python | 3.11+ | [python.org](https://www.python.org/downloads/) |
| [uv](https://docs.astral.sh/uv/) | qualquer | `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| make | qualquer | `sudo apt-get install -y make` |
| [Terraform](https://developer.hashicorp.com/terraform/install) | 1.7+ | ver instruções abaixo |
| [AWS CLI](https://docs.aws.amazon.com/cli/latest/userguide/install-cliv2.html) | 2.x | ver link |
| Docker | qualquer | [docs.docker.com](https://docs.docker.com/get-docker/) |

> Terraform, AWS CLI e Docker são necessários apenas para o deploy na AWS.

### Instalar o uv

**Linux/macOS:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
```

**Windows (PowerShell):**
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

---

## Setup do Ambiente

### 1. Criar e ativar o ambiente virtual

**Linux/macOS:**
```bash
uv venv .venv
source .venv/bin/activate
```

**Windows (PowerShell):**
```powershell
uv venv .venv
.venv\Scripts\Activate.ps1
```

### 2. Instalar dependências

```bash
uv pip install -e ".[dev]"
```

### 3. Verificar instalação

```bash
python -c "import torch, sklearn, numpy, mlflow, fastapi; print('OK')"
```

---

## Comandos do Projeto

| Comando | O que faz |
|---|---|
| `make init` | Baixa dataset + `tf-init` + `tf-apply` (provisiona infra na AWS) |
| `make deploy` | `train` + `ecr-push` (treina, envia artefatos e sobe a imagem) |
| `make train` | Treina o pipeline completo e salva artefatos em `data/processed/` |
| `make batch` | Executa predição em lote para todos os clientes |
| `make test` | Roda todos os testes com pytest |
| `make lint` | Verifica qualidade do código com ruff |
| `make run` | Sobe a API localmente com hot-reload |
| `make tf-init` | Inicializa o Terraform (state local) |
| `make tf-plan` | Mostra o plano de execução sem aplicar |
| `make tf-apply` | Provisiona toda a infraestrutura na AWS |
| `make tf-destroy` | Destrói todos os recursos AWS |
| `make cleanup` | Remove recursos AWS orphãos e limpa o state local |
| `make ecr-push` | Build + push da imagem Docker + reinicia o ECS service |
| `make artifacts-push` | Upload manual dos artefatos para S3 |

---

## Desenvolvimento Local

### Treinar o modelo

```bash
make train
# Windows: python -m src.pipeline
```

Output esperado:
```
INFO src.preprocessing — Data split — train: 5634, test: 1409, churn rate: 26.54%
INFO src.train — Epoch 1/100 — train_loss: 0.5821 | val_loss: 0.5634
...
INFO src.train — Early stopping na epoch 47
INFO src.evaluate — Evaluation — {'f1': 0.628, 'roc_auc': 0.840, 'precision': 0.579, 'recall': 0.687}
INFO src.pipeline — Preprocessor salvo em data/processed/preprocessor.pkl
INFO src.pipeline — Model state_dict salvo em data/processed/model.pth
INFO src.pipeline — Threshold ótimo salvo em data/processed/threshold.json — threshold=0.1000
```

### Gerar scores em lote

```bash
make batch
# Windows: python -m src.batch_predict
```

> Requer dataset em `data/raw/` e artefatos em `data/processed/` (rode `make train` antes).

### Rodar a API localmente

```bash
make run
# Windows: uvicorn api.main:app --reload
```

Acesse `http://localhost:8000/docs` para o Swagger UI interativo.

### Rodar os testes

```bash
make test
# Windows: pytest tests/ -v
```

### Verificar qualidade do código

```bash
make lint
# Windows: ruff check .
```

### MLflow — visualizar experimentos

```bash
mlflow ui
# Acesse: http://localhost:5000
```

## Endpoints da API

| Rota | Método | Descrição |
|------|--------|----------|
| `/health` | `GET` | Verifica se a API está no ar e se o modelo foi carregado |
| `/predict` | `POST` | Inferência on-demand — recebe features e retorna probabilidade de churn |
| `/predict/batch` | `GET` | Consulta score pré-calculado via `?customer_id=<id>` |
| `/docs` | `GET` | Swagger UI com documentação interativa |

### `GET /health`

```bash
curl http://localhost:8000/health
```

```json
{"status": "ok", "model_loaded": true}
```

### `POST /predict`

**Linux/macOS:**
```bash
curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "Female",
    "Senior Citizen": "Yes",
    "partner": "No",
    "dependents": "No",
    "Phone Service": "Yes",
    "Multiple Lines": "Yes",
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
    "Monthly Charges": 85.5,
    "Total Charges": 171.0
  }'
```

```json
{"churn_probability": 0.8712, "churn_prediction": true}
```

| Campo | Tipo | Descrição |
|---|---|---|
| `churn_probability` | float [0–1] | Score contínuo de probabilidade de churn |
| `churn_prediction` | bool | `true` se `churn_probability >= threshold` calibrado por custo |

### `GET /predict/batch`

```bash
curl "http://localhost:8000/predict/batch?customer_id=7590-VHVEG"
```

```json
{"churn_probability": 0.6500, "churn_prediction": true}
```

> Requer que `make batch` tenha sido executado antes. Retorna 404 se o cliente não existir na base.

Payloads de exemplo disponíveis em `examples/payloads.json`.

---

## MLflow

```bash
mlflow ui
# Acesse: http://localhost:5000
```

Experimentos registrados:
- `churn-baselines` — DummyClassifier e Regressão Logística
- `churn-mlp` — execuções do MLP com hiperparâmetros e métricas

---

## Docker

### Build

```bash
docker build -t churn-api .
```

### Run local (com artefatos locais)

```bash
docker run -p 8000:8000 \
  -v $(pwd)/data/processed:/app/data/processed \
  churn-api
```

### Testar

```bash
curl http://localhost:8000/health
```

> Os artefatos em `data/processed/` precisam estar gerados antes do build (`make train`).

---

## Deploy AWS

A API é servida via **ECS Fargate** + **Application Load Balancer**.
Os artefatos do modelo ficam em um **bucket S3** separado e são baixados pelo container no startup — permitindo atualizar o modelo sem rebuild de imagem.

> **Por que ECS Fargate e não Lambda?**
> PyTorch demora ~10–15s para importar, excedendo o timeout de init do Lambda (10s) e o limite do API Gateway (29s). O ECS Fargate não tem essas restrições.

### Estimativa de custo

| Recurso | Custo/mês (24h) |
|---|---|
| Fargate (1 vCPU + 2GB) | ~$36 |
| ALB | ~$16 |
| ECR + S3 | ~$0.20 |
| **Total** | **~$52/mês** |

> **Para uso acadêmico:** suba, grave o vídeo demonstrando o endpoint (~2–3h) e destrua com `make tf-destroy`. Custo total: ~$0.20.

### Pré-requisitos AWS

**0. Instalar o Terraform** (obrigatório — não incluído no `uv`)

```bash
# Adicionar repositório HashiCorp
wget -O- https://apt.releases.hashicorp.com/gpg | \
  sudo gpg --dearmor -o /usr/share/keyrings/hashicorp-archive-keyring.gpg

echo "deb [signed-by=/usr/share/keyrings/hashicorp-archive-keyring.gpg] \
  https://apt.releases.hashicorp.com $(lsb_release -cs) main" | \
  sudo tee /etc/apt/sources.list.d/hashicorp.list

sudo apt-get update && sudo apt-get install -y terraform
terraform version
```

> Para outros sistemas operacionais: [developer.hashicorp.com/terraform/install](https://developer.hashicorp.com/terraform/install)

**1. Configurar credenciais**

```bash
aws configure
# AWS Access Key ID:     <sua access key>
# AWS Secret Access Key: <sua secret key>
# Default region name:   us-east-1
# Default output format: json
```

**2. Configurar o profile no `.env`**

```bash
cp .env.example .env
# Editar .env e definir AWS_PROFILE e AWS_REGION
```

Para verificar qual conta está ativa:
```bash
aws sts get-caller-identity
```

### Variáveis de ambiente (`.env`)

| Variável | Obrigatório | Descrição | Exemplo |
|---|---|---|---|
| `AWS_PROFILE` | Sim | Nome do profile configurado em `~/.aws/credentials` | `default`, `lab`, `prod` |
| `AWS_REGION` | Sim | Região AWS usada pelo Makefile e AWS CLI | `us-east-1` |

> Para o Terraform, a região é controlada por `aws_region` em `infra/terraform.tfvars` — mantenha consistente com `AWS_REGION` no `.env`.

**3. Configurar variáveis do Terraform**

```bash
cp infra/terraform.tfvars.example infra/terraform.tfvars
# Ajustar project_id se necessário para evitar colisão de nomes
```

> O state do Terraform é **local** (`infra/terraform.tfstate`) — não requer bucket S3.
> O arquivo de state está no `.gitignore` e não é versionado.

### Primeiro deploy

```bash
make init
```

Output esperado:
```
[1/3] Baixando dataset...
[2/3] Inicializando Terraform...
[3/3] Provisionando infra na AWS...
aws_vpc.main: Creating...
aws_ecr_repository.app: Creating...
aws_s3_bucket.artifacts: Creating...
...
Apply complete! Resources: 18 added, 0 changed, 0 destroyed.

Outputs:

api_url              = "http://churn-mlp-XXXXXXXXXXXX.us-east-1.elb.amazonaws.com"
artifacts_bucket     = "churn-mlp-artifacts-tc1"
cloudwatch_log_group = "/ecs/churn-mlp"
ecr_repository_url   = "XXXXXXXXXXXX.dkr.ecr.us-east-1.amazonaws.com/churn-mlp"
ecs_cluster_name     = "churn-mlp"
ecs_service_name     = "churn-mlp"

✓ init concluído -- próximo passo: make deploy
```

```bash
make deploy
```

Output esperado:
```
[1/2] Treinando modelo e enviando artefatos para S3...
INFO src.pipeline — Upload concluído — s3://churn-mlp-artifacts-tc1/models/latest/model.pth
INFO src.pipeline — Upload concluído — s3://churn-mlp-artifacts-tc1/models/latest/preprocessor.pkl
INFO src.pipeline — Upload concluído — s3://churn-mlp-artifacts-tc1/models/latest/threshold.json
[2/2] Build, push da imagem Docker e reinicializando ECS...
Login Succeeded
latest: digest: sha256:... size: ...
✓ deploy concluído
```

Após o deploy, aguarde ~60s para o container subir.

### Validar o deploy

**1. Verificar se o container subiu**

```bash
aws ecs describe-services \
  --cluster $(terraform -chdir=infra output -raw ecs_cluster_name) \
  --services $(terraform -chdir=infra output -raw ecs_service_name) \
  --region us-east-1 \
  --query 'services[0].{Status:status,Running:runningCount,Desired:desiredCount}'
# Esperado: Status ACTIVE, Running 1, Desired 1
```

**2. Health check**

```bash
curl $(terraform -chdir=infra output -raw api_url)/health
# {"status":"ok","model_loaded":true}
```

**3. Predição de alto risco (deve retornar `churn_prediction: true`)**

```bash
curl -X POST $(terraform -chdir=infra output -raw api_url)/predict \
  -H "Content-Type: application/json" \
  -d '{"gender":"Female","Senior Citizen":"Yes","partner":"No","dependents":"No","Phone Service":"Yes","Multiple Lines":"Yes","Internet Service":"Fiber optic","Online Security":"No","Online Backup":"No","Device Protection":"No","Tech Support":"No","Streaming TV":"Yes","Streaming Movies":"Yes","contract":"Month-to-month","Paperless Billing":"Yes","Payment Method":"Electronic check","Tenure Months":2,"Monthly Charges":95.5,"Total Charges":191.0}'
# Esperado: {"churn_probability": ~0.87, "churn_prediction": true}
```

**4. Predição de baixo risco (deve retornar `churn_prediction: false`)**

```bash
curl -X POST $(terraform -chdir=infra output -raw api_url)/predict \
  -H "Content-Type: application/json" \
  -d '{"gender":"Male","Senior Citizen":"No","partner":"Yes","dependents":"Yes","Phone Service":"Yes","Multiple Lines":"No","Internet Service":"DSL","Online Security":"Yes","Online Backup":"Yes","Device Protection":"Yes","Tech Support":"Yes","Streaming TV":"No","Streaming Movies":"No","contract":"Two year","Paperless Billing":"No","Payment Method":"Bank transfer (automatic)","Tenure Months":60,"Monthly Charges":55.0,"Total Charges":3300.0}'
# Esperado: {"churn_probability": ~0.05, "churn_prediction": false}
```

**5. Swagger UI interativo**

```bash
echo "Acesse: $(terraform -chdir=infra output -raw api_url)/docs"
```

### Fluxo de retreino

```bash
make deploy
# Retreina, sobe novos artefatos para S3 e reinicia o container ECS automaticamente
```

### Monitorar logs do container

```bash
aws logs tail $(terraform -chdir=infra output -raw cloudwatch_log_group) --follow --region us-east-1
```

### Destruir a infraestrutura

```bash
make tf-destroy
# Executa cleanup de recursos orphãos + terraform destroy + limpa o state local
```

Se o destroy travar:
```bash
make cleanup   # limpa recursos AWS via CLI e reseta o state
make tf-apply  # recria tudo do zero
```

### Outros membros do time

```bash
cp infra/terraform.tfvars.example infra/terraform.tfvars
cp .env.example .env  # definir AWS_PROFILE e AWS_REGION
make tf-init          # inicializa o Terraform com state local
```

> O state é local (`infra/terraform.tfstate`) — cada desenvolvedor mantém sua própria infra.

---

## Dataset

**Telco Customer Churn (IBM)** — 7.043 registros × 33 features
Localização: `data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv`

O dataset é baixado automaticamente pelo `make init`. Para baixar manualmente:

**Linux/macOS:**
```bash
curl -L "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv" \
  -o data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv
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

---

## Resultados

Validação cruzada estratificada (StratifiedKFold, k=5) no conjunto de treino.

| Modelo | F1 | ROC-AUC | PR-AUC | Precision | Recall |
|---|---|---|---|---|---|
| DummyClassifier (most_frequent) | 0.000 | 0.500 | 0.265 | 0.000 | 0.000 |
| Logistic Regression | 0.639 | 0.856 | 0.645 | 0.532 | 0.800 |
| **MLP — PyTorch** (produção) | **0.628** | **0.840** | **0.637** | **0.579** | **0.687** |

O MLP apresenta precision superior (+4.7pp vs LR), reduzindo campanhas de retenção desnecessárias.
O threshold é calibrado por análise de custo (FN=R$500, FP=R$50) e salvo em `data/processed/threshold.json`.

---

## Análise de Custo de Negócio

- **Falso Negativo (FN):** cliente que cancela sem ser detectado → **R$ 500** de receita perdida
- **Falso Positivo (FP):** campanha de retenção desnecessária → **R$ 50** por ação

| Modelo | FP | FN | Custo Total |
|--------|----|----|-------------|
| DummyClassifier | 0 | ~366 | ~R$ 183.000 |
| Logistic Regression | 272 | 86 | R$ 56.600 |
| MLP (threshold=0.5) | 180 | 121 | R$ 69.500 |
| **MLP (threshold=0.10)** | **~350** | **~40** | **~R$ 37.500** |

O MLP com threshold calibrado em 0.10 reduz o custo total em **80% em relação ao baseline ingênuo**, ao trocar falsos negativos caros (R$ 500 cada) por falsos positivos baratos (R$ 50 cada).

---

## Troubleshooting

### `model_loaded: false` no `/health`

O container subiu mas não encontrou os artefatos no S3.

```bash
# Verificar se os artefatos existem no bucket
aws s3 ls s3://$(terraform -chdir=infra output -raw artifacts_bucket)/models/latest/

# Se estiver vazio, fazer upload manual
make artifacts-push

# Forçar restart do container
aws ecs update-service \
  --cluster $(terraform -chdir=infra output -raw ecs_cluster_name) \
  --service $(terraform -chdir=infra output -raw ecs_service_name) \
  --force-new-deployment --region us-east-1
```

### Sessão SSO expirada

```
Error: No valid credential sources found
```

```bash
aws sso login --profile lab
aws sts get-caller-identity
```

### State desatualizado (`AlreadyExists` ou ARN inválido)

```bash
make cleanup   # limpa recursos AWS via CLI e reseta o state
make tf-apply  # recria tudo do zero
```

### Container não sobe (`Running: 0`)

```bash
aws logs tail $(terraform -chdir=infra output -raw cloudwatch_log_group) --follow --region us-east-1
```

Causas comuns:
- Artefatos ausentes no S3 — rode `make artifacts-push`
- Imagem não encontrada no ECR — rode `make ecr-push`
- Memória insuficiente — aumente `memory` em `infra/ecs.tf` (padrão: 2048MB)

### `make batch` falha

Ver pré-requisitos na seção [Desenvolvimento Local](#desenvolvimento-local) — requer dataset em `data/raw/` e artefatos em `data/processed/`.

### Terraform trava no destroy

```bash
make cleanup
```

---


## Documentação Adicional

- [ML Canvas](docs/ml_canvas.md) — Problema de negócio, stakeholders, features, SLOs e riscos
- [Model Card](docs/model_card.md) — Arquitetura, métricas detalhadas, viéses e plano de retreino
- [Plano de Monitoramento](docs/monitoring_plan.md) — PSI, alertas, playbook de resposta a incidentes
