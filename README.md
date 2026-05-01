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
├── infra/
│   ├── main.tf               # Provider AWS + backend S3
│   ├── variables.tf          # Inputs da infra
│   ├── locals.tf             # Valores derivados
│   ├── data.tf               # Data sources IAM
│   ├── outputs.tf            # Outputs: api_url, ecr_url, bucket
│   ├── s3.tf                 # Bucket de artefatos
│   ├── ecr.tf                # Repositório ECR
│   ├── iam.tf                # Role + policies da Lambda
│   ├── lambda.tf             # Função Lambda
│   ├── api_gateway.tf        # HTTP API Gateway
│   └── terraform.tfvars.example  # Template de configuração
├── Makefile                  # Atalhos: lint, test, train, run, tf-*, ecr-push
├── Dockerfile                # Imagem para deploy
└── pyproject.toml            # Única fonte de verdade (deps + ruff + pytest)
```

---

## Pré-requisitos

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) — gerenciador de pacotes e ambientes virtuais
- `make` — executor de comandos do projeto

### Instalar dependências de sistema (Debian/WSL)

```bash
sudo apt-get update && sudo apt-get install -y make
```

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

---

## Deploy AWS (Lambda + API Gateway)

A API é servida via **AWS Lambda** (container Docker) + **API Gateway HTTP API**.
Os artefatos do modelo (`model.pth`, `preprocessor.pkl`, `threshold.json`) ficam em um **bucket S3** separado, carregados pela Lambda no cold start. Isso permite atualizar o modelo sem rebuild de imagem.

### Pré-requisitos

- [Terraform](https://developer.hashicorp.com/terraform/install) >= 1.7
- [AWS CLI](https://docs.aws.amazon.com/cli/latest/userguide/install-cliv2.html) configurado (`aws configure`)
- Docker

### Configurar credenciais AWS

Antes de qualquer comando Terraform ou AWS CLI, configure suas credenciais:

```bash
aws configure
```

```
AWS Access Key ID:     <sua access key>
AWS Secret Access Key: <sua secret key>
Default region name:   us-east-2
Default output format: json
```

> Para gerar as chaves: **AWS Console → IAM → Users → seu usuário → Security credentials → Create access key**

### Estrutura da infra

```
infra/
├── main.tf           # provider AWS + backend S3 (state remoto)
├── variables.tf      # inputs: região, project_name, image_tag
├── locals.tf         # valores derivados (bucket name, image URI)
├── data.tf           # data sources (IAM policy documents)
├── outputs.tf        # api_url, ecr_repository_url, artifacts_bucket
├── s3.tf             # bucket de artefatos do modelo
├── ecr.tf            # repositório ECR para a imagem Docker
├── iam.tf            # role Lambda + policies (CloudWatch + S3)
├── lambda.tf         # função Lambda + permission para API Gateway
├── api_gateway.tf    # HTTP API + integração Lambda + stage
├── terraform.tfvars          # valores reais (não commitado)
└── terraform.tfvars.example  # template para o time
```

### Primeiro deploy (uma única vez por conta AWS)

**1. Criar o bucket de tfstate** (state remoto compartilhado entre o time):

```bash
aws s3api create-bucket --bucket churn-mlp-tfstate-{{id}} \
  --region us-east-2 --create-bucket-configuration LocationConstraint=us-east-2

aws s3api put-bucket-versioning --bucket churn-mlp-tfstate-{{id}} \
  --versioning-configuration Status=Enabled
```

**2. Configurar variáveis locais:**

```bash
cp infra/terraform.tfvars.example infra/terraform.tfvars
# edite se necessário (região, project_name, etc.)
```

**3. Provisionar a infraestrutura:**

```bash
make tf-init    # inicializa o Terraform e conecta ao state remoto
make tf-plan    # revisa o que será criado
make tf-apply   # cria ECR, S3, IAM, Lambda, API Gateway
```

**4. Fazer push da imagem Docker para o ECR:**

```bash
make ecr-push   # build + push + atualiza a Lambda automaticamente
```

**5. Treinar e publicar os artefatos no S3:**

```bash
export ARTIFACTS_BUCKET=churn-mlp-artifacts-{{id}}
make train      # treina e já faz upload para S3 automaticamente
```

**6. Testar o endpoint público:**

```bash
API_URL=$(terraform -chdir=infra output -raw api_url)
curl $API_URL/health
```

### Fluxo de retreino (versões futuras)

Quando um novo modelo for treinado, basta:

```bash
export ARTIFACTS_BUCKET=churn-mlp-artifacts-{{id}}
make train
# Os novos artefatos são enviados para s3://churn-mlp-artifacts-{{id}}/models/latest/
# A Lambda carrega os novos artefatos no próximo cold start — sem rebuild de imagem
```

### Outros membros do time

```bash
# Clonar o repo e conectar ao state remoto existente
cp infra/terraform.tfvars.example infra/terraform.tfvars
make tf-init    # baixa o state do S3 automaticamente
```

### Destruir a infraestrutura (após a entrega)

```bash
make tf-destroy
```

### Variáveis de ambiente da Lambda

| Variável | Descrição | Valor padrão |
|---|---|---|
| `ARTIFACTS_BUCKET` | Nome do bucket S3 com os artefatos | definido pelo Terraform |
| `ARTIFACTS_PREFIX` | Prefixo S3 dos artefatos | `models/latest` |

### Comandos Makefile

| Comando | Descrição |
|---|---|
| `make tf-init` | Inicializa o Terraform |
| `make tf-plan` | Mostra o plano de execução |
| `make tf-apply` | Provisiona a infraestrutura |
| `make tf-destroy` | Destrói todos os recursos |
| `make ecr-push` | Build + push da imagem + atualiza Lambda |
| `make artifacts-push` | Upload manual dos artefatos para S3 |

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
