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
│   ├── iam.tf                # Roles ECS (execution + task)
│   ├── networking.tf         # VPC, subnets, IGW, security groups
│   ├── alb.tf                # Application Load Balancer
│   ├── ecs.tf                # Cluster + task definition + service Fargate
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

## Deploy AWS (ECS Fargate + ALB)

A API é servida via **ECS Fargate** (container Docker sem gerenciamento de servidor) + **Application Load Balancer**.
Os artefatos do modelo (`model.pth`, `preprocessor.pkl`, `threshold.json`) ficam em um **bucket S3** separado, carregados pelo container no startup. Isso permite atualizar o modelo sem rebuild de imagem.

> **Por que ECS Fargate e não Lambda?**
> PyTorch demora ~10-15s para importar, o que excede o timeout de init do Lambda (10s) e o limite do API Gateway (29s). O ECS Fargate não tem essas restrições — o container sobe uma vez e fica em memória.

### Estimativa de custo

| Recurso | Custo/hora | Custo/mês (24h) |
|---|---|---|
| Fargate (1 vCPU + 2GB) | ~$0.05 | ~$36 |
| ALB | ~$0.008 | ~$16 |
| ECR + S3 | — | ~$0.20 |
| **Total** | | **~$52/mês** |

> **Para uso acadêmico:** suba, grave o vídeo STAR demonstrando o endpoint (~2-3h), e destrua com `make tf-destroy`. Custo total: **~$0.20**.

### Pré-requisitos

- [Terraform](https://developer.hashicorp.com/terraform/install) >= 1.7
- [AWS CLI](https://docs.aws.amazon.com/cli/latest/userguide/install-cliv2.html)
- Docker

### Configurar credenciais AWS

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
├── main.tf           # provider AWS + backend S3 (state remoto compartilhado)
├── variables.tf      # inputs: região, project_name, image_tag
├── locals.tf         # valores derivados (bucket name, image URI)
├── data.tf           # data sources (IAM policy documents)
├── outputs.tf        # api_url (ALB), ecr_repository_url, artifacts_bucket
├── s3.tf             # bucket de artefatos do modelo (com versionamento)
├── ecr.tf            # repositório ECR para a imagem Docker
├── iam.tf            # ecs_execution role + ecs_task role (s3:GetObject)
├── networking.tf     # VPC, subnets públicas, IGW, security groups
├── alb.tf            # Application Load Balancer + target group + listener
├── ecs.tf            # cluster + task definition + service Fargate
├── terraform.tfvars          # valores reais (não commitado — ver .gitignore)
└── terraform.tfvars.example  # template para o time
```

### Primeiro deploy

**1. Criar o bucket de tfstate** (state remoto — uma única vez por conta AWS):

```bash
aws s3api create-bucket --bucket churn-mlp-tfstate-tc1 \
  --region us-east-2 --create-bucket-configuration LocationConstraint=us-east-2

aws s3api put-bucket-versioning --bucket churn-mlp-tfstate-tc1 \
  --versioning-configuration Status=Enabled
```

**2. Configurar variáveis:**

```bash
cp infra/terraform.tfvars.example infra/terraform.tfvars
```

**3. Provisionar a infraestrutura:**

```bash
make tf-init    # conecta ao state remoto
make tf-plan    # revisa o que será criado
make tf-apply   # cria VPC, ECR, S3, IAM, ALB, ECS cluster + service
```

**4. Fazer push da imagem para o ECR:**

```bash
make ecr-push   # build + push + force-new-deployment no ECS
```

**5. Treinar e publicar os artefatos no S3:**

```bash
export ARTIFACTS_BUCKET=churn-mlp-artifacts-tc1
make train      # treina e faz upload automático para S3
```

**6. Testar o endpoint público:**

```bash
API_URL=$(terraform -chdir=infra output -raw api_url)
curl $API_URL/health
```

### Fluxo de retreino

```bash
export ARTIFACTS_BUCKET=churn-mlp-artifacts-tc1
make train
# Novos artefatos sobem para s3://churn-mlp-artifacts-tc1/models/latest/
# O ECS service pega os novos artefatos no próximo restart do container
make ecr-push   # force-new-deployment para reiniciar o container
```

### Outros membros do time

```bash
cp infra/terraform.tfvars.example infra/terraform.tfvars
make tf-init    # baixa o state do S3 automaticamente
```

### Destruir a infraestrutura (após a entrega)

```bash
make tf-destroy  # destrói todos os recursos em ~2 minutos
```

### Variáveis de ambiente do container

| Variável | Descrição | Valor padrão |
|---|---|---|
| `ARTIFACTS_BUCKET` | Nome do bucket S3 com os artefatos | definido pelo Terraform |
| `ARTIFACTS_PREFIX` | Prefixo S3 dos artefatos | `models/latest` |

### Comandos Makefile

| Comando | Descrição |
|---|---|
| `make tf-init` | Inicializa o Terraform e conecta ao state remoto |
| `make tf-plan` | Mostra o plano de execução |
| `make tf-apply` | Provisiona toda a infraestrutura |
| `make tf-destroy` | Destrói todos os recursos AWS |
| `make ecr-push` | Build + push da imagem + reinicia o ECS service |
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
