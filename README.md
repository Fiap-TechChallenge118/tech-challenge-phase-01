# Churn MLP — Tech Challenge Fase 01

Rede Neural (MLP) para previsão de churn em operadora de telecomunicações.  
Pipeline end-to-end: EDA → Baselines → MLP (PyTorch) → API (FastAPI) → Deploy (AWS ECS Fargate).

---

## Arquitetura do Projeto

```
churn-mlp/
├── data/
│   ├── raw/                  # Dataset original (não versionado)
│   └── processed/            # Artefatos gerados (preprocessor.pkl, model.pth, threshold.json)
├── notebooks/
│   ├── 01_eda.ipynb          # Análise exploratória
│   ├── 02_baseline.ipynb     # DummyClassifier + LogisticRegression
└── 03_mlp_training.ipynb # Treinamento e avaliação da MLP
├── src/
│   ├── preprocessing.py      # ColumnTransformer, load_and_split
│   ├── model.py              # ChurnMLP (nn.Module) + ChurnMLPWrapper (sklearn)
│   ├── train.py              # Loop de treino + early stopping
│   ├── evaluate.py           # Métricas + análise de custo FP/FN
│   ├── pipeline.py           # Orquestrador end-to-end (treino + artefatos + S3)
│   └── batch_predict.py      # Predição em lote para todos os clientes
├── api/
│   ├── main.py               # FastAPI app (/health, /predict, /predict/batch)
│   └── schemas.py            # Pydantic models (validação de input + output)
├── tests/
│   ├── test_smoke.py         # Forward pass da MLP
│   ├── test_schema.py        # Validação do dataset com pandera
│   └── test_api.py           # Testes dos endpoints
├── docs/
│   ├── ml_canvas.md          # ML Canvas (stakeholders, métricas, SLOs)
│   ├── model_card.md         # Model Card (limitações, vieses, métricas)
│   └── monitoring_plan.md    # Plano de monitoramento e alertas
├── infra/                    # Terraform (VPC, ECR, ECS Fargate, ALB, S3)
├── examples/
│   ├── payloads.json         # Payloads de exemplo (alto e baixo risco)
│   └── curl_examples.sh      # Exemplos de chamada com curl
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

---

## Setup Local

### 1. Criar e ativar o ambiente virtual

```bash
uv venv .venv
source .venv/bin/activate
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
| `make test` | Roda todos os testes com pytest |
| `make lint` | Verifica qualidade do código com ruff |
| `make run` | Sobe a API localmente com hot-reload |
| `make batch` | Executa predição em lote para todos os clientes |
| `make tf-init` | Inicializa o Terraform e conecta ao state remoto |
| `make tf-plan` | Mostra o plano de execução sem aplicar |
| `make tf-apply` | Provisiona toda a infraestrutura na AWS |
| `make tf-destroy` | Destrói todos os recursos AWS |
| `make ecr-push` | Build + push da imagem Docker + reinicia o ECS service |
| `make artifacts-push` | Upload manual dos artefatos para S3 |

---

## Desenvolvimento Local

### Treinar o modelo

```bash
make train
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
INFO src.pipeline — Threshold ótimo salvo em data/processed/threshold.json — threshold=0.3800
```

Artefatos gerados em `data/processed/`:
- `preprocessor.pkl` — ColumnTransformer fitado no treino
- `model.pth` — pesos da rede (state_dict)
- `threshold.json` — threshold calibrado por análise de custo (FN=R$500, FP=R$50)

### Rodar os testes

```bash
make test
```

Output esperado:
```
tests/test_smoke.py::test_forward_pass PASSED
tests/test_smoke.py::test_output_shape PASSED
tests/test_schema.py::test_dataset_schema PASSED
tests/test_api.py::test_health PASSED
tests/test_api.py::test_predict_valid_payload PASSED
tests/test_api.py::test_predict_invalid_payload PASSED

6 passed in X.XXs
```

### Rodar a API localmente

```bash
make run
```

Output esperado:
```
INFO:     Artefatos carregados — source=local input_dim=46 threshold=0.3800
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
```

Acesse `http://localhost:8000/docs` para o Swagger UI interativo.

### Verificar qualidade do código

```bash
make lint
# Esperado: sem output (zero erros)
```

### MLflow — visualizar experimentos

```bash
mlflow ui
# Acesse: http://localhost:5000
```

---

## API — Endpoints

### `GET /health`

Verifica se a API está no ar e se os artefatos foram carregados.

```bash
curl http://localhost:8000/health
```

Resposta:
```json
{
  "status": "ok",
  "model_loaded": true
}
```

### `POST /predict`

Inferência on-demand para um cliente.

```bash
curl -X POST http://localhost:8000/predict \
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
    "Monthly Charges": 95.5,
    "Total Charges": 191.0
  }'
```

Resposta:
```json
{
  "churn_probability": 0.8732,
  "churn_prediction": true
}
```

| Campo | Tipo | Descrição |
|---|---|---|
| `churn_probability` | float [0–1] | Score contínuo de probabilidade de churn |
| `churn_prediction` | bool | `true` se `churn_probability >= threshold` calibrado por custo |

### `GET /predict/batch`

Consulta score pré-calculado pelo batch semanal sem executar o modelo.

```bash
curl "http://localhost:8000/predict/batch?customer_id=7590-VHVEG"
```

Resposta:
```json
{
  "churn_probability": 0.7234,
  "churn_prediction": true
}
```

> Requer que `make batch` tenha sido executado antes. Retorna 404 se o cliente não existir na base.

Payloads de exemplo (alto e baixo risco) disponíveis em `examples/payloads.json`.

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

O Terraform é um binário Go independente e precisa ser instalado separadamente.
A forma recomendada é via repositório oficial HashiCorp:

```bash
# Adicionar repositório HashiCorp
wget -O- https://apt.releases.hashicorp.com/gpg | \
  sudo gpg --dearmor -o /usr/share/keyrings/hashicorp-archive-keyring.gpg

echo "deb [signed-by=/usr/share/keyrings/hashicorp-archive-keyring.gpg] \
  https://apt.releases.hashicorp.com $(lsb_release -cs) main" | \
  sudo tee /etc/apt/sources.list.d/hashicorp.list

sudo apt-get update && sudo apt-get install -y terraform

# Verificar
terraform version
# Terraform v1.x.x
```

> Para outros sistemas operacionais: [developer.hashicorp.com/terraform/install](https://developer.hashicorp.com/terraform/install)

**1. Configurar credenciais**

```bash
aws configure
# AWS Access Key ID:     <sua access key>
# AWS Secret Access Key: <sua secret key>
# Default region name:   us-east-2
# Default output format: json
```

**2. Configurar o profile no `.env`**

```bash
cp .env.example .env
# editar .env e definir AWS_PROFILE com o nome do seu profile
```

Para verificar qual conta está ativa:
```bash
aws sts get-caller-identity
```

**3. Criar o bucket de state remoto do Terraform** (uma única vez por conta AWS)

O bucket precisa existir antes do `make tf-init`. Contas com políticas restritivas
(ex: AWS Academy / Vocareum) não permitem criar buckets via CLI — use o Console:

1. Acesse [AWS Console → S3 → Create bucket](https://s3.console.aws.amazon.com/s3/bucket/create)
2. **Bucket name:** `churn-mlp-tfstate-tc1` (ou `churn-mlp-tfstate-{project_id}` se alterou o `project_id`)
3. **Region:** `us-east-2`
4. Em **Bucket Versioning**, selecione **Enable**
5. Clique em **Create bucket**

> Em contas sem restrições de IAM, é possível criar via CLI:
> ```bash
> aws s3api create-bucket --bucket churn-mlp-tfstate-tc1 \
>   --region us-east-2 --create-bucket-configuration LocationConstraint=us-east-2
> aws s3api put-bucket-versioning --bucket churn-mlp-tfstate-tc1 \
>   --versioning-configuration Status=Enabled
> ```

**4. Configurar variáveis do Terraform**

```bash
cp infra/terraform.tfvars.example infra/terraform.tfvars
# Ajustar project_id se necessário para evitar colisão de nomes
```

### Primeiro deploy

```bash
make init
```

Output esperado:
```
[1/3] Baixando dataset...
[2/3] Inicializando Terraform...
Initializing the backend...
Successfully configured the backend "s3"!
[3/3] Provisionando infra na AWS...
aws_vpc.main: Creating...
aws_ecr_repository.app: Creating...
aws_s3_bucket.artifacts: Creating...
...
Apply complete! Resources: 18 added, 0 changed, 0 destroyed.

Outputs:

api_url              = "http://churn-mlp-XXXXXXXXXXXX.us-east-2.elb.amazonaws.com"
artifacts_bucket     = "churn-mlp-artifacts-tc1"
cloudwatch_log_group = "/ecs/churn-mlp"
ecr_repository_url   = "XXXXXXXXXXXX.dkr.ecr.us-east-2.amazonaws.com/churn-mlp"
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
INFO src.pipeline — Threshold ótimo salvo em data/processed/threshold.json — threshold=0.3800
INFO src.pipeline — Upload concluído — s3://churn-mlp-artifacts-tc1/models/latest/model.pth
INFO src.pipeline — Upload concluído — s3://churn-mlp-artifacts-tc1/models/latest/preprocessor.pkl
INFO src.pipeline — Upload concluído — s3://churn-mlp-artifacts-tc1/models/latest/threshold.json
[2/2] Build, push da imagem Docker e reinicializando ECS...
Login Succeeded
Successfully tagged XXXXXXXXXXXX.dkr.ecr.us-east-2.amazonaws.com/churn-mlp:latest
The push refers to repository [...]
latest: digest: sha256:... size: ...
{
    "service": "churn-mlp",
    "clusterArn": "arn:aws:ecs:us-east-2:..."
}
✓ deploy concluído
```

Após o deploy, aguarde ~60s para o container subir e testar:

```bash
# A URL é exibida nos outputs do tf-apply e também pode ser consultada a qualquer momento:
terraform -chdir=infra output api_url

curl $(terraform -chdir=infra output -raw api_url)/health
# {"status":"ok","model_loaded":true}
```

### Fluxo de retreino

Para atualizar o modelo em produção sem recriar a infra:

```bash
make deploy
# Retreina, sobe novos artefatos para S3 e reinicia o container ECS automaticamente
```

### Monitorar logs do container

```bash
aws logs tail $(terraform -chdir=infra output -raw cloudwatch_log_group) --follow
```

### Destruir a infraestrutura

```bash
make tf-destroy
# Destrói todos os recursos AWS em ~2 minutos
```

### Outros membros do time

```bash
cp infra/terraform.tfvars.example infra/terraform.tfvars
cp .env.example .env  # definir AWS_PROFILE
make tf-init          # baixa o state do S3 automaticamente
```

---

## Dataset

**Telco Customer Churn (IBM)** — 7.043 registros × 33 features  
Localização: `data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv`

O dataset é baixado automaticamente pelo `make init`. Para baixar manualmente:

```bash
curl -L "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv" \
  -o data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv
```

---

## Resultados

Validação cruzada estratificada (StratifiedKFold, k=5) no conjunto de treino.

| Modelo | F1 | ROC-AUC | Precision | Recall |
|---|---|---|---|---|
| DummyClassifier (most_frequent) | 0.000 | 0.500 | 0.000 | 0.000 |
| Logistic Regression | 0.639 | 0.856 | 0.532 | 0.800 |
| **MLP — PyTorch** (produção) | **0.628** | **0.840** | **0.579** | **0.687** |

O MLP apresenta precision superior (+4.7pp vs LR), reduzindo campanhas de retenção desnecessárias.  
O threshold é calibrado por análise de custo (FN=R$500, FP=R$50) e salvo em `data/processed/threshold.json`.

---

## Documentação Adicional

- [ML Canvas](docs/ml_canvas.md)
- [Model Card](docs/model_card.md)
- [Plano de Monitoramento](docs/monitoring_plan.md)
