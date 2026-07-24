# Carrega variáveis do .env se existir (ex: AWS_PROFILE, AWS_REGION, TF_VAR_aws_region)
# O hífen em -include faz o make não falhar caso o arquivo não exista
-include .env
export

AWS_REGION  ?= us-east-1
IMAGE_TAG   ?= latest

# Lê os outputs do Terraform (requer `make tf-apply` antes)
ECR_URI          := $(shell terraform -chdir=infra output -raw ecr_repository_url 2>/dev/null)
ARTIFACTS_BUCKET := $(shell terraform -chdir=infra output -raw artifacts_bucket 2>/dev/null)
ECS_CLUSTER      := $(shell terraform -chdir=infra output -raw ecs_cluster_name 2>/dev/null)
ECS_SERVICE      := $(shell terraform -chdir=infra output -raw ecs_service_name 2>/dev/null)

.PHONY: init deploy lint test run train batch tf-init tf-plan tf-apply tf-destroy ecr-push artifacts-push cleanup

## Fluxo completo de primeiro uso:
##   make init    -- baixa dataset + inicializa Terraform + provisiona infra
##   make deploy  -- treina modelo + envia artefatos para S3 + build/push imagem + reinicia ECS

init: ## Baixa dataset, inicializa e provisiona toda a infra na AWS
	@echo "[1/3] Baixando dataset..."
	@bash scripts/download_dataset.sh
	@echo "[2/3] Inicializando Terraform..."
	$(MAKE) tf-init
	@echo "[3/3] Provisionando infra na AWS..."
	$(MAKE) tf-apply
	@echo "✓ init concluído -- próximo passo: make deploy"

deploy: ## Treina o modelo, envia artefatos para S3, build/push da imagem e reinicia o ECS
	@echo "[1/2] Treinando modelo e enviando artefatos para S3..."
	$(MAKE) train
	@echo "[2/2] Build, push da imagem Docker e reinicializando ECS..."
	$(MAKE) ecr-push
	@echo "✓ deploy concluído"

destroy: ## Destrói todos os recursos AWS e limpa o state local
	@echo "Executando limpeza de recursos orphãos antes do destroy..."
	@bash scripts/cleanup_aws.sh || true
	@echo "Destruindo recursos gerenciados pelo Terraform..."
	terraform -chdir=infra destroy -auto-approve
	@rm -f infra/terraform.tfstate infra/terraform.tfstate.backup
	@echo "✓ Infra destruida e state limpo"

lint:
	ruff check .

test:
	pytest tests/

run:
	uvicorn api.main:app --reload

train:
	python -m src.pipeline

batch:
	python -m src.batch_predict

# --- Terraform ---

tf-init: ## Inicializa o Terraform com state local
	@test -f infra/terraform.tfvars || (echo "❌ infra/terraform.tfvars não encontrado. Execute: cp infra/terraform.tfvars.example infra/terraform.tfvars" && exit 1)
	terraform -chdir=infra init

tf-plan:
	terraform -chdir=infra plan

tf-apply:
	terraform -chdir=infra apply -auto-approve

tf-destroy:
	terraform -chdir=infra destroy -auto-approve


cleanup: ## Remove recursos AWS orphãos (state desatualizado) e limpa o state local
	@bash scripts/cleanup_aws.sh
	@rm -f infra/terraform.tfstate infra/terraform.tfstate.backup
	@echo "✓ Limpeza concluída -- rode make tf-apply para recriar a infra"

# --- Docker + ECR ---

ecr-push:
	aws ecr get-login-password --region $(AWS_REGION) | \
	  docker login --username AWS --password-stdin $(ECR_URI)
	docker build -t $(ECR_URI):$(IMAGE_TAG) .
	docker push $(ECR_URI):$(IMAGE_TAG)
	aws ecs update-service \
	  --cluster $(ECS_CLUSTER) \
	  --service $(ECS_SERVICE) \
	  --force-new-deployment \
	  --region $(AWS_REGION)

# --- Artefatos do modelo → S3 ---

artifacts-push:
	aws s3 cp data/processed/model.pth        s3://$(ARTIFACTS_BUCKET)/models/latest/model.pth
	aws s3 cp data/processed/preprocessor.pkl s3://$(ARTIFACTS_BUCKET)/models/latest/preprocessor.pkl
	aws s3 cp data/processed/threshold.json   s3://$(ARTIFACTS_BUCKET)/models/latest/threshold.json
