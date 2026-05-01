AWS_REGION  ?= us-east-2
IMAGE_TAG   ?= latest

# Lê o ECR URL do output do Terraform (requer `make tf-apply` antes)
ECR_URI     := $(shell terraform -chdir=infra output -raw ecr_repository_url 2>/dev/null)
ARTIFACTS_BUCKET := $(shell terraform -chdir=infra output -raw artifacts_bucket 2>/dev/null)

.PHONY: lint test run train batch tf-init tf-plan tf-apply tf-destroy ecr-push artifacts-push

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

tf-init:
	terraform -chdir=infra init

tf-plan:
	terraform -chdir=infra plan

tf-apply:
	terraform -chdir=infra apply -auto-approve

tf-destroy:
	terraform -chdir=infra destroy -auto-approve

# --- Docker + ECR ---

ecr-push:
	aws ecr get-login-password --region $(AWS_REGION) | \
	  docker login --username AWS --password-stdin $(ECR_URI)
	docker build -t $(ECR_URI):$(IMAGE_TAG) .
	docker push $(ECR_URI):$(IMAGE_TAG)
	aws ecs update-service \
	  --cluster churn-mlp \
	  --service churn-mlp \
	  --force-new-deployment \
	  --region $(AWS_REGION)

# --- Artefatos do modelo → S3 ---

artifacts-push:
	aws s3 cp data/processed/model.pth       s3://$(ARTIFACTS_BUCKET)/models/latest/model.pth
	aws s3 cp data/processed/preprocessor.pkl s3://$(ARTIFACTS_BUCKET)/models/latest/preprocessor.pkl
	aws s3 cp data/processed/threshold.json  s3://$(ARTIFACTS_BUCKET)/models/latest/threshold.json
