terraform {
  required_version = ">= 1.7"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }

  # State remoto em S3 — compartilhado entre todos os membros do grupo.
  # O bucket precisa existir antes do primeiro `terraform init`.
  # Crie-o manualmente uma única vez:
  #
  #   aws s3api create-bucket --bucket churn-mlp-tfstate-${var.project_id} \
  #     --region us-east-2 --create-bucket-configuration LocationConstraint=us-east-2
  #   aws s3api put-bucket-versioning --bucket churn-mlp-tfstate-${var.project_id} \
  #     --versioning-configuration Status=Enabled
  #
  # Após isso, qualquer membro do time roda `terraform init` e recebe o state atualizado.
  backend "s3" {
    bucket = "churn-mlp-tfstate-123" # Substitua pelo bucket criado, usando o project_id para evitar colisões entre times
    key    = "terraform.tfstate"
    region = "us-east-2"
  }
}

provider "aws" {
  region = var.aws_region
}
