# Inputs do módulo raiz.
# Os valores padrão cobrem o ambiente de produção do projeto.
# Para sobrescrever, edite terraform.tfvars (não commitado) ou passe via -var na CLI.

variable "aws_region" {
  description = "Região AWS onde todos os recursos serão criados"
  default     = "us-east-2"
}

variable "project_name" {
  description = "Prefixo usado em todos os recursos (ECR, Lambda, S3, IAM)"
  default     = "churn-mlp"
}

variable "image_tag" {
  description = "Tag da imagem Docker no ECR. Use 'latest' em dev ou o git SHA em CI/CD"
  default     = "latest"
}

variable "artifacts_prefix" {
  description = "Prefixo S3 onde os artefatos do modelo estão armazenados. Altere para promover uma versão específica (ex: models/v2)"
  default     = "models/latest"
}

variable "project_id" {
  description = "ID único para evitar colisões de nomes entre times. Use o sufixo do repositório Git (ex: tc1)"
  default     = "tc1"
}