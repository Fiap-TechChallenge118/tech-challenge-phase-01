# Bucket S3 para armazenar os artefatos do modelo treinado.
#
# O bucket deve ser criado MANUALMENTE no AWS Console antes do `make tf-apply`.
# Contas Academy não permitem s3:CreateBucket via API/Terraform.
#
# Passos para criar:
#   1. Acesse: https://s3.console.aws.amazon.com/s3/bucket/create
#   2. Nome: churn-mlp-artifacts-{project_id}  (ex: churn-mlp-artifacts-tc1)
#   3. Região: mesma de var.aws_region
#   4. Habilite Bucket Versioning
#   5. Mantenha Block all public access marcado
#
# O Terraform não cria nem destrói este bucket — apenas lê seus atributos.
# `make tf-destroy` não apaga o bucket nem os artefatos.

data "aws_s3_bucket" "artifacts" {
  bucket = local.artifacts_bucket_name
}
