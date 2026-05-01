# Bucket S3 para armazenar os artefatos do modelo treinado.
#
# Fluxo de uso:
#   1. `make train` (com ARTIFACTS_BUCKET exportado) salva model.pth, preprocessor.pkl
#      e threshold.json diretamente neste bucket sob o prefixo definido em var.artifacts_prefix.
#   2. A Lambda baixa esses arquivos para /tmp no cold start e os mantém em cache
#      entre invocações do mesmo container.
#
# O versionamento está habilitado para permitir rollback de artefatos sem perda de histórico.
# O acesso público está bloqueado — apenas a role da Lambda pode ler os objetos via IAM.

resource "aws_s3_bucket" "artifacts" {
  bucket        = local.artifacts_bucket_name
  force_destroy = true
}

resource "aws_s3_bucket_versioning" "artifacts" {
  bucket = aws_s3_bucket.artifacts.id

  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_public_access_block" "artifacts" {
  bucket = aws_s3_bucket.artifacts.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}
