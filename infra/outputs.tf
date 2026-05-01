# Outputs exibidos após `terraform apply`.
# Use `terraform -chdir=infra output -raw api_url` para capturar em scripts.

output "api_url" {
  description = "URL pública da API"
  value       = aws_apigatewayv2_stage.default.invoke_url
}

output "ecr_repository_url" {
  description = "URL do repositório ECR — use em `make ecr-push`"
  value       = aws_ecr_repository.app.repository_url
}

output "artifacts_bucket" {
  description = "Nome do bucket S3 para upload dos artefatos — exporte como ARTIFACTS_BUCKET antes de `make train`"
  value       = aws_s3_bucket.artifacts.bucket
}
