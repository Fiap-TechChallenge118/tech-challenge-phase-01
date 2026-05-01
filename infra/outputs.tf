# Outputs exibidos após `terraform apply`.

output "api_url" {
  description = "URL pública da API (ALB)"
  value       = "http://${aws_lb.main.dns_name}"
}

output "ecr_repository_url" {
  description = "URL do repositório ECR — use em `make ecr-push`"
  value       = aws_ecr_repository.app.repository_url
}

output "artifacts_bucket" {
  description = "Nome do bucket S3 — exporte como ARTIFACTS_BUCKET antes de `make train`"
  value       = aws_s3_bucket.artifacts.bucket
}
