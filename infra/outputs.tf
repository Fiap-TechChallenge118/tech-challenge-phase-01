# Outputs exibidos após `terraform apply`.
# Todos os valores que o desenvolvedor precisa copiar ou que o Makefile consome
# devem estar aqui -- evita hardcode espalhado pelo projeto.

output "api_url" {
  description = "URL pública da API (ALB) -- acesse /docs para o Swagger"
  value       = "http://${aws_lb.main.dns_name}"
}

output "ecr_repository_url" {
  description = "URL do repositório ECR -- use em `make ecr-push`"
  value       = aws_ecr_repository.app.repository_url
}

output "artifacts_bucket" {
  description = "Nome do bucket S3 -- exporte como ARTIFACTS_BUCKET antes de `make train`"
  value       = data.aws_s3_bucket.artifacts.bucket
}

output "ecs_cluster_name" {
  description = "Nome do cluster ECS -- usado em `aws ecs update-service` e `aws ecs list-tasks`"
  value       = aws_ecs_cluster.main.name
}

output "ecs_service_name" {
  description = "Nome do service ECS -- usado em `aws ecs update-service --force-new-deployment`"
  value       = aws_ecs_service.app.name
}

output "cloudwatch_log_group" {
  description = "Nome do log group CloudWatch -- use `aws logs tail <nome> --follow` para acompanhar o container"
  value       = aws_cloudwatch_log_group.ecs.name
}
