# Valores derivados das variáveis — centralizados aqui para evitar repetição entre arquivos.
# Referencie como local.<nome> em qualquer arquivo .tf do módulo.

locals {
  lambda_role_name      = "${var.project_name}-lambda-role"
  artifacts_bucket_name = "${var.project_name}-artifacts-${var.project_id}"
  ecr_image_uri         = "${aws_ecr_repository.app.repository_url}:${var.image_tag}"
}
