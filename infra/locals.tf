# Valores derivados das variáveis — centralizados aqui para evitar repetição entre arquivos.

locals {
  artifacts_bucket_name = "${var.project_name}-artifacts-tc1"
  ecr_image_uri         = "${aws_ecr_repository.app.repository_url}:${var.image_tag}"
}
