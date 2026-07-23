# Valores derivados das variáveis — centralizados aqui para evitar repetição entre arquivos.

locals {
  # Nome do bucket de artefatos — inclui account ID e região para garantir unicidade global.
  # Padrão gerado pelo Console AWS Academy: {project_name}-artifacts-{project_id}-{account_id}-{region}-an
  artifacts_bucket_name = "${var.project_name}-artifacts-${var.project_id}-${data.aws_caller_identity.current.account_id}-${var.aws_region}-an"
  ecr_image_uri         = "${aws_ecr_repository.app.repository_url}:${var.image_tag}"
}
