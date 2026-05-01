# Repositório ECR para armazenar a imagem Docker da FastAPI.
#
# A imagem contém apenas o código da aplicação (src/, api/) — os artefatos do modelo
# ficam no S3 e são baixados em runtime. Isso separa o ciclo de vida do código
# do ciclo de vida do modelo: é possível atualizar o modelo sem rebuild de imagem.
#
# force_delete = true permite destruir o repositório mesmo com imagens — facilita
# o `make tf-destroy` após a entrega sem precisar limpar manualmente.

resource "aws_ecr_repository" "app" {
  name         = var.project_name
  force_delete = true

  image_scanning_configuration {
    scan_on_push = true
  }
}
