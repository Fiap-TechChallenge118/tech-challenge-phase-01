# Função Lambda que serve a FastAPI via Mangum (ASGI adapter).
#
# Configurações relevantes:
#   - package_type = "Image": usa container Docker em vez de zip — necessário para PyTorch
#     que excede o limite de 250MB de layers.
#   - memory_size = 1024MB: PyTorch CPU ocupa ~600MB em runtime; 1GB garante margem.
#   - timeout = 30s: cobre o cold start (download S3 + carregamento do modelo ~5-10s)
#     mais o tempo de inferência.
#   - ignore_changes = [image_uri]: atualizações de imagem são feitas via
#     `aws lambda update-function-code` (make ecr-push), não pelo Terraform.
#     Sem isso, o Terraform reverteria a imagem para var.image_tag a cada apply.
#
# Variáveis de ambiente injetadas:
#   - ARTIFACTS_BUCKET: nome do bucket S3 com os artefatos
#   - ARTIFACTS_PREFIX: prefixo S3 (ex: models/latest)

resource "aws_lambda_function" "app" {
  function_name = var.project_name
  role          = aws_iam_role.lambda.arn
  package_type  = "Image"
  image_uri     = local.ecr_image_uri

  memory_size = 1024
  timeout     = 30

  image_config {
    # Mangum handler — converte eventos API Gateway para ASGI (FastAPI)
    command = ["api.main.handler"]
  }

  environment {
    variables = {
      ARTIFACTS_BUCKET = aws_s3_bucket.artifacts.bucket
      ARTIFACTS_PREFIX = var.artifacts_prefix
    }
  }

  lifecycle {
    # Atualizações de imagem são feitas via `aws lambda update-function-code` (make ecr-push).
    # Sem isso, o Terraform reverteria a imagem para a tag definida na variável a cada apply.
    ignore_changes = [image_uri]
  }
}

resource "aws_lambda_permission" "apigw_invoke" {
  statement_id  = "AllowAPIGatewayInvoke"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.app.function_name
  principal     = "apigateway.amazonaws.com"
  source_arn    = "${aws_apigatewayv2_api.http.execution_arn}/*/*"
}
