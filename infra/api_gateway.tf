# HTTP API Gateway — ponto de entrada público para a Lambda.
#
# Escolha: HTTP API (v2) em vez de REST API (v1).
#   - ~70% mais barato por request
#   - Latência menor (sem overhead de transformação de payload)
#   - auto_deploy = true: mudanças no stage são aplicadas imediatamente sem deploy manual
#
# A rota $default (catch-all) encaminha qualquer método/path para a Lambda.
# A FastAPI internamente roteia para /health, GET /predict e POST /predict/online.
# payload_format_version = "2.0" é necessário para compatibilidade com o Mangum ASGI adapter.

resource "aws_apigatewayv2_api" "http" {
  name          = var.project_name
  protocol_type = "HTTP"
}

resource "aws_apigatewayv2_integration" "lambda" {
  api_id                 = aws_apigatewayv2_api.http.id
  integration_type       = "AWS_PROXY"
  integration_uri        = aws_lambda_function.app.invoke_arn
  payload_format_version = "2.0"
}

resource "aws_apigatewayv2_route" "default" {
  api_id    = aws_apigatewayv2_api.http.id
  route_key = "$default"
  target    = "integrations/${aws_apigatewayv2_integration.lambda.id}"
}

resource "aws_apigatewayv2_stage" "default" {
  api_id      = aws_apigatewayv2_api.http.id
  name        = "$default"
  auto_deploy = true
}
