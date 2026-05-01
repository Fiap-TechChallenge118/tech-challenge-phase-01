# IAM: role de execução da Lambda e suas políticas de permissão.
#
# A role possui duas permissões:
#   1. AWSLambdaBasicExecutionRole (managed) — permite escrever logs no CloudWatch.
#   2. s3-artifacts-read (inline) — permite apenas s3:GetObject no bucket de artefatos.
#      Escopo mínimo: a Lambda só lê, nunca escreve no bucket de produção.
#
# Os data sources com as policy documents estão em data.tf.

resource "aws_iam_role" "lambda" {
  name               = local.lambda_role_name
  assume_role_policy = data.aws_iam_policy_document.lambda_trust.json
}

resource "aws_iam_role_policy_attachment" "lambda_basic_execution" {
  role       = aws_iam_role.lambda.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

resource "aws_iam_role_policy" "s3_artifacts_read" {
  name   = "s3-artifacts-read"
  role   = aws_iam_role.lambda.id
  policy = data.aws_iam_policy_document.s3_artifacts_read.json
}
