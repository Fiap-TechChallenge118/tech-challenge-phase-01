# Data sources — leitura de dados externos sem criação de recursos.
# Separados em data.tf para deixar claro que não há side effects aqui.

# Trust policy: permite que o serviço Lambda assuma a role
data "aws_iam_policy_document" "lambda_trust" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["lambda.amazonaws.com"]
    }
  }
}

# Policy inline: permite que a Lambda leia artefatos do bucket S3
data "aws_iam_policy_document" "s3_artifacts_read" {
  statement {
    actions   = ["s3:GetObject"]
    resources = ["${aws_s3_bucket.artifacts.arn}/*"]
  }
}
