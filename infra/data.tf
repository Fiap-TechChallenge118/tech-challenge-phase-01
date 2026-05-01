# Data sources — leitura de dados externos sem criação de recursos.

# Trust policy: permite que o serviço ECS assuma as roles de execução e task
data "aws_iam_policy_document" "ecs_trust" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["ecs-tasks.amazonaws.com"]
    }
  }
}

# Policy inline: permite que o container leia artefatos do bucket S3
data "aws_iam_policy_document" "s3_artifacts_read" {
  statement {
    actions   = ["s3:GetObject"]
    resources = ["${aws_s3_bucket.artifacts.arn}/*"]
  }
}
