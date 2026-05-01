# IAM: duas roles para ECS Fargate.
#
# execution_role: usada pelo agente ECS para pull da imagem ECR e envio de logs ao CloudWatch.
# task_role:      usada pelo container em runtime — apenas leitura no bucket de artefatos S3.

resource "aws_iam_role" "ecs_execution" {
  name               = "${var.project_name}-ecs-execution"
  assume_role_policy = data.aws_iam_policy_document.ecs_trust.json
}

resource "aws_iam_role_policy_attachment" "ecs_execution_policy" {
  role       = aws_iam_role.ecs_execution.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

resource "aws_iam_role" "ecs_task" {
  name               = "${var.project_name}-ecs-task"
  assume_role_policy = data.aws_iam_policy_document.ecs_trust.json
}

resource "aws_iam_role_policy" "s3_artifacts_read" {
  name   = "s3-artifacts-read"
  role   = aws_iam_role.ecs_task.id
  policy = data.aws_iam_policy_document.s3_artifacts_read.json
}
