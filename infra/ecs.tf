# ECS Fargate — executa o container da FastAPI sem gerenciar servidores.
#
# Configurações relevantes:
#   - cpu = 1024 (1 vCPU), memory = 2048 (2GB): PyTorch CPU precisa de ~600MB em runtime;
#     2GB garante margem para o modelo + overhead do SO.
#   - assign_public_ip = true: necessário em subnet pública sem NAT Gateway.
#   - ARTIFACTS_BUCKET / ARTIFACTS_PREFIX: injetados como env vars para o download do S3.

resource "aws_ecs_cluster" "main" {
  name = var.project_name
}

resource "aws_ecs_task_definition" "app" {
  family                   = var.project_name
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"
  cpu                      = 1024
  memory                   = 2048
  execution_role_arn       = aws_iam_role.ecs_execution.arn
  task_role_arn            = aws_iam_role.ecs_task.arn

  container_definitions = jsonencode([{
    name      = var.project_name
    image     = local.ecr_image_uri
    essential = true

    portMappings = [{
      containerPort = 8000
      protocol      = "tcp"
    }]

    environment = [
      { name = "ARTIFACTS_BUCKET", value = data.aws_s3_bucket.artifacts.bucket },
      { name = "ARTIFACTS_PREFIX", value = var.artifacts_prefix }
    ]

    logConfiguration = {
      logDriver = "awslogs"
      options = {
        "awslogs-group"         = "/ecs/${var.project_name}"
        "awslogs-region"        = var.aws_region
        "awslogs-stream-prefix" = "ecs"
      }
    }
  }])
}

resource "aws_cloudwatch_log_group" "ecs" {
  name              = "/ecs/${var.project_name}"
  retention_in_days = 7
}

resource "aws_ecs_service" "app" {
  name            = var.project_name
  cluster         = aws_ecs_cluster.main.id
  task_definition = aws_ecs_task_definition.app.arn
  desired_count   = 1
  launch_type     = "FARGATE"

  network_configuration {
    subnets          = [aws_subnet.public_a.id, aws_subnet.public_b.id]
    security_groups  = [aws_security_group.ecs_task.id]
    assign_public_ip = true
  }

  load_balancer {
    target_group_arn = aws_lb_target_group.app.arn
    container_name   = var.project_name
    container_port   = 8000
  }

  # Garante que o ALB e o log group existam antes de criar o serviço
  depends_on = [aws_lb_listener.http, aws_cloudwatch_log_group.ecs]

  lifecycle {
    # Atualizações de imagem são feitas via `aws ecs update-service --force-new-deployment`
    ignore_changes = [task_definition]
  }
}
