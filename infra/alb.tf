# Application Load Balancer — ponto de entrada público.
# Distribui tráfego entre as tasks ECS e fornece a URL pública estável.

resource "aws_lb" "main" {
  name               = var.project_name
  internal           = false
  load_balancer_type = "application"
  security_groups    = [aws_security_group.alb.id]
  subnets            = [aws_subnet.public_a.id, aws_subnet.public_b.id]
}

resource "aws_lb_target_group" "app" {
  name        = var.project_name
  port        = 8000
  protocol    = "HTTP"
  vpc_id      = aws_vpc.main.id
  target_type = "ip" # obrigatório para Fargate

  health_check {
    path                = "/health"
    healthy_threshold   = 3   # 3 checks consecutivos OK antes de marcar healthy
    unhealthy_threshold = 3
    interval            = 60  # 60s entre checks — tolera startup lento (download S3 + PyTorch load)
    timeout             = 10
  }
}

resource "aws_lb_listener" "http" {
  load_balancer_arn = aws_lb.main.arn
  port              = 80
  protocol          = "HTTP"

  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.app.arn
  }
}
