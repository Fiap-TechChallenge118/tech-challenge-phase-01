#!/bin/bash
# Limpa todos os recursos AWS do projeto churn-mlp para permitir
# um terraform apply limpo do zero.
# Uso: bash scripts/cleanup_aws.sh

set -e
REGION=${AWS_REGION:-us-east-1}
PROJECT=churn-mlp

echo "Limpando recursos do projeto $PROJECT na região $REGION..."
echo ""

# --- ALB ---
ALB_ARN=$(aws elbv2 describe-load-balancers \
  --names $PROJECT --region $REGION \
  --query 'LoadBalancers[0].LoadBalancerArn' \
  --output text 2>/dev/null || echo "")

if [ -n "$ALB_ARN" ] && [ "$ALB_ARN" != "None" ]; then
  echo "[1/7] Deletando ALB: $ALB_ARN"
  aws elbv2 delete-load-balancer --load-balancer-arn $ALB_ARN --region $REGION
  echo "      Aguardando deleção do ALB..."
  aws elbv2 wait load-balancers-deleted --load-balancer-arns $ALB_ARN --region $REGION
  echo "      ✓ ALB deletado"
else
  echo "[1/7] ALB não encontrado -- pulando"
fi

# --- Listeners (devem ser deletados antes do Target Group) ---
ALB_ARN_FOR_LISTENERS=$(aws elbv2 describe-load-balancers \
  --names $PROJECT --region $REGION \
  --query 'LoadBalancers[0].LoadBalancerArn' \
  --output text 2>/dev/null || echo "")

if [ -n "$ALB_ARN_FOR_LISTENERS" ] && [ "$ALB_ARN_FOR_LISTENERS" != "None" ]; then
  LISTENER_ARNS=$(aws elbv2 describe-listeners \
    --load-balancer-arn $ALB_ARN_FOR_LISTENERS --region $REGION \
    --query 'Listeners[*].ListenerArn' --output text 2>/dev/null || echo "")
  for LARN in $LISTENER_ARNS; do
    echo "[2/7] Deletando Listener: $LARN"
    aws elbv2 delete-listener --listener-arn $LARN --region $REGION
    echo "      ✓ Listener deletado"
  done
else
  echo "[2/7] Nenhum Listener encontrado -- pulando"
fi

# --- Target Group ---
TG_ARN=$(aws elbv2 describe-target-groups \
  --names $PROJECT --region $REGION \
  --query 'TargetGroups[0].TargetGroupArn' \
  --output text 2>/dev/null || echo "")

if [ -n "$TG_ARN" ] && [ "$TG_ARN" != "None" ]; then
  echo "[3/7] Deletando Target Group: $TG_ARN"
  aws elbv2 delete-target-group --target-group-arn $TG_ARN --region $REGION
  echo "      ✓ Target Group deletado"
else
  echo "[3/7] Target Group não encontrado -- pulando"
fi

# --- ECS Service ---
CLUSTER_EXISTS=$(aws ecs describe-clusters \
  --clusters $PROJECT --region $REGION \
  --query 'clusters[0].status' --output text 2>/dev/null || echo "")

if [ "$CLUSTER_EXISTS" = "ACTIVE" ]; then
  echo "[4/7] Deletando ECS Service..."
  aws ecs update-service --cluster $PROJECT --service $PROJECT \
    --desired-count 0 --region $REGION > /dev/null 2>&1 || true
  aws ecs delete-service --cluster $PROJECT --service $PROJECT \
    --force --region $REGION > /dev/null 2>&1 || true
  echo "      Deletando ECS Cluster..."
  aws ecs delete-cluster --cluster $PROJECT --region $REGION > /dev/null 2>&1 || true
  echo "      ✓ ECS deletado"
else
  echo "[4/7] ECS Cluster não encontrado -- pulando"
fi

# --- ECR ---
echo "[5/7] Deletando repositório ECR: $PROJECT"
aws ecr delete-repository --repository-name $PROJECT --force \
  --region $REGION > /dev/null 2>&1 && echo "      ✓ ECR deletado" || echo "      ECR não encontrado -- pulando"

# --- IAM ---
echo "[6/7] Deletando IAM roles..."
aws iam delete-role-policy \
  --role-name ${PROJECT}-ecs-task \
  --policy-name s3-artifacts-read 2>/dev/null || true
aws iam detach-role-policy \
  --role-name ${PROJECT}-ecs-execution \
  --policy-arn arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy 2>/dev/null || true
aws iam delete-role --role-name ${PROJECT}-ecs-execution 2>/dev/null && \
  echo "      ✓ Role ecs-execution deletada" || echo "      Role ecs-execution não encontrada -- pulando"
aws iam delete-role --role-name ${PROJECT}-ecs-task 2>/dev/null && \
  echo "      ✓ Role ecs-task deletada" || echo "      Role ecs-task não encontrada -- pulando"

# --- CloudWatch Log Group ---
echo "[7/7] Deletando Log Group: /ecs/$PROJECT"
aws logs delete-log-group --log-group-name /ecs/$PROJECT \
  --region $REGION 2>/dev/null && echo "      ✓ Log Group deletado" || echo "      Log Group não encontrado -- pulando"

# --- VPC e recursos de rede ---
echo "[8/7] Verificando VPCs do projeto..."
VPC_ID=$(aws ec2 describe-vpcs \
  --filters "Name=tag:Name,Values=$PROJECT" \
  --region $REGION \
  --query 'Vpcs[0].VpcId' --output text 2>/dev/null || echo "")

if [ -n "$VPC_ID" ] && [ "$VPC_ID" != "None" ]; then
  echo "      VPC encontrada: $VPC_ID"
  echo "      Nota: a VPC e seus recursos (subnets, IGW, SGs, route tables)"
  echo "      serão recriados pelo Terraform -- não é necessário deletar manualmente."
else
  echo "      VPC não encontrada -- pulando"
fi

echo ""
echo "✓ Limpeza concluída. Próximos passos:"
echo "  rm -f infra/terraform.tfstate infra/terraform.tfstate.backup"
echo "  make tf-apply"
