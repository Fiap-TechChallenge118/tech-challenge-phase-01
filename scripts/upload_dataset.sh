#!/usr/bin/env bash
set -euo pipefail

# * Constants
DATASET_FILE="data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"
S3_PREFIX="dataset"
S3_FILENAME="WA_Fn-UseC_-Telco-Customer-Churn.csv"

# * Validate ARTIFACTS_BUCKET is set
if [[ -z "${ARTIFACTS_BUCKET:-}" ]]; then
  echo "Erro: ARTIFACTS_BUCKET não definido."
  echo "Execute: export ARTIFACTS_BUCKET=\$(terraform -chdir=infra output -raw artifacts_bucket)"
  exit 1
fi

# * Validate dataset file exists
if [[ ! -f "${DATASET_FILE}" ]]; then
  echo "Erro: arquivo ${DATASET_FILE} não encontrado."
  echo "Execute 'make init' ou 'bash scripts/download_dataset.sh' primeiro."
  exit 1
fi

# * Upload to S3
S3_URI="s3://${ARTIFACTS_BUCKET}/${S3_PREFIX}/${S3_FILENAME}"
echo "Fazendo upload para ${S3_URI}..."
aws s3 cp "${DATASET_FILE}" "${S3_URI}"

echo "✓ Dataset publicado em: ${S3_URI}"
