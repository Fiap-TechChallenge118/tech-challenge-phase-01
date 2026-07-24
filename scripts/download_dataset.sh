#!/usr/bin/env bash
set -euo pipefail

# * Constants
DATASET_URL="https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
DEST_DIR="data/raw"
DEST_FILE="${DEST_DIR}/WA_Fn-UseC_-Telco-Customer-Churn.csv"
MIN_SIZE_KB=100

# * Ensure destination directory exists
mkdir -p "${DEST_DIR}"

# * Skip download if file already exists
if [[ -f "${DEST_FILE}" ]]; then
  EXISTING_SIZE=$(du -k "${DEST_FILE}" | cut -f1)
  echo "Dataset já existe em ${DEST_FILE} (${EXISTING_SIZE}KB) — pulando download."
  exit 0
fi

# * Download dataset
echo "Baixando dataset de ${DATASET_URL}..."
curl -L --fail --progress-bar "${DATASET_URL}" -o "${DEST_FILE}"

# * Validate minimum file size
ACTUAL_SIZE=$(du -k "${DEST_FILE}" | cut -f1)
if [[ "${ACTUAL_SIZE}" -lt "${MIN_SIZE_KB}" ]]; then
  echo "Erro: arquivo baixado tem apenas ${ACTUAL_SIZE}KB (mínimo esperado: ${MIN_SIZE_KB}KB)."
  echo "O download pode ter falhado ou retornado uma resposta de erro."
  rm -f "${DEST_FILE}"
  exit 1
fi

echo "✓ Dataset baixado com sucesso: ${DEST_FILE} (${ACTUAL_SIZE}KB)"
