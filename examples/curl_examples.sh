#!/usr/bin/env bash
# Exemplos de uso da Churn Prediction API via curl
# Pré-requisito: API rodando em http://localhost:8000 (make run)

BASE_URL="http://localhost:8000"

echo "=== GET /health ==="
curl -s "$BASE_URL/health" | python3 -m json.tool

echo ""
echo "=== POST /predict — cliente com ALTO risco de churn ==="
# Perfil: contrato mensal, fibra óptica, sem suporte técnico, sem segurança, pagamento por boleto eletrônico
curl -s -X POST "$BASE_URL/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "Female",
    "Senior Citizen": "Yes",
    "partner": "No",
    "dependents": "No",
    "Phone Service": "Yes",
    "Multiple Lines": "Yes",
    "Internet Service": "Fiber optic",
    "Online Security": "No",
    "Online Backup": "No",
    "Device Protection": "No",
    "Tech Support": "No",
    "Streaming TV": "Yes",
    "Streaming Movies": "Yes",
    "contract": "Month-to-month",
    "Paperless Billing": "Yes",
    "Payment Method": "Electronic check",
    "Tenure Months": 2,
    "Monthly Charges": 95.5,
    "Total Charges": 191.0
  }' | python3 -m json.tool

echo ""
echo "=== POST /predict — cliente com BAIXO risco de churn ==="
# Perfil: contrato de 2 anos, DSL, com suporte técnico e segurança, longa permanência
curl -s -X POST "$BASE_URL/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "Male",
    "Senior Citizen": "No",
    "partner": "Yes",
    "dependents": "Yes",
    "Phone Service": "Yes",
    "Multiple Lines": "No",
    "Internet Service": "DSL",
    "Online Security": "Yes",
    "Online Backup": "Yes",
    "Device Protection": "Yes",
    "Tech Support": "Yes",
    "Streaming TV": "No",
    "Streaming Movies": "No",
    "contract": "Two year",
    "Paperless Billing": "No",
    "Payment Method": "Bank transfer (automatic)",
    "Tenure Months": 60,
    "Monthly Charges": 55.0,
    "Total Charges": 3300.0
  }' | python3 -m json.tool

echo ""
echo "=== POST /predict — payload inválido (campo obrigatório ausente) ==="
curl -s -X POST "$BASE_URL/predict" \
  -H "Content-Type: application/json" \
  -d '{"gender": "Male"}' | python3 -m json.tool
