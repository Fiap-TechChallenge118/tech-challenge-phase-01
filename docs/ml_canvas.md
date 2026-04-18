# ML Canvas — Churn MLP

## 1. Stakeholders
- **Solicitante:** Diretoria de Retenção de Clientes
- **Usuários do modelo:** Time de CRM / Atendimento ao Cliente
- **Responsável técnico:** Time de ML

## 2. Problema de Negócio
Uma operadora de telecomunicações está perdendo clientes em ritmo acelerado.  
O objetivo é identificar **antecipadamente** quais clientes têm alto risco de cancelamento (churn) para que o time de retenção possa agir antes da perda.

## 3. Definição do Label
- **Positivo (1):** Cliente cancelou o serviço (`Churn = Yes`)
- **Negativo (0):** Cliente permaneceu ativo (`Churn = No`)
- **Desbalanceamento observado:** ~26% positivos / ~74% negativos

## 4. Dados
- **Dataset:** Telco Customer Churn (IBM)
- **Volume:** 7.043 registros × 33 features
- **Features relevantes:** tenure, MonthlyCharges, TotalCharges, Contract, InternetService, PaymentMethod, etc.
- **Features descartadas:** customerID (identificador sem valor preditivo)
- **Problema de qualidade:** TotalCharges contém espaços em branco → converter para float, preencher nulos com mediana

## 5. Métricas Técnicas
| Métrica   | Justificativa |
|-----------|---------------|
| F1-Score  | Principal — equilibra precisão e recall em dataset desbalanceado |
| PR-AUC    | Mais informativa que ROC-AUC em classes desbalanceadas |
| ROC-AUC   | Comparação com baselines |
| Recall    | Crítico — minimizar falsos negativos (churns não detectados) |

**Threshold padrão:** 0.5 (ajustável via análise de custo)

## 6. Métrica de Negócio
- **Custo de um Falso Negativo (FN):** R$ 500 — cliente perdido sem intervenção
- **Custo de um Falso Positivo (FP):** R$ 50 — oferta de retenção desnecessária
- **Objetivo:** minimizar custo total = `FN × 500 + FP × 50`

## 7. SLOs (Service Level Objectives)
| SLO | Valor alvo |
|-----|-----------|
| Latência p99 da API | < 200ms |
| Disponibilidade | > 99% |
| F1-Score mínimo em produção | > 0.60 |
| Retreino automático trigger | F1 cai > 5% vs baseline |

## 8. Riscos e Limitações
- Dataset estático (snapshot): não captura sazonalidade ou mudanças de comportamento
- Viés geográfico: dados de uma única operadora, pode não generalizar
- Features de cobrança podem mudar com reajustes de preço
- Modelo não deve ser usado para decisões discriminatórias (ex: negar serviços)
