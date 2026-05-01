# ML Canvas — Churn MLP

## 1. Stakeholders
- **Solicitante:** Diretoria de Negócios
- **Usuários do modelo:** Time de retenção de clientes da operadora
- **Responsável técnico:** Time de ML

## 2. Problema de Negócio
Uma operadora de telecomunicações está perdendo clientes em ritmo
acelerado e sem capacidade de prever quais clientes estão em risco
antes do cancelamento. O objetivo é identificar **antecipadamente**
quais clientes têm alto risco de churn para que o time de retenção
possa agir proativamente, reduzindo custos de aquisição de novos
clientes.

## 3. Definição do Label
- **Positivo (1):** Cliente cancelou o serviço (`Churn = Yes`)
- **Negativo (0):** Cliente permaneceu ativo (`Churn = No`)
- **Desbalanceamento observado:** ~26% positivos / ~74% negativos

## 4. Output do Modelo
O modelo retorna uma **probabilidade contínua de churn (score 0–1)**.
A classificação binária (em risco / não em risco) é uma **decisão de
negócio** aplicada via threshold de corte, calibrado com base na
análise de custo de Falso Positivo vs. Falso Negativo.

**Threshold padrão:** 0.5 (ajustável — dado o custo assimétrico
FN > FP, o threshold deve favorecer recall)

## 5. Dados
- **Dataset:** Telco Customer Churn (IBM) — arquivo CSV
  disponibilizado periodicamente para retreinamento regular
- **Volume:** 7.043 registros × 33 features
- **Features relevantes:** tenure, MonthlyCharges, TotalCharges,
  Contract, InternetService, PaymentMethod, etc.
- **Features descartadas:** customerID (identificador sem valor
  preditivo)
- **Problema de qualidade:** TotalCharges contém espaços em branco
  → converter para float, preencher nulos com mediana

## 6. Features
- **Demográficas:** Gender, Senior Citizen, Partner, Dependents
- **Geográficas:** City, State, Zip Code, Latitude, Longitude
- **Relacionamento:** Tenure Months (tempo de casa em meses)
- **Serviços contratados:** Phone Service, Multiple Lines, Internet
  Service, Online Security, Online Backup, Device Protection,
  Tech Support, Streaming TV, Streaming Movies
- **Contratuais/Financeiras:** Contract, Paperless Billing,
  Payment Method, Monthly Charge, Total Charges
- **Valor do cliente:** CLTV (Customer Lifetime Value calculado
  previamente)

## 7. Métricas Técnicas
| Métrica   | Justificativa                                                                                     |
|-----------|---------------------------------------------------------------------------------------------------|
| ROC-AUC   | Comparação com baselines e monitoramento de degradação em produção                                |
| PR-AUC    | Métrica principal de avaliação — mais informativa em classes desbalanceadas (~26% positivos)      |
| Recall    | Prioridade — minimizar FN dado custo assimétrico FN > FP                                          |
| Precision | Controle de FP — custo de campanhas desnecessárias                                                |
| F1-Score  | Equilíbrio entre Precision e Recall em dataset desbalanceado                                      |

Todas as métricas técnicas monitoradas a cada ciclo de retreinamento.

## 8. Métricas de Negócio
- **Taxa de churn real vs. previsto:** aderência do modelo ao
  comportamento real da base
- **Conversão das ações de retenção:** % dos clientes classificados
  em risco que, após contato, permaneceram ativos
- **Receita preservada estimada:** LTV médio × clientes retidos

## 9. Impacto e Custo
- **Falso Negativo (FN):** cliente churn não detectado →
  custo = LTV perdido
- **Falso Positivo (FP):** cliente sem risco recebe campanha →
  custo = desconto ou ligação
- **Assimetria de custo:** FN > FP → threshold deve favorecer recall
- **Dado para simulação:** taxa de churn histórica × LTV médio ×
  volume de clientes

## 10. Making Predictions
- **Batch semanal:** execução sobre toda a base de clientes,
  gerando e armazenando os scores de churn
- **Consulta sob demanda:** endpoint `/predict` retorna a predição
  já calculada no batch mais recente, sem nova inferência — útil
  para uso em tempo real pela equipe de retenção durante atendimentos
- **Pipeline de serving:** modelo PyTorch serializado (.pt)
  encapsulado em wrapper sklearn-compatível para inferência via
  FastAPI
- **Recursos:** CPU

## 11. Decisions
- As previsões gerarão uma lista priorizada de clientes com risco
  elevado
- Integração com sistemas de dashboard para facilitar a tomada de
  decisões
- O processo será implementado em lote periódico (batch semanal)
- API REST (FastAPI) com endpoint `/predict` e validação Pydantic,
  consumível por qualquer sistema interno

## 12. SLOs (Service Level Objectives)
| SLO | Valor alvo |
|-----|-----------|
| Latência p99 da API | < 200ms |
| Disponibilidade | > 99% |
| F1-Score mínimo em produção | > 0.60 |
| Retreino automático trigger | F1 cai > 5% vs baseline |

## 13. Monitoramento
- **Alertas técnicos:** degradação de AUC-ROC abaixo do threshold
  definido; F1 cai > 5% vs baseline
- **Alertas de negócio:** queda na taxa de conversão das ações
- **Revisão:** semanal de predições
- **Retreinamento:** mensal ou quando alertas forem acionados

## 14. Riscos e Limitações
- Dataset estático (snapshot): não captura sazonalidade ou mudanças
  de comportamento ao longo do tempo
- Viés geográfico: dados de uma única operadora, pode não
  generalizar para outros contextos
- Features de cobrança podem mudar com reajustes de preço,
  impactando a distribuição dos dados de entrada
- Modelo não deve ser usado para decisões discriminatórias
  (ex: negar serviços a clientes)