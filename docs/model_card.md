# Model Card — Churn MLP

## Descrição do Modelo

| Campo          | Valor                                                       |
|----------------|-------------------------------------------------------------|
| Tipo           | Rede Neural MLP (Multi-Layer Perceptron)                    |
| Framework      | PyTorch 2.x                                                 |
| Tarefa         | Classificação binária (churn = 1 / não-churn = 0)           |
| Dataset        | Telco Customer Churn — IBM (7.043 registros, 19 features)   |
| Data de treino | Abril 2026                                                  |
| Versão         | 1.0.0                                                       |

### Arquitetura

```
Input (46) → Linear(64) → BatchNorm → ReLU → Dropout(0.3)
           → Linear(32) → BatchNorm → ReLU → Dropout(0.3)
           → Linear(1)  → Sigmoid
```

- **Loss:** `BCELoss` com `pos_weight` para compensar desbalanceamento (~26% churn)
- **Optimizer:** Adam (lr = 0.001)
- **Early stopping:** patience = 10 epochs
- **Seed fixada:** 42

---

## Métricas Finais

Avaliação no test set (20% holdout, estratificado).

| Modelo                | F1-Score   | ROC-AUC    | PR-AUC        | Precision  | Recall    |
|-----------------------|------------|------------|---------------|------------|-----------|
| DummyClassifier       | 0.000      | 0.500      | 0.265         | 0.000      | 0.000     |
| Logistic Regression   | 0.613      | 0.841      | 0.632         | 0.503      | 0.783     |
| **MLP (produção)**    | **0.623**  | **0.835**  | **0.626**     | **0.576** | **0.679**  |

O MLP apresenta precision superior à Regressão Logística (+4.7 pp) com recall menor (-11.3 pp). Para o problema de churn, em que falsos negativos têm custo maior, a Regressão Logística leva vantagem em recall. 
O threshold do MLP pode ser ajustado para equilibrar esse trade-off.

---

## Threshold de Decisão

O threshold padrão de 0.5 foi avaliado em relação a outros valores usando a função de custo de negócio (FP = R$ 50, FN = R$ 500).

| Threshold                  | Descrição                              | Custo Total Estimado |
|----------------------------|----------------------------------------|----------------------|
| **0.10 (ótimo por custo)** | Mais agressivo (captura mais churners) | **~R$ 61.000**       |
| 0.50 (padrão)              | Threshold neutro                       | R$ 69.500            |
| 0.70                       | Conservador — menos alarmes falsos     | ~R$ 99.500           |

**Threshold escolhido para produção: 0.10**

**Justificativa:** como um falso negativo (cliente que vai embora sem intervenção) custa 10× mais do que um falso positivo (campanha de retenção desnecessária), é racional reduzir o threshold para aumentar o recall. Com threshold = 0.10, o modelo captura mais clientes em risco ao custo de mais ações de retenção desnecessárias, mas o custo total de negócio cai aproximadamente 12% em relação ao threshold padrão de 0.5.

O gráfico completo de custo por threshold está disponível em `docs/threshold_cost.png`.

---

## Custo de Negócio por Tipo de Erro

### Definição dos custos

| Tipo de Erro        | Definição no Contexto                                                                | Custo Unitário |
|---------------------|--------------------------------------------------------------------------------------|----------------|
| Falso Positivo (FP) | Modelo previu churn, mas o cliente ficaria (ação de retenção desnecessária)          | R$ 50          |
| Falso Negativo (FN) | Modelo não previu churn, mas o cliente foi embora (receita perdida sem intervenção)  | R$ 500         |

O custo do FN é 10× maior que o do FP. Clientes perdidos representam receita recorrente cancelada permanentemente, enquanto ações de retenção são custos pontuais (desconto, ligação comercial).

### Custo comparativo entre modelos

Threshold = 0.5, test set (n = 1.409).

| Modelo              | Falsos Positivos | Falsos Negativos | Custo FP     | Custo FN     | Custo Total  |
|---------------------|------------------|------------------|--------------|--------------|--------------|
| MLP (PyTorch)       | 180              | 121              | R$ 9.000,00  | R$ 60.500,00 | R$ 69.500,00 |
| Logistic Regression | 272              | 86               | R$ 13.600,00 | R$ 43.500,00 | R$ 56.600,00 |

**Análise:** com threshold = 0.5, a Regressão Logística tem custo total menor porque detecta mais churners (recall mais alto), apesar de gerar mais falsos alarmes. No entanto, ao ajustar o threshold do MLP para 0.10, o custo total cai para ~R$ 61.000, aproximando-se do custo da Regressão Logística com threshold padrão.

---

## Performance por Subgrupo

### Por tipo de contrato (`Contract`)

A feature `Contract` é o preditor mais forte do dataset. A taxa de churn real varia radicalmente entre os grupos, o que impacta diretamente a performance do modelo em cada segmento.

| Contract       | n   | % Churn Real | F1    | ROC-AUC | PR-AUC | Precision | Recall | Impacto no Modelo                                                             |
|----------------|-----|--------------|-------|---------|--------|-----------|--------|-------------------------------------------------------------------------------|
| Month-to-month | 773 | 42.6%        | 0.665 | 0.748   | 0.665  | 0.560     | 0.818  | Subgrupo mais representado no treino (modelo mais calibrado aqui)             |
| One year       | 300 | 12.0%        | 0.245 | 0.787   | 0.358  | 0.462     | 0.167  | Recall menor (poucos exemplos positivos para aprender)                        |
| Two year       | 336 | 2.7%         | 0.000 | 0.839   | 0.110  | 0.000     | 0.000  | Recall muito baixo (churn extremamente raro; modelo tende a prever não-churn) |

**Risco identificado:** o modelo foi treinado majoritariamente com churners do tipo mensal. Clientes com contratos anuais ou bianuais que desenvolvem intenção de churn têm menor probabilidade de serem detectados.

### Por perfil demográfico (`Senior Citizen`)

| Senior Citizen | n     | % Churn Real | F1    | ROC-AUC | PR-AUC | Precision | Recall | Comportamento Esperado                                                                |
|----------------|-------|--------------|-------|---------|--------|-----------|--------|---------------------------------------------------------------------------------------|
| No             | 1.187 | 23.3%        | 0.605 | 0.847   | 0.625  | 0.540     | 0.688  | Grupo dominante (modelo bem calibrado)                                                |
| Yes            | 222   | 44.1%        | 0.711 | 0.776   | 0.670  | 0.603     | 0.867  | Taxa de churn elevada; modelo funciona melhor, mas variável é potencialmente sensível |

### Por serviço de internet (`Internet Service`)

| Internet Service | n   | % Churn Real | F1    | ROC-AUC | PR-AUC | Precision | Recall |
|------------------|-----|--------------|-------|---------|--------|-----------|--------|
| DSL              | 484 | 20.0%        | 0.532 | 0.812   | 0.538  | 0.479     | 0.598  |
| Fiber optic      | 613 | 41.1%        | 0.696 | 0.793   | 0.687  | 0.583     | 0.861  |
| No               | 312 | 8.0%         | 0.000 | 0.868   | 0.361  | 0.000     | 0.000  |

Clientes de fibra têm churn alto e são bem representados no treino; o modelo tende a ser mais agressivo em prever churn para este segmento.

---

## Uso Pretendido

- **Recomendado:** priorização de clientes para campanhas de retenção proativa.
- **Não recomendado:** decisões automatizadas sem revisão humana (ex.: cancelamento de contratos, bloqueio de serviços).
- **Fora do escopo:** clientes de outros segmentos ou países com perfis demográficos distintos do dataset original.

---

## Limitações e Vieses

- **Distribuição geográfica:** dataset restrito ao mercado norte-americano (IBM). Pode não generalizar para outros mercados.
- **Desbalanceamento:** ~26% de churn no dataset. O modelo usa `pos_weight` para compensar, mas pode ter performance degradada em datasets com proporções muito diferentes.
- **Features ausentes:** não inclui dados comportamentais (ex.: frequência de uso, tickets de suporte), que podem ser preditores relevantes.
- **Drift temporal:** o dataset é estático. Em produção, a distribuição de features pode mudar ao longo do tempo (ver plano de monitoramento).
- **Viés demográfico:** variáveis como `Senior Citizen`, `Partner` e `Dependents` podem introduzir viés por perfil demográfico. O modelo não foi auditado para fairness por subgrupo.
- **Viés por tipo de contrato:** clientes com contrato mensal representam ~55% da base, com taxa de churn de 43%. O modelo foi exposto majoritariamente a churners deste perfil.
- **Viés por método de pagamento:** cheque eletrônico tem 45% de churn vs. 15% de débito automático — o modelo associa esse método de pagamento a risco elevado.
- **Viés demográfico (sêniors):** clientes sêniors têm 44.1% de churn vs. 23.3% de não-sêniors (confirmado nos dados de subgrupo). A variável está incluída no modelo e influencia diretamente as predições.

---

## Cenários de Falha

| Cenário                                        | Impacto                                                          | Mitigação                                            |
|------------------------------------------------|------------------------------------------------------------------|------------------------------------------------------|
| Input com features fora do range de treino     | Predição degradada silenciosamente                               | Validação Pydantic na API + monitoramento de PSI     |
| Novo valor categórico não visto no treino      | OneHotEncoder ignora (`handle_unknown="ignore"`) — feature zerada | Alertar quando categoria desconhecida for frequente  |
| Modelo não carregado no startup                | API retorna 503                                                  | `/health` expõe `model_loaded: false`                |
| Dataset com proporção de churn muito diferente | Threshold 0.5 pode ser inadequado                                | Recalibrar threshold com dados de produção           |

---

## Plano de Retreino

- **Frequência:** mensal, ou quando o F1 em produção cair mais de 5% em relação ao baseline.
- **Trigger automático:** monitoramento de PSI (Population Stability Index) nas features numéricas principais (`Tenure Months`, `Monthly Charges`). PSI > 0.2 indica drift significativo.
- **Processo:** reexecutar `make train` com dados atualizados → registrar novo run no MLflow → promover modelo se as métricas melhorarem.