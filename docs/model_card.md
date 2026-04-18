# Model Card — Churn MLP

## Descrição do Modelo

| Campo | Valor |
|---|---|
| Tipo | Rede Neural MLP (Multi-Layer Perceptron) |
| Framework | PyTorch 2.x |
| Tarefa | Classificação binária (churn = 1 / não-churn = 0) |
| Dataset | Telco Customer Churn — IBM (7.043 registros, 19 features) |
| Data de treino | Abril 2025 |
| Versão | 1.0.0 |

### Arquitetura

```
Input (46) → Linear(64) → BatchNorm → ReLU → Dropout(0.3)
           → Linear(32) → BatchNorm → ReLU → Dropout(0.3)
           → Linear(1)  → Sigmoid
```

- Loss: `BCELoss` com `pos_weight` para compensar desbalanceamento (~26% churn)
- Optimizer: Adam (lr=0.001)
- Early stopping: patience=10 epochs
- Seed fixado: 42

---

## Métricas Finais (test set — 20% holdout, estratificado)

| Modelo | F1 | ROC-AUC | Precision | Recall |
|---|---|---|---|---|
| DummyClassifier (most_frequent) | 0.000 | 0.500 | 0.000 | 0.000 |
| Logistic Regression | 0.639 | 0.856 | 0.532 | 0.800 |
| **MLP (produção)** | **0.628** | **0.840** | **0.579** | **0.687** |

O MLP apresenta precision superior à Regressão Logística (+4.7pp) com recall menor (-11.3pp). Para o problema de churn, onde falsos negativos têm custo maior, a Regressão Logística tem vantagem em recall — o threshold do MLP pode ser ajustado para equilibrar esse trade-off.

---

## Uso Pretendido

- **Recomendado:** Priorização de clientes para campanhas de retenção proativa.
- **Não recomendado:** Decisões automatizadas sem revisão humana (ex: cancelamento de contratos, bloqueio de serviços).
- **Fora do escopo:** Clientes de outros segmentos ou países com perfis demográficos distintos do dataset original.

---

## Limitações e Vieses

- **Distribuição geográfica:** Dataset restrito ao mercado norte-americano (IBM). Pode não generalizar para outros mercados.
- **Desbalanceamento:** ~26% de churn no dataset. O modelo usa `pos_weight` para compensar, mas pode ter performance degradada em datasets com proporções muito diferentes.
- **Features ausentes:** Não inclui dados comportamentais (ex: frequência de uso, tickets de suporte), que podem ser preditores relevantes.
- **Viés demográfico:** Variáveis como `Senior Citizen`, `Partner` e `Dependents` podem introduzir viés por perfil demográfico. O modelo não foi auditado para fairness por subgrupo.
- **Drift temporal:** O dataset é estático. Em produção, a distribuição de features pode mudar ao longo do tempo (ver plano de monitoramento).

---

## Cenários de Falha

| Cenário | Impacto | Mitigação |
|---|---|---|
| Input com features fora do range de treino | Predição degradada silenciosamente | Validação Pydantic na API + monitoramento de PSI |
| Novo valor categórico não visto no treino | `OneHotEncoder` ignora (`handle_unknown="ignore"`) — feature zerada | Alertar quando categoria desconhecida for frequente |
| Modelo não carregado no startup | API retorna 503 | `/health` expõe `model_loaded: false` |
| Dataset com proporção de churn muito diferente | Threshold 0.5 pode ser inadequado | Recalibrar threshold com dados de produção |

---

## Plano de Retreino

- **Frequência:** Mensal ou quando F1 em produção cair > 5% em relação ao baseline.
- **Trigger automático:** Monitoramento de PSI (Population Stability Index) nas features numéricas principais (`Tenure Months`, `Monthly Charges`). PSI > 0.2 indica drift significativo.
- **Processo:** Reexecutar `make train` com dados atualizados → registrar novo run no MLflow → promover modelo se métricas melhorarem.
