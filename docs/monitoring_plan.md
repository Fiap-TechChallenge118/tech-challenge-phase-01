# Plano de Monitoramento — Churn MLP

## 1. Métricas de Modelo

| Métrica | Frequência | Alerta |
|---|---|---|
| F1-Score (produção) | Semanal | < 0.58 (queda > 5% vs baseline 0.628) |
| ROC-AUC | Semanal | < 0.79 (queda > 5% vs baseline 0.840) |
| Taxa de churn previsto | Diária | Desvio > 10pp da média histórica (~26%) |
| Calibração (Brier Score) | Mensal | > 0.20 |

> Métricas de modelo requerem ground truth — calcular com lag de 30 dias (tempo médio para confirmar churn real).

---

## 2. Métricas de Infraestrutura

| Métrica | SLO | Alerta |
|---|---|---|
| Latência p99 `/predict` | < 200ms | > 200ms por 5 min consecutivos |
| Taxa de erro 5xx | < 1% | > 1% em janela de 5 min |
| Disponibilidade | > 99% | Qualquer downtime > 1 min |

Fonte: middleware de latência já implementado em `api/main.py` → logs estruturados → CloudWatch Logs Insights.

---

## 3. Data Drift

Monitorar distribuição das features mais preditivas com **PSI (Population Stability Index)**:

| Feature | Tipo | Método | Threshold |
|---|---|---|---|
| `Tenure Months` | Numérica | PSI (10 bins) | > 0.2 = drift crítico |
| `Monthly Charges` | Numérica | PSI (10 bins) | > 0.2 = drift crítico |
| `Contract` | Categórica | Chi-quadrado | p-value < 0.05 |
| `Internet Service` | Categórica | Chi-quadrado | p-value < 0.05 |

**Referência:** distribuição do dataset de treino (`data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv`).  
**Ferramenta sugerida:** [Evidently AI](https://www.evidentlyai.com/) — gera relatórios HTML de drift com uma chamada.

```python
# Exemplo de uso com Evidently
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

report = Report(metrics=[DataDriftPreset()])
report.run(reference_data=df_train, current_data=df_production)
report.save_html("drift_report.html")
```

---

## 4. Alertas e Playbook de Resposta

### Alerta 1 — Degradação de F1 > 5%
1. Verificar se há drift nas features (PSI > 0.2).
2. Se drift confirmado → acionar retreino com dados recentes.
3. Se sem drift → investigar mudança na distribuição do target (conceito drift).
4. Registrar novo experimento no MLflow e promover modelo se F1 melhorar.

### Alerta 2 — Latência p99 > 200ms
1. Verificar logs do middleware para identificar requests lentos.
2. Se problema de cold start (Lambda) → aumentar `ProvisionedConcurrency`.
3. Se problema de inferência → avaliar quantização do modelo ou redução de arquitetura.

### Alerta 3 — Taxa de erro 5xx > 1%
1. Verificar logs de exceção no CloudWatch.
2. Causa mais provável: artefatos corrompidos ou input fora do schema.
3. Rollback para versão anterior via MLflow Model Registry.

---

## 5. Ferramentas

| Ferramenta | Uso |
|---|---|
| AWS CloudWatch | Coleta de logs estruturados, métricas de infraestrutura, alarmes |
| Evidently AI | Relatórios de data drift e qualidade de dados |
| MLflow | Rastreamento de experimentos, versionamento de modelos, comparação de runs |
