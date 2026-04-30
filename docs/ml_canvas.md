# ML Canvas — Churn MLP

## 1. Stakeholders
- **Solicitante:** Diretoria de Retenção de Clientes
- **Usuários do modelo:** Time de CRM / Atendimento ao Cliente
- **Responsável técnico:** Time de ML

---

## 2. Problema de Negócio
Uma operadora de telecomunicações está perdendo clientes em ritmo acelerado.  
O objetivo é identificar **antecipadamente** quais clientes têm alto risco de cancelamento (churn) para que o time de retenção possa agir antes da perda.

---

## 3. Fontes de Dados (Data Sources)
- **Dataset:** Telco Customer Churn (IBM)
- **Formato:** arquivo CSV
- **Volume:** 7.043 registros × 33 features
- **Features relevantes:** tenure, MonthlyCharges, TotalCharges, Contract, InternetService, PaymentMethod, entre outras
- **Features descartadas:** customerID (identificador sem valor preditivo)
- **Problema de qualidade identificado:** TotalCharges contém espaços em branco → convertido para float, nulos preenchidos com mediana

---

## 4. Definição do Label
- **Positivo (1):** Cliente cancelou o serviço (`Churn = Yes`)
- **Negativo (0):** Cliente permaneceu ativo (`Churn = No`)
- **Desbalanceamento observado:** ~26% positivos / ~74% negativos

---

## 5. Prediction Task (Tarefa de Predição)
O modelo entrega como saída um **score de probabilidade contínuo entre 0 e 1**, representando a propensão de churn de cada cliente.

- **Output do modelo:** `P(churn | features)` — valor contínuo no intervalo [0, 1]
- **Classificação binária ("em risco" / "não risco"):** decisão de negócio aplicada via threshold sobre o score, **não é um output do modelo**
- O threshold padrão é 0.5, mas deve ser calibrado com base na análise de custo assimétrico (FN >> FP)

---

## 6. Building Models (Construção do Modelo)
- **Arquitetura:** MLP (Multi-Layer Perceptron) implementado em **PyTorch**, conforme exigência do tech challenge
- **Serialização:** modelo salvo como `.pt` e encapsulado em wrapper sklearn-compatível para integração com pipeline de serving
- **Pré-processamento:** StandardScaler para features numéricas, OneHotEncoder para features categóricas
- **Tratamento de desbalanceamento:** class_weight aplicado na loss function

O modelo retorna **apenas o score de probabilidade**. A definição do threshold de corte é responsabilidade do negócio, calibrada com base na análise de custo de FP vs. FN.

---

## 7. Métricas de Avaliação Offline
| Métrica   | Justificativa |
|-----------|---------------|
| AUC-ROC   | Avaliação global da capacidade discriminativa |
| Recall    | **Prioridade** — minimizar falsos negativos (churns não detectados), dado o custo assimétrico FN > FP |
| Precision | Controle sobre falsos positivos (ofertas desnecessárias) |
| F1-Score  | Equilíbrio entre Precision e Recall em dataset desbalanceado |
| PR-AUC    | Mais informativa que ROC-AUC em classes desbalanceadas |

**Custo assimétrico:**
- Falso Negativo (FN): R$ 500 — cliente perdido sem intervenção
- Falso Positivo (FP): R$ 50 — oferta de retenção desnecessária
- Objetivo: minimizar custo total = `FN × 500 + FP × 50`

---

## 8. Making Predictions (Inferência em Produção)
A estratégia de serving adota **batch semanal como modo principal**, com endpoint de consulta como interface para o time de negócio:

### Batch semanal (modo principal)
- Toda semana o pipeline processa **toda a base de clientes ativos**
- Os scores calculados são armazenados em banco de dados (tabela `churn_scores`)
- O modelo PyTorch serializado (`.pt`) é carregado uma vez e executado em lote

### Endpoint `/predict` (modo de consulta)
- Recebe o `customerID` como parâmetro
- **Consulta o score já calculado** no banco de dados — o modelo não é executado novamente
- Latência alvo p99 < 200ms, garantida pela consulta em banco e não pela inferência online
- Usado pelo time de CRM para ações pontuais entre os ciclos de batch

---

## 9. Monitoring (Monitoramento em Produção)

### Camada Técnica
Monitoradas a cada ciclo de retreinamento (semanal):

| Métrica        | Alerta |
|----------------|--------|
| AUC-ROC        | Queda > 5% vs baseline |
| Recall         | Queda > 5% vs baseline (prioridade por custo assimétrico) |
| Precision      | Variação > 10% |
| F1-Score mínimo | < 0.60 |
| Distribuição de features (data drift) | PSI > 0.2 em qualquer feature principal |

### Camada de Negócio
Monitoradas mensalmente em conjunto com o time de retenção:

| Métrica                          | Descrição |
|----------------------------------|-----------|
| Taxa de churn real vs. previsto  | Comparação entre clientes sinalizados como risco e os que efetivamente cancelaram |
| Conversão das ações de retenção  | % de clientes contatados que permaneceram ativos |
| Receita preservada estimada      | Valor retido com base nas ações disparadas pelo modelo |

### Ciclo de revisão
- **Retreinamento automático:** triggerizado quando F1 cai > 5% vs baseline ou PSI > 0.2
- **Revisão manual:** trimestral, com análise de custo-benefício das decisões de threshold

---

## 10. SLOs (Service Level Objectives)
| SLO | Valor alvo |
|-----|-----------|
| Latência p99 do endpoint `/predict` | < 200ms |
| Disponibilidade da API | > 99% |
| F1-Score mínimo em produção | > 0.60 |
| Trigger de retreino automático | F1 cai > 5% vs baseline |

---

## 11. Riscos e Limitações
- Dataset estático (snapshot): não captura sazonalidade ou mudanças de comportamento ao longo do tempo
- Viés geográfico: dados de uma única operadora, pode não generalizar para outros contextos
- Features de cobrança podem mudar com reajustes de preço, exigindo monitoramento de data drift
- Modelo não deve ser usado para decisões discriminatórias (ex: negar serviços a clientes)
