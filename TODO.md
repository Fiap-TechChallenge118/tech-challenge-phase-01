# TODO List — Tech Challenge Fase 01 (MLP Churn)
> Agente: execute cada item sequencialmente. Marque `[x]` ao concluir.

---

## ETAPA 1 — Entendimento e Preparação

### 1.1 Estrutura do Projeto
- [x] Criar estrutura de pastas:
  ```
  churn-mlp/
  ├── data/raw/
  ├── data/processed/
  ├── notebooks/
  ├── src/
  │   ├── __init__.py
  │   ├── preprocessing.py
  │   ├── train.py
  │   ├── evaluate.py
  │   └── predict.py
  ├── api/
  │   ├── __init__.py
  │   ├── main.py
  │   └── schemas.py
  ├── tests/
  │   ├── test_smoke.py
  │   ├── test_schema.py
  │   └── test_api.py
  ├── docs/
  │   ├── ml_canvas.md
  │   ├── model_card.md
  │   └── monitoring_plan.md
  ├── Makefile
  ├── Dockerfile
  └── pyproject.toml
  ```
- [x] Criar `pyproject.toml` com dependências: `torch`, `scikit-learn`, `mlflow`, `fastapi`, `uvicorn`, `pydantic`, `pandas`, `numpy`, `pandera`, `pytest`, `ruff`, `joblib`
- [x] Configurar `ruff` no `pyproject.toml` (target Python 3.11, regras E/F/W/I)
- [x] Configurar `pytest` no `pyproject.toml` (testpaths = `tests/`)
- [x] Criar `.gitignore` (ignorar `data/`, `mlruns/`, `__pycache__/`, `.env`, `*.pth`, `*.pkl`)
- [x] Inicializar repositório git: `git init`
- [x] Commit: `chore: initial project structure`

### 1.2 Dataset
- [x] Baixar dataset para `data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv`:
  ```
  curl -L "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv" \
    -o data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv
  ```
- [x] Verificar: `python -c "import pandas as pd; df = pd.read_csv('data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv'); assert df.shape[0] >= 5000 and df.shape[1] >= 10, df.shape; print('OK', df.shape)"`

### 1.3 ML Canvas
- [x] Criar `docs/ml_canvas.md` com seções:
  - Stakeholders e problema de negócio
  - Definição de churn (label: coluna `Churn`, Yes=1 / No=0)
  - Métrica técnica: F1-Score (classe positiva = churn)
  - Métrica de negócio: custo evitado por churn detectado (ex: R$500/cliente)
  - SLOs: latência p99 < 200ms, disponibilidade > 99%

### 1.4 EDA
- [x] Criar `notebooks/01_eda.ipynb`
- [x] Verificar shape, dtypes, nulos por coluna
- [x] Plotar distribuição da variável target (`Churn`) — verificar desbalanceamento
- [x] Plotar distribuição de features numéricas (histogramas)
- [x] Plotar correlação entre features numéricas (heatmap)
- [x] Identificar e documentar: colunas categóricas, numéricas, colunas a dropar (ex: `customerID`)
- [x] Registrar achados no `docs/ml_canvas.md`

### 1.5 Baselines
- [x] Criar `notebooks/02_baseline.ipynb`
- [x] Definir `NUMERIC_FEATURES` e `CATEGORICAL_FEATURES` explicitamente no notebook
- [x] Converter `Total Charges` e `Monthly Charges` para numérico (`pd.to_numeric`)
- [x] Implementar `numeric_pipe`: `SimpleImputer(median)` → `StandardScaler`
- [x] Implementar `categorical_pipe`: `OneHotEncoder(handle_unknown="ignore")`
- [x] Combinar via `ColumnTransformer` (num + cat)
- [x] `train_test_split` estratificado (80/20) para avaliação direta
- [x] Treinar `DummyClassifier(strategy="most_frequent")` — baseline ingênuo
- [x] Treinar `LogisticRegression(max_iter=1000, class_weight="balanced")` — baseline linear
- [x] Avaliar ambos com `accuracy_score`, `confusion_matrix`, `classification_report`
- [x] Validação cruzada `StratifiedKFold(n_splits=5)` com F1, ROC-AUC, Precision, Recall
- [x] Logar parâmetros e métricas no MLflow (`churn-baselines`)
- [x] Commit: `feat: baseline notebook with SimpleImputer, StratifiedKFold and MLflow tracking`

---

## ETAPA 2 — Modelagem com PyTorch

### 2.1 Preparação dos Dados
- [x] Criar `src/preprocessing.py`:
  - Função `build_preprocessor()` → retorna `ColumnTransformer` (fit apenas no treino)
  - Função `load_and_split(path, test_size=0.2, seed=42)` → retorna `X_train, X_test, y_train, y_test`
- [x] Fixar seeds em `src/train.py`: `random.seed(42)`, `np.random.seed(42)`, `torch.manual_seed(42)`

### 2.2 Modelo MLP
- [x] Criar `src/model.py`:
  - Classe `ChurnMLP(nn.Module)` com `__init__(input_dim, hidden_dims, dropout)`
  - Camadas: `Linear → BatchNorm → ReLU → Dropout` repetido por hidden layer
  - Camada de saída: `Linear(hidden_dims[-1], 1)` + `Sigmoid`
  - Método `forward(x)` implementado

### 2.3 Loop de Treino
- [x] Criar `src/train.py`:
  - Função `train_model(model, X_train, y_train, config)`:
    - `TensorDataset` + `DataLoader` com `batch_size` configurável
    - Loss: `BCELoss` com `pos_weight` para desbalanceamento
    - Optimizer: `Adam`
    - Early stopping: parar se `val_loss` não melhorar por `patience` epochs
    - Logar métricas por epoch no MLflow: `mlflow.log_metric`
    - Retornar modelo treinado e histórico de loss

### 2.4 Avaliação
- [x] Criar `src/evaluate.py`:
  - Função `evaluate(model, X_test, y_test)` → retorna dict com F1, ROC-AUC, Precision, Recall
  - Função `cost_analysis(y_true, y_pred, cost_fp, cost_fn)` → retorna custo total
- [x] Criar `notebooks/03_mlp_training.ipynb`:
  - Treinar MLP com `mlflow.set_experiment("churn-mlp")`
  - Comparar MLP vs baselines em tabela
  - Plotar curva de loss (treino vs validação)
  - Plotar matriz de confusão
  - Analisar trade-off FP vs FN com `cost_analysis`
- [x] Logar modelo final: `mlflow.pytorch.log_model(model, "model")`
- [x] Commit: `feat: MLP model with early stopping and MLflow tracking`

---

## ETAPA 3 — Engenharia e API

### 3.1 Refatoração
- [x] Garantir que todos os módulos em `src/` estão sem `print()`
- [x] Substituir todos os `print()` por `logging.getLogger(__name__).info()`
- [x] Criar `src/pipeline.py`:
  - Função `run_pipeline(data_path, config)` — orquestra preprocessamento + treino + avaliação + log MLflow
- [x] Salvar artefatos em `data/processed/`:
  - Preprocessor: `joblib.dump(preprocessor, "data/processed/preprocessor.pkl")`
  - Modelo: `torch.save(model.state_dict(), "data/processed/model.pth")`

### 3.2 API FastAPI
- [x] Criar `api/schemas.py`:
  - `CustomerFeatures(BaseModel)` com todos os campos de input e tipos Pydantic
  - `PredictionResponse(BaseModel)` com `churn_probability: float` e `churn_prediction: bool`
- [x] Criar `api/main.py`:
  - Carregar preprocessor e modelo no startup via `@app.on_event("startup")`
  - `GET /health` → retorna `{"status": "ok", "model_loaded": bool}`
  - `POST /predict` → recebe `CustomerFeatures`, retorna `PredictionResponse`
  - Middleware de latência: logar tempo de resposta em cada request
  - Logging estruturado em todas as rotas (sem `print()`)

### 3.3 Testes
- [x] Criar `tests/test_smoke.py`:
  - Instanciar `ChurnMLP` com dimensões fixas
  - Fazer forward pass com tensor aleatório
  - Assertar shape do output `== (batch_size, 1)`
- [x] Criar `tests/test_schema.py`:
  - Carregar `data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv`
  - Definir schema `pandera` com tipos e ranges esperados
  - Assertar que o dataframe passa na validação
- [x] Criar `tests/test_api.py`:
  - Usar `TestClient(app)` do FastAPI
  - Testar `GET /health` → status 200
  - Testar `POST /predict` com payload válido → status 200 e campos corretos no response

### 3.4 Makefile e Qualidade
- [x] Criar `Makefile` com targets:
  ```makefile
  lint:
      ruff check .

  test:
      pytest tests/ -v

  run:
      uvicorn api.main:app --reload

  train:
      python -m src.pipeline
  ```
- [x] Rodar `ruff check .` → corrigir todos os erros até zero
- [x] Rodar `pytest tests/ -v` → todos os 3+ testes passando
- [x] Commit: `feat: FastAPI, tests, Makefile and ruff clean`

---

## ETAPA 4 — Documentação, Deploy e Entrega

### 4.1 Model Card
- [ ] Criar `docs/model_card.md` com seções:
  - Descrição do modelo (arquitetura, dataset, data de treino)
  - Métricas finais (F1, ROC-AUC, Precision, Recall no test set)
  - Limitações (distribuição do dataset, viés por perfil demográfico)
  - Uso pretendido e uso não recomendado
  - Plano de retreino (frequência, trigger de degradação)

### 4.2 Plano de Monitoramento
- [ ] Criar `docs/monitoring_plan.md` com:
  - Métricas de modelo: F1 em produção, taxa de churn previsto vs real
  - Métricas de infraestrutura: latência p99, taxa de erro 5xx
  - Data drift: monitorar distribuição de features com PSI ou KS-test
  - Alertas: degradação de F1 > 5% → trigger retreino
  - Ferramentas sugeridas: AWS CloudWatch + Evidently AI

### 4.3 README
- [ ] Criar `README.md` com seções:
  - Descrição do problema e solução
  - Arquitetura do projeto (diagrama ou texto)
  - Setup: `pip install -e .` ou `docker build`
  - Como rodar: `make train`, `make test`, `make run`
  - Como acessar MLflow UI: `mlflow ui`
  - Links para Model Card e ML Canvas
  - Tabela de resultados finais (métricas comparativas)

### 4.4 Docker
- [ ] Criar `Dockerfile`:
  ```dockerfile
  FROM python:3.11-slim
  WORKDIR /app
  COPY pyproject.toml .
  COPY src/ src/
  COPY api/ api/
  COPY data/processed/ data/processed/
  RUN pip install -e .
  CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
  ```
- [ ] Testar build: `docker build -t churn-api .`
- [ ] Testar execução: `docker run -p 8000:8000 churn-api`
- [ ] Testar endpoint: `curl http://localhost:8000/health`

### 4.5 Deploy AWS (Bônus)
- [ ] Criar `template.yaml` (AWS SAM):
  - Empacotar container Docker no ECR
  - Lambda com image URI + API Gateway HTTP API
  - Variáveis de ambiente para paths dos artefatos
- [ ] Executar: `sam build && sam deploy --guided`
- [ ] Testar endpoint público: `curl https://<api-id>.execute-api.<region>.amazonaws.com/predict`
- [ ] Commit: `feat: AWS deploy via SAM`

### 4.6 Checklist Final e Entrega
- [ ] Rodar checklist de qualidade:
  - `ruff check .` → 0 erros
  - `pytest tests/ -v` → todos passando
  - `docker build -t churn-api .` → sem erros
  - `make train && make test && make run` → pipeline roda do zero
  - `grep -r "print(" src/ api/` → saída vazia (zero prints)
  - Nenhuma chave/secret hardcoded no código
- [ ] Verificar histórico git: `git log --oneline` → commits semânticos limpos
- [ ] Gravar vídeo 5 min (método STAR):
  - **S**ituation: problema de churn na telecom
  - **T**ask: construir pipeline ML end-to-end com MLP
  - **A**ction: EDA → baselines → MLP → API → deploy
  - **R**esult: métricas finais, demo da API ao vivo, link do repo
- [ ] Publicar repositório público no GitHub
- [ ] Commit final: `docs: finalize README and model card`

---

## Resumo de Critérios de Avaliação

| Critério                        | Peso | Itens Relacionados                          |
|---------------------------------|------|---------------------------------------------|
| PyTorch (MLP + early stopping)  | 25%  | 2.2, 2.3, 2.4                               |
| Qualidade do Código             | 20%  | 3.1, 3.4 (ruff, SOLID, sem print)           |
| Pipeline / FastAPI / Testes     | 30%  | 3.1, 3.2, 3.3, pyproject.toml, Pydantic     |
| Documentação / Vídeo            | 20%  | 4.1, 4.2, 4.3, 4.6 (STAR)                  |
| Cloud Deploy (bônus)            | 5%   | 4.5                                         |
