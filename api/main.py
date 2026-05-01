"""FastAPI application for Churn MLP inference."""

import json
import logging
import os
import sqlite3
import time
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
import torch
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse

from api.schemas import CustomerFeatures, HealthResponse, PredictionResponse
from src.model import ChurnMLP

logger = logging.getLogger(__name__)

# Quando rodando em Lambda, os artefatos são baixados do S3 para /tmp (único diretório gravável).
# Localmente, usa data/processed/ gerado pelo src/pipeline.py.
# A variável ARTIFACTS_BUCKET é injetada pelo Terraform como env var da Lambda.
_LAMBDA = bool(os.environ.get("AWS_LAMBDA_FUNCTION_NAME"))
_LOCAL_DIR = Path("/tmp/artifacts") if _LAMBDA else Path("data/processed")
_S3_BUCKET = os.environ.get("ARTIFACTS_BUCKET", "")
_S3_PREFIX = os.environ.get("ARTIFACTS_PREFIX", "models/latest")

PREPROCESSOR_PATH = _LOCAL_DIR / "preprocessor.pkl"
MODEL_PATH        = _LOCAL_DIR / "model.pth"
THRESHOLD_PATH    = _LOCAL_DIR / "threshold.json"
SCORES_DB_PATH    = Path("data/processed/scores.db")  # não disponível em Lambda — endpoint retorna 503

# Estado global carregado no startup — evita re-load a cada request
_state: dict = {"model": None, "preprocessor": None, "threshold": 0.5}


def _download_artifacts_from_s3() -> None:
    """Baixa os artefatos do S3 para /tmp/artifacts no cold start da Lambda.

    /tmp é o único diretório gravável em Lambda.
    Os arquivos ficam em cache entre invocações do mesmo container — o download só ocorre
    no cold start, não em cada request.
    """
    import boto3

    _LOCAL_DIR.mkdir(parents=True, exist_ok=True)
    s3 = boto3.client("s3")
    for filename in ("preprocessor.pkl", "model.pth", "threshold.json"):
        dest = _LOCAL_DIR / filename
        if not dest.exists():  # cache: não re-baixa se já está em /tmp do container quente
            s3_key = f"{_S3_PREFIX}/{filename}"
            logger.info("Baixando s3://%s/%s → %s", _S3_BUCKET, s3_key, dest)
            s3.download_file(_S3_BUCKET, s3_key, str(dest))


def _load_artifacts() -> None:
    """Carrega preprocessor e modelo do disco para o estado global."""
    if not (PREPROCESSOR_PATH.exists() and MODEL_PATH.exists()):
        logger.warning("Artefatos não encontrados em %s", _LOCAL_DIR)
        return

    preprocessor = joblib.load(PREPROCESSOR_PATH)

    input_dim = preprocessor.transform(
        pd.DataFrame([{
            "Gender": "Male", "Senior Citizen": "No", "Partner": "No",
            "Dependents": "No", "Phone Service": "Yes", "Multiple Lines": "No",
            "Internet Service": "DSL", "Online Security": "No", "Online Backup": "No",
            "Device Protection": "No", "Tech Support": "No", "Streaming TV": "No",
            "Streaming Movies": "No", "Contract": "Month-to-month",
            "Paperless Billing": "Yes", "Payment Method": "Electronic check",
            "Tenure Months": 1.0, "Monthly Charges": 50.0, "Total Charges": 50.0,
        }])
    ).shape[1]

    model = ChurnMLP(input_dim=input_dim, hidden_dims=[64, 32], dropout=0.3)
    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
    model.eval()

    if THRESHOLD_PATH.exists():
        with open(THRESHOLD_PATH) as f:
            _state["threshold"] = json.load(f).get("threshold", 0.5)

    _state["model"] = model
    _state["preprocessor"] = preprocessor
    logger.info("Artefatos carregados — source=%s input_dim=%d threshold=%.4f",
                "s3" if _LAMBDA else "local", input_dim, _state["threshold"])


# Em Lambda, o carregamento acontece aqui — no escopo do módulo, durante o container init.
# Isso ocorre ANTES de qualquer request chegar, portanto não está sujeito ao timeout
# de 29s do API Gateway. Requests subsequentes encontram o modelo já em memória.
# Localmente, o lifespan abaixo faz o mesmo trabalho.
if _LAMBDA:
    _download_artifacts_from_s3()
    _load_artifacts()


# Em Lambda, o carregamento acontece aqui — no escopo do módulo, durante o container init.
# Isso ocorre ANTES de qualquer request chegar, portanto não está sujeito ao timeout
# de 29s do API Gateway. Requests subsequentes encontram o modelo já em memória.
# Localmente, o lifespan abaixo faz o mesmo trabalho.
if _LAMBDA:
    _download_artifacts_from_s3()
    _load_artifacts()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Carrega os artefatos no startup — usado apenas localmente (uvicorn).
    Em Lambda o carregamento já ocorreu no escopo do módulo acima.
    """
    if not _LAMBDA:
        _load_artifacts()

    yield

    _state["model"] = None
    _state["preprocessor"] = None


_DESCRIPTION = """
API de predição de churn para operadora de telecomunicações.

## Estratégia de inferência

O pipeline semanal (`src/batch_predict.py`) processa todos os clientes e armazena os
scores em banco de dados. O endpoint **GET /predict** consulta esse banco sem executar
o modelo, garantindo latência p99 < 200ms.

Para clientes recém-cadastrados ou uso em desenvolvimento, o endpoint
**POST /predict/online** executa a inferência diretamente.

## Output do modelo

O modelo retorna um **score de probabilidade contínuo [0, 1]**. A classificação binária
(`churn_prediction`) é aplicada via threshold calibrado por análise de custo
(FN = R$500, FP = R$50), salvo em `data/processed/threshold.json`.

## Documentação adicional

- [ML Canvas](../docs/ml_canvas.md)
- [Model Card](../docs/model_card.md)
- [Plano de Monitoramento](../docs/monitoring_plan.md)
"""

_TAGS = [
    {
        "name": "health",
        "description": "Liveness check da API e status de carregamento dos artefatos.",
    },
    {
        "name": "predict",
        "description": (
            "Endpoints de predição. **GET /predict** faz lookup no banco de scores pré-calculados. "
            "**POST /predict/online** executa inferência on-demand."
        ),
    },
]

app = FastAPI(
    title="Churn Prediction API",
    version="1.0.0",
    description=_DESCRIPTION,
    openapi_tags=_TAGS,
    contact={"name": "Tech Challenge — Fase 01"},
    lifespan=lifespan,
)


@app.middleware("http")
async def latency_middleware(request: Request, call_next):
    """Loga o tempo de resposta de cada request em milissegundos.

    Middleware é executado antes e depois de cada request, permitindo medir a latência real
    incluindo o tempo de serialização da resposta — mais preciso que medir só dentro da rota.
    """
    start = time.perf_counter()
    response = await call_next(request)
    latency_ms = (time.perf_counter() - start) * 1000
    logger.info(
        "method=%s path=%s status=%d latency_ms=%.2f",
        request.method, request.url.path, response.status_code, latency_ms,
    )
    return response


@app.get("/health", response_model=HealthResponse, tags=["health"])
def health():
    """Verifica se a API está no ar e se os artefatos foram carregados com sucesso."""
    return HealthResponse(status="ok", model_loaded=_state["model"] is not None)


@app.get(
    "/predict",
    response_model=PredictionResponse,
    tags=["predict"],
    responses={
        404: {"description": "Cliente não encontrado na base de scores do último ciclo batch."},
        503: {"description": "Base de scores não encontrada — execute src/batch_predict.py primeiro."},
    },
)
def predict_lookup(customer_id: str = Query(..., description="CustomerID do cliente (ex: 7590-VHVEG)")):
    """Consulta o score de churn já calculado pelo batch semanal para um cliente específico.

    O score é buscado no banco de dados local (scores.db) populado por src/batch_predict.py.
    O modelo **não é executado** durante esta chamada — latência garantida pela consulta em banco.
    Retorna 404 se o cliente não foi incluído no último ciclo de batch.
    """
    if not SCORES_DB_PATH.exists():
        raise HTTPException(
            status_code=503,
            detail="Base de scores não encontrada. Execute src/batch_predict.py primeiro.",
        )

    conn = sqlite3.connect(SCORES_DB_PATH)
    row = conn.execute(
        "SELECT churn_probability, churn_prediction FROM churn_scores WHERE customer_id = ?",
        (customer_id,),
    ).fetchone()
    conn.close()

    if row is None:
        raise HTTPException(status_code=404, detail=f"Cliente '{customer_id}' não encontrado na base de scores.")

    prob, pred = row
    logger.info("Score lookup — customer_id=%s churn_probability=%.4f", customer_id, prob)
    return PredictionResponse(churn_probability=prob, churn_prediction=bool(pred))


@app.post(
    "/predict/online",
    response_model=PredictionResponse,
    tags=["predict"],
    responses={
        422: {"description": "Payload inválido — campo ausente, tipo incorreto ou valor categórico fora do conjunto permitido."},  # noqa: E501
        503: {"description": "Artefatos de modelo não carregados — execute src/pipeline.py primeiro."},
    },
)
def predict_online(customer: CustomerFeatures):
    """Executa inferência on-demand para um cliente com features fornecidas no payload.

    Uso recomendado: desenvolvimento, testes e clientes recém-cadastrados que ainda não
    passaram pelo ciclo de batch semanal. Para consulta de scores da base existente, use **GET /predict**.

    O modelo retorna um score de probabilidade contínuo [0, 1]. A classificação binária é
    aplicada via threshold calibrado por análise de custo (FN = R$500, FP = R$50).
    """
    if _state["model"] is None:
        raise HTTPException(status_code=503, detail="Modelo não disponível. Execute src/pipeline.py primeiro.")

    threshold = _state["threshold"]
    row = pd.DataFrame([customer.to_dataframe_row()])
    X = _state["preprocessor"].transform(row)

    with torch.no_grad():
        prob = float(torch.sigmoid(_state["model"](torch.tensor(X, dtype=torch.float32))).squeeze())

    logger.info("Online prediction — churn_probability=%.4f churn_prediction=%s", prob, prob >= threshold)
    return PredictionResponse(churn_probability=round(prob, 4), churn_prediction=prob >= threshold)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Captura exceções não tratadas e retorna 500 com log estruturado."""
    logger.exception("Unhandled exception on %s %s", request.method, request.url.path)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})
