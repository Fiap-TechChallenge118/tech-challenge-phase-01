"""FastAPI application for Churn MLP inference."""

import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from api.schemas import CustomerFeatures, PredictionResponse
from src.model import ChurnMLP

logger = logging.getLogger(__name__)

# Paths dos artefatos gerados pelo src/pipeline.py
PREPROCESSOR_PATH = Path("data/processed/preprocessor.pkl")
MODEL_PATH        = Path("data/processed/model.pth")

# Threshold padrão — abaixar aumenta recall (detecta mais churns) às custas de mais FPs
THRESHOLD = 0.5

# Estado global carregado no startup — evita re-load a cada request
_state: dict = {"model": None, "preprocessor": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Carrega os artefatos uma única vez no startup da aplicação.

    Usar lifespan em vez de @app.on_event("startup") é a abordagem recomendada no FastAPI moderno.
    Se os arquivos não existirem (ex: pipeline ainda não rodou), a API sobe mas /predict retorna 503.
    """
    if PREPROCESSOR_PATH.exists() and MODEL_PATH.exists():
        preprocessor = joblib.load(PREPROCESSOR_PATH)

        # Recria a arquitetura do modelo e carrega os pesos salvos pelo pipeline
        # input_dim é inferido do preprocessor para não precisar hardcodar o valor
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
        # model.eval() desativa Dropout e BatchNorm — obrigatório para inferência determinística
        model.eval()

        _state["model"] = model
        _state["preprocessor"] = preprocessor
        logger.info("Artefatos carregados — input_dim=%d", input_dim)
    else:
        logger.warning("Artefatos não encontrados em %s / %s", PREPROCESSOR_PATH, MODEL_PATH)

    yield  # aplicação fica ativa aqui

    # cleanup ao desligar (não necessário para este caso, mas é boa prática)
    _state["model"] = None
    _state["preprocessor"] = None


app = FastAPI(title="Churn Prediction API", version="1.0.0", lifespan=lifespan)


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


@app.get("/health")
def health():
    """Verifica se a API está no ar e se os artefatos foram carregados com sucesso."""
    return {"status": "ok", "model_loaded": _state["model"] is not None}


@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerFeatures):
    """Recebe as features de um cliente e retorna a probabilidade e classificação de churn.

    Fluxo:
        1. Valida o payload via Pydantic (feito automaticamente pelo FastAPI antes de entrar aqui)
        2. Converte para DataFrame com os nomes de coluna do dataset original
        3. Aplica o preprocessor (imputer + scaler + OHE)
        4. Passa pelo modelo e retorna probabilidade + classificação binária
    """
    if _state["model"] is None:
        # 503 = Service Unavailable — a API está no ar mas o modelo não foi carregado
        raise HTTPException(status_code=503, detail="Modelo não disponível. Execute src/pipeline.py primeiro.")

    # Monta DataFrame com os nomes originais do dataset para o preprocessor reconhecer as colunas
    row = pd.DataFrame([customer.to_dataframe_row()])
    X = _state["preprocessor"].transform(row)

    with torch.no_grad():
        prob = float(_state["model"](torch.tensor(X, dtype=torch.float32)).squeeze())

    logger.info("Prediction — churn_probability=%.4f churn_prediction=%s", prob, prob >= THRESHOLD)
    return PredictionResponse(churn_probability=round(prob, 4), churn_prediction=prob >= THRESHOLD)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Captura exceções não tratadas e retorna 500 com log estruturado.

    Sem isso, erros internos vão aparecer no log mas o cliente recebe uma resposta vazia.
    """
    logger.exception("Unhandled exception on %s %s", request.method, request.url.path)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})
