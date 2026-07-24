FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml .
COPY src/ src/
COPY api/ api/

RUN pip install --no-cache-dir -e .

# Variáveis de ambiente com valores padrão — sobrescreva via docker run -e ou docker-compose
ENV MODEL_PATH=data/processed/model.pth \
    PREPROCESSOR_PATH=data/processed/preprocessor.pkl \
    THRESHOLD_PATH=data/processed/threshold.json \
    SCORES_DB_PATH=data/processed/scores.db \
    LOG_LEVEL=INFO

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
