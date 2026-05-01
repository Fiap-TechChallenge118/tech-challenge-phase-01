.PHONY: lint test run train batch

lint:
	ruff check .

test:
	pytest tests/

run:
	uvicorn api.main:app --reload

train:
	python -m src.pipeline

batch:
	python -m src.batch_predict
