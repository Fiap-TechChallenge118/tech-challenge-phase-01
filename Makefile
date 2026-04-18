.PHONY: lint test run train

lint:
	ruff check .

test:
	pytest tests/ -v

run:
	uvicorn api.main:app --reload

train:
	python -m src.pipeline
