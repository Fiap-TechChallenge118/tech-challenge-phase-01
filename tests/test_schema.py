"""Schema test: valida que o CSV raw respeita os tipos e ranges esperados com pandera."""

import pandas as pd
import pandera as pa
from pandera import Column, DataFrameSchema

# Schema declarativo — cada coluna define seu dtype e constraints de negócio.
# Se o CSV vier corrompido ou com colunas faltando, o teste falha antes de qualquer treino.
CHURN_SCHEMA = DataFrameSchema(
    columns={
        "Tenure Months":    Column(int,   pa.Check.ge(0)),
        "Monthly Charges":  Column(float, pa.Check.ge(0), nullable=True),
        "Total Charges":    Column(float, pa.Check.ge(0), nullable=True),
        "Churn Value":      Column(int,   pa.Check.isin([0, 1])),
        "Gender":           Column(str),
        "Contract":         Column(str,   pa.Check.isin(["Month-to-month", "One year", "Two year"])),
        "Internet Service": Column(str,   pa.Check.isin(["DSL", "Fiber optic", "No"])),
    },
    coerce=True,   # tenta converter os tipos antes de validar (ex: "Total Charges" chega como object)
)


def test_raw_dataset_schema():
    """O dataset raw deve passar na validação de schema sem erros."""
    from src.preprocessing import _clean

    df = pd.read_csv("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    df = _clean(df)  # normaliza colunas e converte tipos (suporta formato compacto e estendido)

    validated = CHURN_SCHEMA.validate(df)
    assert len(validated) >= 5000, "Dataset deve ter pelo menos 5.000 registros"
