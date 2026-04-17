"""Pydantic schemas for request validation and response serialization."""

from pydantic import BaseModel, Field


class CustomerFeatures(BaseModel):
    """Campos de entrada espelhando as features usadas no treinamento.

    Todos os campos são obrigatórios — a API rejeita requests com campos ausentes
    ou tipos incorretos antes de chegar na lógica de inferência (validação Pydantic).
    Os nomes usam underscore por convenção Python; o alias com espaço é o nome original do dataset.
    """

    # Categóricas
    gender: str = Field(..., examples=["Male", "Female"])
    senior_citizen: str = Field(..., alias="Senior Citizen", examples=["Yes", "No"])
    partner: str = Field(..., examples=["Yes", "No"])
    dependents: str = Field(..., examples=["Yes", "No"])
    phone_service: str = Field(..., alias="Phone Service", examples=["Yes", "No"])
    multiple_lines: str = Field(..., alias="Multiple Lines", examples=["Yes", "No", "No phone service"])
    internet_service: str = Field(..., alias="Internet Service", examples=["DSL", "Fiber optic", "No"])
    online_security: str = Field(..., alias="Online Security", examples=["Yes", "No", "No internet service"])
    online_backup: str = Field(..., alias="Online Backup", examples=["Yes", "No", "No internet service"])
    device_protection: str = Field(..., alias="Device Protection", examples=["Yes", "No", "No internet service"])
    tech_support: str = Field(..., alias="Tech Support", examples=["Yes", "No", "No internet service"])
    streaming_tv: str = Field(..., alias="Streaming TV", examples=["Yes", "No", "No internet service"])
    streaming_movies: str = Field(..., alias="Streaming Movies", examples=["Yes", "No", "No internet service"])
    contract: str = Field(..., examples=["Month-to-month", "One year", "Two year"])
    paperless_billing: str = Field(..., alias="Paperless Billing", examples=["Yes", "No"])
    payment_method: str = Field(..., alias="Payment Method", examples=["Electronic check", "Mailed check"])

    # Numéricas
    tenure_months: float = Field(..., alias="Tenure Months", ge=0, examples=[12])
    monthly_charges: float = Field(..., alias="Monthly Charges", ge=0, examples=[65.5])
    total_charges: float = Field(..., alias="Total Charges", ge=0, examples=[786.0])

    model_config = {
        # populate_by_name=True permite usar tanto o nome Python (gender) quanto o alias (Gender)
        "populate_by_name": True,
    }

    def to_dataframe_row(self) -> dict:
        """Converte o schema para um dict com os nomes de coluna originais do dataset.

        O preprocessor (ColumnTransformer) foi treinado com os nomes originais do CSV,
        então o DataFrame precisa ter exatamente as mesmas colunas para o transform funcionar.
        """
        return {
            "Gender":             self.gender,
            "Senior Citizen":     self.senior_citizen,
            "Partner":            self.partner,
            "Dependents":         self.dependents,
            "Phone Service":      self.phone_service,
            "Multiple Lines":     self.multiple_lines,
            "Internet Service":   self.internet_service,
            "Online Security":    self.online_security,
            "Online Backup":      self.online_backup,
            "Device Protection":  self.device_protection,
            "Tech Support":       self.tech_support,
            "Streaming TV":       self.streaming_tv,
            "Streaming Movies":   self.streaming_movies,
            "Contract":           self.contract,
            "Paperless Billing":  self.paperless_billing,
            "Payment Method":     self.payment_method,
            "Tenure Months":      self.tenure_months,
            "Monthly Charges":    self.monthly_charges,
            "Total Charges":      self.total_charges,
        }


class PredictionResponse(BaseModel):
    """Resposta da rota /predict."""

    churn_probability: float = Field(..., description="Probabilidade de churn entre 0 e 1")
    # churn_prediction=True significa que o modelo classifica o cliente como provável churner
    churn_prediction: bool = Field(..., description="True se churn_probability >= threshold (padrão 0.5)")
