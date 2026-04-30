"""Pydantic schemas for request validation and response serialization."""

from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

# --- Tipos para campos categóricos ---
# Usando Literal garante que valores inválidos sejam rejeitados com 422 antes de chegar
# na lógica de inferência — sem precisar de if/else manual na rota.

YesNo          = Literal["Yes", "No"]
YesNoPhone     = Literal["Yes", "No", "No phone service"]
YesNoInternet  = Literal["Yes", "No", "No internet service"]
InternetType   = Literal["DSL", "Fiber optic", "No"]
ContractType   = Literal["Month-to-month", "One year", "Two year"]
PaymentType    = Literal[
    "Electronic check",
    "Mailed check",
    "Bank transfer (automatic)",
    "Credit card (automatic)",
]
GenderType     = Literal["Male", "Female"]


class CustomerFeatures(BaseModel):
    """Payload de entrada para predição de churn.

    Todos os campos são obrigatórios. Campos categóricos aceitam apenas os valores
    presentes no dataset de treinamento — valores fora do conjunto retornam 422.
    Os nomes usam underscore por convenção Python; os aliases com espaço são os nomes
    originais do dataset IBM Telco.
    """

    # Demográficas
    gender:          GenderType = Field(..., examples=["Male"])
    senior_citizen:  YesNo      = Field(..., alias="Senior Citizen", examples=["No"])
    partner:         YesNo      = Field(..., examples=["Yes"])
    dependents:      YesNo      = Field(..., examples=["No"])

    # Serviços de telefonia
    phone_service:   YesNo      = Field(..., alias="Phone Service",   examples=["Yes"])
    multiple_lines:  YesNoPhone = Field(..., alias="Multiple Lines",  examples=["No"])

    # Serviços de internet
    internet_service:   InternetType   = Field(..., alias="Internet Service",   examples=["DSL"])
    online_security:    YesNoInternet  = Field(..., alias="Online Security",    examples=["No"])
    online_backup:      YesNoInternet  = Field(..., alias="Online Backup",      examples=["No"])
    device_protection:  YesNoInternet  = Field(..., alias="Device Protection",  examples=["No"])
    tech_support:       YesNoInternet  = Field(..., alias="Tech Support",       examples=["No"])
    streaming_tv:       YesNoInternet  = Field(..., alias="Streaming TV",       examples=["No"])
    streaming_movies:   YesNoInternet  = Field(..., alias="Streaming Movies",   examples=["No"])

    # Contrato e cobrança
    contract:           ContractType   = Field(..., examples=["Month-to-month"])
    paperless_billing:  YesNo          = Field(..., alias="Paperless Billing",  examples=["Yes"])
    payment_method:     PaymentType    = Field(..., alias="Payment Method",     examples=["Electronic check"])

    # Numéricas
    tenure_months:    float = Field(..., alias="Tenure Months",    ge=0,   examples=[12])
    monthly_charges:  float = Field(..., alias="Monthly Charges",  ge=0,   examples=[65.5])
    total_charges:    float = Field(..., alias="Total Charges",    ge=0,   examples=[786.0])

    model_config = {
        "populate_by_name": True,
    }

    @field_validator("tenure_months", "monthly_charges", "total_charges", mode="before")
    @classmethod
    def _coerce_numeric(cls, v):
        """Aceita strings numéricas no payload (ex: '12' → 12.0)."""
        try:
            return float(v)
        except (TypeError, ValueError):
            raise ValueError(f"Valor numérico inválido: {v!r}")

    @model_validator(mode="after")
    def _validate_service_dependencies(self) -> "CustomerFeatures":
        """Valida consistência entre serviços contratados.

        Regras de negócio do dataset IBM Telco:
        - Se phone_service = 'No' → multiple_lines deve ser 'No phone service'
        - Se internet_service = 'No' → todos os add-ons de internet devem ser 'No internet service'
        """
        if self.phone_service == "No" and self.multiple_lines != "No phone service":
            raise ValueError(
                "Se 'Phone Service' é 'No', 'Multiple Lines' deve ser 'No phone service'."
            )
        if self.phone_service != "No" and self.multiple_lines == "No phone service":
            raise ValueError(
                "Se 'Phone Service' está ativo, 'Multiple Lines' não pode ser 'No phone service'."
            )

        internet_addons = {
            "Online Security":   self.online_security,
            "Online Backup":     self.online_backup,
            "Device Protection": self.device_protection,
            "Tech Support":      self.tech_support,
            "Streaming TV":      self.streaming_tv,
            "Streaming Movies":  self.streaming_movies,
        }

        if self.internet_service == "No":
            invalid = [k for k, v in internet_addons.items() if v != "No internet service"]
            if invalid:
                raise ValueError(
                    f"Se 'Internet Service' é 'No', os campos {invalid} "
                    "devem ser 'No internet service'."
                )
        else:
            # Com internet contratado, 'No internet service' não é válido
            invalid = [k for k, v in internet_addons.items() if v == "No internet service"]
            if invalid:
                raise ValueError(
                    f"Com 'Internet Service' ativo, os campos {invalid} "
                    "não podem ser 'No internet service'."
                )

        return self

    def to_dataframe_row(self) -> dict:
        """Converte o schema para um dict com os nomes de coluna originais do dataset."""
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
    """Resposta dos endpoints de predição."""

    churn_probability: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Score de probabilidade de churn — valor contínuo entre 0 e 1",
        examples=[0.7234],
    )
    churn_prediction: bool = Field(
        ...,
        description="Classificação binária: True se churn_probability >= threshold calibrado por custo",
        examples=[True],
    )


class HealthResponse(BaseModel):
    """Resposta do endpoint de health check."""

    status: str = Field(..., description="Sempre 'ok' quando a API está no ar", examples=["ok"])
    model_loaded: bool = Field(
        ...,
        description="True se os artefatos de modelo foram carregados com sucesso no startup",
        examples=[True],
    )
