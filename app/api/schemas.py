"""Esquemas de entrada y salida de la API (validación con pydantic)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Job = Literal["admin.", "blue-collar", "entrepreneur", "housemaid", "management", "retired",
              "self-employed", "services", "student", "technician", "unemployed", "unknown"]
Marital = Literal["divorced", "married", "single"]
Education = Literal["primary", "secondary", "tertiary", "unknown"]
YesNo = Literal["no", "yes"]
Contact = Literal["cellular", "telephone", "unknown"]
Month = Literal["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
Poutcome = Literal["failure", "other", "success", "unknown"]

LEAKAGE_MESSAGE = ("El campo 'duration' no se admite: la duración de la llamada solo se conoce después de "
                   "contactar al cliente (fuga de información). El modelo es pre-contacto.")


class ClientIn(BaseModel):
    """Datos de un cliente potencial, tal como están ANTES de llamarlo."""

    model_config = ConfigDict(extra="forbid", json_schema_extra={"example": {
        "age": 35, "job": "management", "marital": "single", "education": "tertiary", "default": "no",
        "balance": 1500, "housing": "no", "loan": "no", "contact": "cellular", "day": 15, "month": "oct",
        "campaign": 1, "pdays": -1, "previous": 0, "poutcome": "unknown"}})

    age: int = Field(ge=18, le=100, description="Edad")
    job: Job
    marital: Marital
    education: Education
    default: YesNo = Field(description="¿Tiene préstamos impagos?")
    balance: int = Field(ge=-20_000, le=200_000, description="Saldo promedio anual")
    housing: YesNo = Field(description="¿Tiene préstamo hipotecario?")
    loan: YesNo = Field(description="¿Tiene préstamo personal?")
    contact: Contact
    day: int = Field(ge=1, le=31, description="Día del mes del contacto")
    month: Month
    campaign: int = Field(ge=1, le=100, description="Contactos en esta campaña (incluido el actual)")
    pdays: int = Field(ge=-1, le=999, description="Días desde el último contacto previo (-1 = nunca)")
    previous: int = Field(ge=0, le=300, description="Contactos previos a esta campaña")
    poutcome: Poutcome = Field(description="Resultado de la campaña anterior")

    @model_validator(mode="before")
    @classmethod
    def reject_duration(cls, data):
        if isinstance(data, dict) and "duration" in data:
            raise ValueError(LEAKAGE_MESSAGE)
        return data

    @model_validator(mode="after")
    def check_consistency(self):
        if self.pdays == -1 and self.previous > 0:
            raise ValueError("Inconsistencia: pdays = -1 (nunca contactado) pero previous > 0.")
        return self


class Factor(BaseModel):
    feature: str
    label: str
    value: float | int | str | None
    shap: float = Field(description="Contribución SHAP en log-odds")
    effect: Literal["aumenta", "reduce"]


class PredictionOut(BaseModel):
    probability: float = Field(description="Probabilidad estimada de conversión")
    contact_recommended: bool
    recommendation: Literal["llamar", "no llamar"]
    threshold: float
    decile: int = Field(ge=1, le=10, description="1 = 10 % de clientes con mayor probabilidad")
    top_factors: list[Factor]
    explanation: str


class HealthOut(BaseModel):
    status: str
    model_loaded: bool
    model_name: str | None = None
