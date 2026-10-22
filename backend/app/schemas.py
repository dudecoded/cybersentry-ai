from typing import Any

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    features: dict[str, Any]


class PredictionResponse(BaseModel):
    prediction: int
    result: str
    attack_probability: float = Field(ge=0.0, le=1.0)
    threshold: float
    severity: str