"""Pydantic request/response models for the inference API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

MAX_FLOWS_PER_REQUEST = 1000


class FlowBatch(BaseModel):
    """CICFlowMeter features keyed by name (e.g. 'Flow Duration'). Extra keys are ignored; null = missing."""

    flows: list[dict[str, float | None]] = Field(..., min_length=1, max_length=MAX_FLOWS_PER_REQUEST)

    model_config = {
        "json_schema_extra": {
            "example": {"flows": [{"Flow Duration": 1234567, "Total Fwd Packets": 12, "Flow Bytes/s": 1052.4}]}
        }
    }


class ClassProbability(BaseModel):
    cls: str = Field(..., alias="class")
    probability: float

    model_config = {"populate_by_name": True}


class FlowVerdict(BaseModel):
    risk_score: float = Field(..., ge=0, le=1, description="Combined SNN + VAE risk, 0 (benign) to 1 (certain attack)")
    anomaly_score: float = Field(..., ge=0, le=1, description="VAE score: 0 = typical benign, 1 = beyond almost all benign flows")
    attack_probability: float = Field(..., ge=0, le=1, description="1 - P(Benign) from the calibrated SNN")
    predicted_class: str
    confidence: float
    action: Literal["allow", "monitor", "block"]
    top_classes: list[ClassProbability]


class PredictResponse(BaseModel):
    results: list[FlowVerdict]
    latency_ms: float
    n_flows: int


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    model_loaded: bool
    detail: str | None = None
