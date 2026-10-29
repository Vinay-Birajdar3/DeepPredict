"""Pydantic schemas and request/response validation models for the DeepPredict REST API."""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field, model_validator


class HealthResponse(BaseModel):
    """Health check response schema."""
    status: str = Field(..., json_schema_extra={"example": "healthy"})
    timestamp: str = Field(..., json_schema_extra={"example": "2026-10-26T12:00:00"})
    model_loaded: bool = Field(..., json_schema_extra={"example": True})
    model_name: Optional[str] = Field(None, json_schema_extra={"example": "LSTMRegressor"})
    model_version: Optional[str] = Field(None, json_schema_extra={"example": "v1.0"})


class ModelInfoResponse(BaseModel):
    """Model metadata and performance response."""
    model_name: str
    model_type: str
    window_length: int
    input_features: List[str]
    input_dimension: int
    training_metrics: Dict[str, float]
    test_metrics: Optional[Dict[str, float]] = None
    created_at: str


class PredictionRequest(BaseModel):
    """Request schema for RUL estimation.

    Accepts either:
    - sequence: A 2D list of shape (window_length, num_features) of float readings, OR
    - readings: A list of dicts, each dict mapping sensor name (e.g. 's_2') to float reading.
    """
    engine_id: Optional[int] = Field(None, gt=0, description="Optional unit identifier")
    sequence: Optional[List[List[float]]] = Field(
        None,
        description="2D array of sensor measurements over the past window_length cycles: shape (L, F)",
    )
    readings: Optional[List[Dict[str, float]]] = Field(
        None,
        description="List of sensor dictionary readings for each cycle in sequence window",
    )
    ground_truth_rul: Optional[float] = Field(
        None,
        ge=0,
        allow_inf_nan=False,
        description="Optional actual ground truth RUL for live accuracy tracking",
    )

    @model_validator(mode="after")
    def validate_input_choice(self):
        if (self.sequence is None) == (self.readings is None):
            raise ValueError("Provide exactly one of 'sequence' or 'readings'.")
        return self


class PredictionResponse(BaseModel):
    """Response schema containing RUL estimate and health assessment."""
    engine_id: Optional[int]
    predicted_rul: float = Field(..., description="Estimated Remaining Useful Life in cycles (clipped >= 0)")
    raw_rul: float = Field(..., description="Unclipped raw model prediction")
    health_status: str = Field(..., description="Operational risk level: HEALTHY, WARNING, CRITICAL")
    model_version: str
    latency_ms: float
    validation_status: str
