"""Safe prediction API response schemas."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class PredictionResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    patient_id: int
    patient_record_id: int
    model_version: str
    predicted_class: int = Field(ge=0, le=1)
    predicted_probability: float = Field(ge=0, le=1)
    created_at: datetime
