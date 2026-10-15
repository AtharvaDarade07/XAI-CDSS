"""Privacy-minimal patient API schemas."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator


class PatientCreate(BaseModel):
    patient_code: str = Field(min_length=1, max_length=100)

    @field_validator("patient_code")
    @classmethod
    def reject_whitespace_only_code(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("patient_code must contain a non-whitespace character")
        # Preserve the submitted code exactly; validation does not normalize it.
        return value


class PatientResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    patient_code: str
    created_at: datetime
    updated_at: datetime


class PatientListResponse(BaseModel):
    items: list[PatientResponse]
    skip: int
    limit: int
    total: int
