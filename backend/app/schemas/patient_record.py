"""PIMA-feature clinical record API schemas."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class PatientRecordCreate(BaseModel):
    # Project dataset boundaries only; these are not universal clinical limits.
    pregnancies: int | None = Field(default=None, ge=0, le=17)
    glucose: float | None = Field(default=None, ge=0, le=199, allow_inf_nan=False)
    blood_pressure: float | None = Field(default=None, ge=0, le=122, allow_inf_nan=False)
    skin_thickness: float | None = Field(default=None, ge=0, le=99, allow_inf_nan=False)
    insulin: float | None = Field(default=None, ge=0, le=846, allow_inf_nan=False)
    bmi: float | None = Field(default=None, ge=0, le=67.1, allow_inf_nan=False)
    diabetes_pedigree_function: float | None = Field(
        # The supplied dataset's observed minimum is 0.078 (about 0.08).
        default=None, ge=0.078, le=2.42, allow_inf_nan=False
    )
    age: int | None = Field(default=None, ge=21, le=81)


class PatientRecordResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    patient_id: int
    pregnancies: int | None
    glucose: float | None
    blood_pressure: float | None
    skin_thickness: float | None
    insulin: float | None
    bmi: float | None
    diabetes_pedigree_function: float | None
    age: int | None
    created_at: datetime
