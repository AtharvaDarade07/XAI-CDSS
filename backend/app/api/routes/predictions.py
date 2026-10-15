"""Authenticated inference endpoint using an existing stored patient record."""

import logging

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from backend.app.api.dependencies import get_db, require_roles
from backend.app.db.models import Patient, PatientRecord, Prediction
from backend.app.schemas.prediction import PredictionResponse
from backend.app.services.prediction_service import (
    ActiveModelUnavailable,
    ModelArtifactUnavailable,
    ensure_active_model_version,
    load_model_pipeline,
    predict_record,
)


logger = logging.getLogger(__name__)
router = APIRouter(tags=["predictions"])


@router.post(
    "/patients/{patient_id}/records/{record_id}/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(require_roles("admin", "clinician"))],
    summary="Create a prediction from a stored patient record",
    description=(
        "Requires authentication and the admin or clinician role. Inference uses the stored "
        "patient record and the frozen research pipeline. The returned value is a raw "
        "model-estimated probability, not a clinically calibrated percentage, diagnosis, "
        "or treatment recommendation."
    ),
)
def create_prediction(
    patient_id: int,
    record_id: int,
    database: Session = Depends(get_db),
) -> PredictionResponse:
    patient = database.get(Patient, patient_id)
    if patient is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Patient not found")

    record = database.scalar(
        select(PatientRecord).where(
            PatientRecord.id == record_id,
            PatientRecord.patient_id == patient_id,
        )
    )
    if record is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Patient record not found")

    try:
        # Load first so missing/corrupt artifacts never leave registry rows behind.
        model_pipeline = load_model_pipeline()
        model_version = ensure_active_model_version(database)
        result = predict_record(record, pipeline=model_pipeline)
    except (ModelArtifactUnavailable, ActiveModelUnavailable) as exc:
        database.rollback()
        logger.exception("Prediction model is unavailable")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Prediction model is currently unavailable",
        ) from exc
    except Exception as exc:
        database.rollback()
        logger.exception("Prediction inference failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Prediction inference failed",
        ) from exc

    prediction = Prediction(
        patient_id=patient.id,
        patient_record_id=record.id,
        model_version_id=model_version.id,
        predicted_class=result.predicted_class,
        predicted_probability=result.predicted_probability,
    )
    database.add(prediction)
    try:
        database.commit()
        database.refresh(prediction)
    except IntegrityError as exc:
        database.rollback()
        logger.exception("Prediction persistence failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Prediction could not be saved",
        ) from exc

    return PredictionResponse(
        id=prediction.id,
        patient_id=prediction.patient_id,
        patient_record_id=prediction.patient_record_id,
        model_version=f"{model_version.name}-{model_version.version}",
        predicted_class=prediction.predicted_class,
        predicted_probability=prediction.predicted_probability,
        created_at=prediction.created_at,
    )
