"""Authenticated post-hoc explanations for saved model predictions."""

from datetime import datetime, timezone
import logging

from fastapi import APIRouter, Depends, HTTPException, status
import numpy as np
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from backend.app.api.dependencies import get_db, require_roles
from backend.app.db.models import Explanation, Patient, PatientRecord, Prediction
from backend.app.schemas.explanation import (
    ExplanationConsistencyResponse,
    ExplanationFeature,
    ExplanationResponse,
)
from backend.app.services.prediction_service import (
    ModelArtifactUnavailable,
    load_model_pipeline,
    predict_record,
)
from backend.app.services.xai_service import (
    LIME_NUM_FEATURES,
    LIME_NUM_SAMPLES,
    LIME_RANDOM_STATE,
    ExplanationUnavailable,
    LocalExplanation,
    compare_explanations,
    generate_lime_explanation,
    generate_shap_explanation,
    verify_prediction_model,
)


logger = logging.getLogger(__name__)
router = APIRouter(tags=["explanations"])
RESPONSIBLE_DESCRIPTION = (
    "Requires authentication and the clinician or admin role. SHAP/LIME are post-hoc methods "
    "that describe feature contributions to this model prediction; they are not causal medical "
    "conclusions, diagnoses, or prescriptions. Clinical validity has not been established. "
    "Consistency metrics apply only to this explanation instance and do not establish global stability."
)


def _get_context(patient_id: int, record_id: int, prediction_id: int, database: Session):
    """Enforce the complete patient → record → prediction ownership chain."""
    patient = database.get(Patient, patient_id)
    if patient is None:
        raise HTTPException(status_code=404, detail="Patient not found")
    record = database.scalar(select(PatientRecord).where(
        PatientRecord.id == record_id,
        PatientRecord.patient_id == patient_id,
    ))
    if record is None:
        raise HTTPException(status_code=404, detail="Patient record not found")
    prediction = database.scalar(select(Prediction).where(
        Prediction.id == prediction_id,
        Prediction.patient_id == patient_id,
        Prediction.patient_record_id == record_id,
    ))
    if prediction is None:
        raise HTTPException(status_code=404, detail="Prediction not found")
    try:
        verify_prediction_model(prediction.model_version)
    except ExplanationUnavailable as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Explanation model is incompatible with this prediction",
        ) from exc
    return patient, record, prediction


def _load_pipeline():
    try:
        return load_model_pipeline()
    except ModelArtifactUnavailable as exc:
        logger.exception("XAI model artifact is unavailable")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Explanation model is currently unavailable",
        ) from exc


def _verified_pipeline(record: PatientRecord, prediction: Prediction):
    """Ensure the saved prediction still matches this record and frozen model."""
    pipeline = _load_pipeline()
    try:
        current = predict_record(record, pipeline=pipeline)
    except Exception as exc:
        logger.exception("Unable to verify the saved prediction against its record")
        raise HTTPException(status_code=500, detail="Prediction consistency validation failed") from exc
    if (
        current.predicted_class != prediction.predicted_class
        or not np.isclose(current.predicted_probability, prediction.predicted_probability, atol=1e-8, rtol=0)
    ):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Stored prediction does not match the current patient record and model",
        )
    return pipeline


def _generate(method: str, record: PatientRecord, pipeline) -> LocalExplanation:
    try:
        if method == "SHAP":
            return generate_shap_explanation(record, pipeline=pipeline)
        return generate_lime_explanation(record, pipeline=pipeline)
    except ExplanationUnavailable as exc:
        logger.exception("XAI explanation validation failed")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Explanation could not be generated for this model",
        ) from exc
    except Exception as exc:
        logger.exception("XAI explanation computation failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Explanation computation failed",
        ) from exc


def _summary(explanation: LocalExplanation) -> str:
    positive = [item.feature for item in explanation.features if item.contribution > 0]
    negative = [item.feature for item in explanation.features if item.contribution < 0]
    return (
        "Features contributing positively toward the model's class-1 output: "
        + (", ".join(positive[:3]) if positive else "none")
        + ". Features contributing negatively: "
        + (", ".join(negative[:3]) if negative else "none")
        + ". These contributions describe model behavior only."
    )


def _persist(database: Session, prediction: Prediction, explanation: LocalExplanation) -> datetime:
    created_at = datetime.now(timezone.utc)
    database.add_all([
        Explanation(
            prediction_id=prediction.id,
            method=explanation.method,
            feature=item.feature,
            feature_value=item.feature_value,
            contribution=item.contribution,
            direction=item.direction,
            rank=item.rank,
            created_at=created_at,
        )
        for item in explanation.features
    ])
    try:
        database.commit()
    except SQLAlchemyError as exc:
        database.rollback()
        logger.exception("Explanation persistence failed")
        raise HTTPException(status_code=500, detail="Explanation could not be saved") from exc
    return created_at


def _response(prediction: Prediction, explanation: LocalExplanation, created_at: datetime) -> ExplanationResponse:
    return ExplanationResponse(
        prediction_id=prediction.id,
        patient_id=prediction.patient_id,
        patient_record_id=prediction.patient_record_id,
        method=explanation.method,
        predicted_class=prediction.predicted_class,
        predicted_probability=prediction.predicted_probability,
        features=[ExplanationFeature(**item.__dict__) for item in explanation.features],
        created_at=created_at,
        summary=_summary(explanation),
        output_space=explanation.output_space or "method-specific contribution scale",
        shap_base_value=explanation.base_value,
        shap_contribution_sum=explanation.contribution_sum,
        shap_model_output=explanation.model_output,
        shap_additivity_error=explanation.validation_error,
        lime_configuration=(
            {"random_state": LIME_RANDOM_STATE, "num_features": LIME_NUM_FEATURES,
             "num_samples": LIME_NUM_SAMPLES}
            if explanation.method == "LIME" else None
        ),
    )


@router.post(
    "/patients/{patient_id}/records/{record_id}/predictions/{prediction_id}/explanation/shap",
    response_model=ExplanationResponse,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(require_roles("admin", "clinician"))],
    summary="Generate a SHAP explanation for a saved prediction",
    description=RESPONSIBLE_DESCRIPTION + " Each request saves a new normalized set of feature contributions.",
)
def create_shap_explanation(
    patient_id: int, record_id: int, prediction_id: int,
    database: Session = Depends(get_db),
) -> ExplanationResponse:
    _, record, prediction = _get_context(patient_id, record_id, prediction_id, database)
    explanation = _generate("SHAP", record, _verified_pipeline(record, prediction))
    return _response(prediction, explanation, _persist(database, prediction, explanation))


@router.post(
    "/patients/{patient_id}/records/{record_id}/predictions/{prediction_id}/explanation/lime",
    response_model=ExplanationResponse,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(require_roles("admin", "clinician"))],
    summary="Generate a LIME explanation for a saved prediction",
    description=RESPONSIBLE_DESCRIPTION + " LIME uses the P2 deterministic configuration. Each request saves a new set.",
)
def create_lime_explanation(
    patient_id: int, record_id: int, prediction_id: int,
    database: Session = Depends(get_db),
) -> ExplanationResponse:
    _, record, prediction = _get_context(patient_id, record_id, prediction_id, database)
    explanation = _generate("LIME", record, _verified_pipeline(record, prediction))
    return _response(prediction, explanation, _persist(database, prediction, explanation))


@router.get(
    "/patients/{patient_id}/records/{record_id}/predictions/{prediction_id}/explanation/consistency",
    response_model=ExplanationConsistencyResponse,
    dependencies=[Depends(require_roles("admin", "clinician"))],
    summary="Compare SHAP and LIME for a saved prediction",
    description=(RESPONSIBLE_DESCRIPTION + " Metrics are an instance-level comparison and are not a stability or clinical validity claim."),
)
def explanation_consistency(
    patient_id: int, record_id: int, prediction_id: int,
    database: Session = Depends(get_db),
) -> ExplanationConsistencyResponse:
    _, record, prediction = _get_context(patient_id, record_id, prediction_id, database)
    pipeline = _verified_pipeline(record, prediction)
    shap_result = _generate("SHAP", record, pipeline)
    lime_result = _generate("LIME", record, pipeline)
    metrics = compare_explanations(shap_result, lime_result)
    return ExplanationConsistencyResponse(
        prediction_id=prediction.id,
        patient_id=prediction.patient_id,
        patient_record_id=prediction.patient_record_id,
        created_at=datetime.now(timezone.utc),
        **metrics,
    )
