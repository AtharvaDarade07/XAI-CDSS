"""Inference-only adapter for the frozen P5.2 selected sklearn pipeline."""

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

# Reuse the research pipeline's canonical order and raw-data zero handling.
# The saved pipeline itself applies its fitted median imputer and scaler.
from ml.pipeline import FEATURES, TARGET, prepare_features

from backend.app.db.models import ModelVersion, PatientRecord


PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODEL_ARTIFACT_PATH = PROJECT_ROOT / "ml" / "registry" / "logistic_regression_selected_v2.joblib"
MODEL_ARTIFACT_RELATIVE = "ml/registry/logistic_regression_selected_v2.joblib"
MODEL_NAME = "logistic-regression-selected"
MODEL_VERSION = "v2"
MODEL_VERSION_LABEL = f"{MODEL_NAME}-{MODEL_VERSION}"
MODEL_ALGORITHM = "Logistic Regression"


class ModelArtifactUnavailable(RuntimeError):
    """Raised when the frozen model artifact is missing or unusable."""


class ActiveModelUnavailable(RuntimeError):
    """Raised when no unambiguous active version matches the loaded artifact."""


@dataclass(frozen=True)
class InferenceResult:
    predicted_class: int
    predicted_probability: float


@lru_cache(maxsize=4)
def _load_pipeline(artifact_path: str) -> Any:
    """Load and validate a saved pipeline once per absolute artifact path."""
    path = Path(artifact_path)
    if not path.is_file():
        raise ModelArtifactUnavailable("The configured model artifact is missing")

    try:
        pipeline = joblib.load(path)
        steps = getattr(pipeline, "named_steps", {})
        if "preprocessor" not in steps or "model" not in steps:
            raise ValueError("Artifact is not the expected preprocessing/model pipeline")
        if not callable(getattr(pipeline, "predict", None)):
            raise ValueError("Artifact does not expose predict")
        if not callable(getattr(pipeline, "predict_proba", None)):
            raise ValueError("Artifact does not expose predict_proba")
        artifact_features = getattr(pipeline, "feature_names_in_", None)
        if artifact_features is not None and list(artifact_features) != FEATURES:
            raise ValueError("Artifact feature order differs from the P1 feature schema")
        if 1 not in list(steps["model"].classes_):
            raise ValueError("Artifact has no positive class 1")
        return pipeline
    except ModelArtifactUnavailable:
        raise
    except Exception as exc:
        raise ModelArtifactUnavailable("The configured model artifact could not be loaded") from exc


def load_model_pipeline(artifact_path: str | Path = MODEL_ARTIFACT_PATH) -> Any:
    """Return the cached, frozen selected pipeline; this function never trains."""
    return _load_pipeline(str(Path(artifact_path).resolve()))


def patient_record_to_features(record: PatientRecord) -> pd.DataFrame:
    """Map ORM fields explicitly into the exact ordered P1 feature columns."""
    raw_values = {
        "Pregnancies": record.pregnancies,
        "Glucose": record.glucose,
        "BloodPressure": record.blood_pressure,
        "SkinThickness": record.skin_thickness,
        "Insulin": record.insulin,
        "BMI": record.bmi,
        "DiabetesPedigreeFunction": record.diabetes_pedigree_function,
        "Age": record.age,
    }
    raw_frame = pd.DataFrame([[raw_values[name] for name in FEATURES]], columns=FEATURES)

    # P1's prepare_features applies the established zero-as-missing rule to
    # medical measurements while leaving Pregnancies=0 unchanged. The fitted
    # imputer and scaler remain inside the serialized sklearn pipeline.
    with_target = raw_frame.assign(**{TARGET: 0})
    prepared_features, _ = prepare_features(with_target)
    return prepared_features[FEATURES]


def predict_record(record: PatientRecord, pipeline: Any | None = None) -> InferenceResult:
    """Run inference for one stored record through the saved full pipeline."""
    model_pipeline = pipeline if pipeline is not None else load_model_pipeline()
    features = patient_record_to_features(record)
    predicted = int(model_pipeline.predict(features)[0])
    class_labels = list(model_pipeline.named_steps["model"].classes_)
    positive_index = class_labels.index(1)
    probability = float(model_pipeline.predict_proba(features)[0, positive_index])

    if predicted not in (0, 1) or not np.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError("The model returned an invalid binary prediction")
    return InferenceResult(predicted_class=predicted, predicted_probability=probability)


def _evaluation_metrics() -> dict[str, float]:
    """Read the existing held-out XGBoost metrics for version metadata."""
    results_path = PROJECT_ROOT / "ml" / "results" / "final_test_results.csv"
    if not results_path.is_file():
        return {}
    try:
        results = pd.read_csv(results_path)
        selected = results.loc[results["model"] == "xgboost"]
        if selected.empty:
            return {}
        row = selected.iloc[0]
        return {
            key: float(row[key])
            for key in (
                "accuracy", "precision", "recall", "f1", "roc_auc",
                "auc_ci_lower", "auc_ci_upper",
            )
            if key in row and pd.notna(row[key])
        }
    except Exception:
        # Metrics are descriptive registry metadata and are not required to
        # perform inference with the already validated artifact.
        return {}


def ensure_active_model_version(database: Session) -> ModelVersion:
    """Reuse the stable v1 row or initialize it once without startup seeding."""
    active_versions = database.scalars(
        select(ModelVersion).where(ModelVersion.is_active.is_(True))
    ).all()
    if len(active_versions) > 1:
        raise ActiveModelUnavailable("Multiple active model versions are configured")

    version = database.scalar(
        select(ModelVersion).where(
            ModelVersion.name == MODEL_NAME,
            ModelVersion.version == MODEL_VERSION,
        )
    )
    if active_versions:
        active = active_versions[0]
        if (
            active.id != (version.id if version is not None else None)
            or active.artifact_path != MODEL_ARTIFACT_RELATIVE
        ):
            raise ActiveModelUnavailable("No active model version matches the configured artifact")
        return active

    if version is not None:
        if version.artifact_path != MODEL_ARTIFACT_RELATIVE:
            raise ActiveModelUnavailable("Registered model artifact path does not match configuration")
        version.is_active = True
        database.flush()
        return version

    version = ModelVersion(
        name=MODEL_NAME,
        version=MODEL_VERSION,
        algorithm=MODEL_ALGORITHM,
        artifact_path=MODEL_ARTIFACT_RELATIVE,
        metrics_json=_evaluation_metrics(),
        is_active=True,
    )
    database.add(version)
    try:
        database.flush()
        return version
    except IntegrityError as exc:
        # A concurrent request may have inserted the unique name/version row.
        database.rollback()
        existing = database.scalar(
            select(ModelVersion).where(
                ModelVersion.name == MODEL_NAME,
                ModelVersion.version == MODEL_VERSION,
            )
        )
        if existing is not None and existing.artifact_path == MODEL_ARTIFACT_RELATIVE:
            existing.is_active = True
            database.flush()
            return existing
        raise ActiveModelUnavailable("Unable to initialize the active model version") from exc
