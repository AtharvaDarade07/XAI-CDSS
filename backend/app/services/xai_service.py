"""SHAP and LIME adapters for the frozen P1 model and stored patient records."""

from dataclasses import dataclass
import re
from typing import Any

import numpy as np
import pandas as pd
import shap
from lime.lime_tabular import LimeTabularExplainer
from scipy.stats import spearmanr

from ml.pipeline import FEATURES, TARGET, load_data, prepare_features

from backend.app.db.models import PatientRecord
from backend.app.services.prediction_service import (
    MODEL_ARTIFACT_RELATIVE,
    MODEL_NAME,
    MODEL_VERSION,
    load_model_pipeline,
    patient_record_to_features,
)


LIME_RANDOM_STATE = 42
LIME_NUM_FEATURES = len(FEATURES)
LIME_NUM_SAMPLES = 5000
SHAP_OUTPUT_SPACE = "raw Logistic Regression output (log-odds; not probability)"


class ExplanationUnavailable(RuntimeError):
    """Raised when an explainer cannot produce a valid local explanation."""


@dataclass(frozen=True)
class FeatureContribution:
    feature: str
    feature_value: float | None
    contribution: float
    direction: str
    rank: int


@dataclass(frozen=True)
class LocalExplanation:
    method: str
    features: list[FeatureContribution]
    base_value: float | None = None
    contribution_sum: float | None = None
    model_output: float | None = None
    validation_error: float | None = None
    output_space: str | None = None


def raw_record_features(record: PatientRecord) -> pd.DataFrame:
    """Map stored values into P1 order without mutating or reinterpreting them."""
    values = {
        "Pregnancies": record.pregnancies,
        "Glucose": record.glucose,
        "BloodPressure": record.blood_pressure,
        "SkinThickness": record.skin_thickness,
        "Insulin": record.insulin,
        "BMI": record.bmi,
        "DiabetesPedigreeFunction": record.diabetes_pedigree_function,
        "Age": record.age,
    }
    return pd.DataFrame([[values[name] for name in FEATURES]], columns=FEATURES)


def _contributions(method: str, record: PatientRecord, values: np.ndarray) -> list[FeatureContribution]:
    raw = raw_record_features(record).iloc[0]
    ordered = sorted(
        enumerate(np.asarray(values, dtype=float).reshape(-1)),
        key=lambda pair: (-abs(float(pair[1])), pair[0]),
    )
    return [
        FeatureContribution(
            feature=FEATURES[index],
            feature_value=None if pd.isna(raw[FEATURES[index]]) else float(raw[FEATURES[index]]),
            contribution=float(value),
            direction=("positive" if value >= 0 else "negative"),
            rank=rank,
        )
        for rank, (index, value) in enumerate(ordered, start=1)
    ]


def _positive_class_shap(explainer: Any, transformed: Any) -> tuple[np.ndarray, float]:
    """Normalize modern Explanation and legacy SHAP return values for class 1."""
    try:
        explanation = explainer(transformed)
        values = np.asarray(explanation.values)
        base = np.asarray(explanation.base_values)
    except (TypeError, AttributeError, NotImplementedError):
        values = np.asarray(explainer.shap_values(transformed))
        base = np.asarray(explainer.expected_value)
    if values.ndim == 3:
        values = values[:, :, 1]
    values = values.reshape(-1)
    if values.size != len(FEATURES):
        raise ExplanationUnavailable("SHAP returned an unexpected feature dimension")
    base = float(base.reshape(-1)[-1])
    return values, base


def generate_shap_explanation(record: PatientRecord, pipeline: Any | None = None) -> LocalExplanation:
    """Explain the frozen classifier in its raw output space and validate additivity."""
    model_pipeline = pipeline or load_model_pipeline()
    raw_prepared = patient_record_to_features(record)
    transformed = model_pipeline.named_steps["preprocessor"].transform(raw_prepared)
    model = model_pipeline.named_steps["model"]
    if hasattr(model, "coef_") and hasattr(model, "decision_function"):
        # Use deterministic training-partition reference rows and explain class-1 log-odds.
        raw_data = load_data()
        reference_x, _ = prepare_features(raw_data)
        from sklearn.model_selection import train_test_split
        reference_x, _, _, _ = train_test_split(
            reference_x, raw_data[TARGET], test_size=0.20,
            stratify=raw_data[TARGET], random_state=42,
        )
        background = model_pipeline.named_steps["preprocessor"].transform(reference_x)
        explainer = shap.LinearExplainer(model, background)
        shap_values, base_value = _positive_class_shap(explainer, transformed)
        raw_output = float(model.decision_function(transformed)[0])
    else:
        explainer = shap.TreeExplainer(model)
        shap_values, base_value = _positive_class_shap(explainer, transformed)
        try:
            dmatrix = model_pipeline.named_steps["preprocessor"].transform(raw_prepared)
            raw_output = float(model.get_booster().inplace_predict(dmatrix, predict_type="margin")[0])
        except (AttributeError, TypeError):
            import xgboost as xgb
            raw_output = float(model.get_booster().predict(xgb.DMatrix(dmatrix), output_margin=True)[0])
    contribution_sum = float(shap_values.sum())
    reconstructed = base_value + contribution_sum
    error = abs(reconstructed - raw_output)
    if not np.isclose(reconstructed, raw_output, atol=2e-4, rtol=2e-4):
        raise ExplanationUnavailable("SHAP additivity validation failed in raw model-output space")
    return LocalExplanation(
        method="SHAP",
        features=_contributions("SHAP", record, shap_values),
        base_value=base_value,
        contribution_sum=contribution_sum,
        model_output=raw_output,
        validation_error=error,
        output_space=SHAP_OUTPUT_SPACE,
    )


def normalize_lime_feature(label: str) -> str:
    """Map LIME's interval conditions back to canonical project feature names."""
    text = str(label).strip()
    for feature in sorted(FEATURES, key=len, reverse=True):
        if text == feature or text.startswith(feature + " ") or re.search(rf"\b{re.escape(feature)}\b", text):
            return feature
    raise ExplanationUnavailable(f"LIME returned an unrecognized feature: {text}")


def _lime_explainer(pipeline: Any) -> LimeTabularExplainer:
    """Build P2's deterministic LIME reference matrix from P1's data rules."""
    raw_data = load_data()
    training_features, _ = prepare_features(raw_data)
    fitted_imputer = (
        pipeline.named_steps["preprocessor"]
        .named_transformers_["numeric"]
        .named_steps["imputer"]
    )
    medians = pd.Series(fitted_imputer.statistics_, index=FEATURES)
    reference = training_features.fillna(medians)[FEATURES].to_numpy(dtype=float)
    return LimeTabularExplainer(
        training_data=reference,
        feature_names=FEATURES,
        class_names=["negative", "positive"],
        mode="classification",
        discretize_continuous=True,
        random_state=LIME_RANDOM_STATE,
    )


def _lime_predict_fn(pipeline: Any):
    """LIME supplies raw ordered rows; P1's preprocessing remains authoritative."""
    def predict_probabilities(raw_matrix: Any) -> np.ndarray:
        frame = pd.DataFrame(np.asarray(raw_matrix), columns=FEATURES)
        frame = frame.assign(**{TARGET: 0})
        prepared, _ = prepare_features(frame)
        return pipeline.predict_proba(prepared[FEATURES])
    return predict_probabilities


def generate_lime_explanation(record: PatientRecord, pipeline: Any | None = None) -> LocalExplanation:
    """Generate the P2 LIME configuration for the stored row and class 1."""
    model_pipeline = pipeline or load_model_pipeline()
    explainer = _lime_explainer(model_pipeline)
    # Match P2's prepared patient row: zero-coded measurements become NaN
    # before LIME sees the instance, while the response still reports the
    # original stored value (for example, an entered zero remains zero).
    row = patient_record_to_features(record).to_numpy(dtype=float)[0]
    explanation = explainer.explain_instance(
        data_row=row,
        predict_fn=_lime_predict_fn(model_pipeline),
        labels=(1,),
        num_features=LIME_NUM_FEATURES,
        num_samples=LIME_NUM_SAMPLES,
    )
    mapped: dict[str, float] = {}
    for label, weight in explanation.as_list(label=1):
        feature = normalize_lime_feature(label)
        mapped[feature] = mapped.get(feature, 0.0) + float(weight)
    missing = set(FEATURES) - set(mapped)
    if missing:
        raise ExplanationUnavailable(f"LIME did not return all expected features: {sorted(missing)}")
    return LocalExplanation(
        method="LIME",
        features=_contributions("LIME", record, np.asarray([mapped[name] for name in FEATURES])),
        output_space="local surrogate weight for positive-class probability",
    )


def compare_explanations(shap_result: LocalExplanation, lime_result: LocalExplanation) -> dict[str, Any]:
    """Compute P2.4's overlap, rank-correlation, and direction metrics."""
    shap_by_name = {item.feature: item for item in shap_result.features}
    lime_by_name = {item.feature: item for item in lime_result.features}
    shared = [name for name in FEATURES if name in shap_by_name and name in lime_by_name]
    if len(shared) != len(FEATURES):
        raise ExplanationUnavailable("SHAP and LIME do not share the complete project feature set")
    shap_order = sorted(shared, key=lambda name: (-abs(shap_by_name[name].contribution), FEATURES.index(name)))
    lime_order = sorted(shared, key=lambda name: (-abs(lime_by_name[name].contribution), FEATURES.index(name)))
    overlap3 = len(set(shap_order[:3]) & set(lime_order[:3]))
    overlap5 = len(set(shap_order[:5]) & set(lime_order[:5]))
    rho, p_value = spearmanr(
        [abs(shap_by_name[name].contribution) for name in shared],
        [abs(lime_by_name[name].contribution) for name in shared],
    )
    agrees = sum(
        (shap_by_name[name].contribution >= 0) == (lime_by_name[name].contribution >= 0)
        for name in shared
    )
    return {
        "top3_overlap_count": overlap3,
        "top3_overlap_percentage": overlap3 / 3 * 100,
        "top5_overlap_count": overlap5,
        "top5_overlap_percentage": overlap5 / 5 * 100,
        "spearman_correlation": float(rho),
        "spearman_p_value": float(p_value),
        "direction_agreement_count": agrees,
        "direction_disagreement_count": len(shared) - agrees,
        "direction_agreement_percentage": agrees / len(shared) * 100,
        "shared_feature_count": len(shared),
        "shap_top_features": shap_order[:3],
        "lime_top_features": lime_order[:3],
        "interpretation": (
            "This compares SHAP and LIME for this explanation instance only. The methods use different "
            "scales and disagreement does not establish that either method is wrong; this single-instance "
            "comparison does not establish global explanation stability or clinical validity."
        ),
    }


def verify_prediction_model(model_version: Any) -> None:
    """Reject predictions tied to a model version other than the frozen P5.2 artifact."""
    if (
        model_version is None
        or model_version.name != MODEL_NAME
        or model_version.version != MODEL_VERSION
        or model_version.artifact_path != MODEL_ARTIFACT_RELATIVE
    ):
        raise ExplanationUnavailable("Prediction model version is incompatible with the XAI artifact")


__all__ = [
    "ExplanationUnavailable", "FeatureContribution", "LocalExplanation",
    "LIME_RANDOM_STATE", "LIME_NUM_FEATURES", "LIME_NUM_SAMPLES",
    "generate_shap_explanation", "generate_lime_explanation",
    "compare_explanations", "verify_prediction_model",
]
