"""Safe API response schemas for model-behavior explanations."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class ExplanationFeature(BaseModel):
    feature: str
    feature_value: float | None
    contribution: float
    direction: str
    rank: int = Field(ge=1)


class ExplanationResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    prediction_id: int
    patient_id: int
    patient_record_id: int
    method: str
    predicted_class: int = Field(ge=0, le=1)
    predicted_probability: float = Field(ge=0, le=1)
    features: list[ExplanationFeature]
    created_at: datetime
    summary: str
    output_space: str
    shap_base_value: float | None = None
    shap_contribution_sum: float | None = None
    shap_model_output: float | None = None
    shap_additivity_error: float | None = None
    lime_configuration: dict[str, int] | None = None


class ExplanationConsistencyResponse(BaseModel):
    prediction_id: int
    patient_id: int
    patient_record_id: int
    top3_overlap_count: int
    top3_overlap_percentage: float
    top5_overlap_count: int
    top5_overlap_percentage: float
    spearman_correlation: float
    spearman_p_value: float
    direction_agreement_count: int
    direction_disagreement_count: int
    direction_agreement_percentage: float
    shared_feature_count: int
    shap_top_features: list[str]
    lime_top_features: list[str]
    interpretation: str
    created_at: datetime
