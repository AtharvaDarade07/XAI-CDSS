"""Evaluate SHAP/LIME on one synthetic case with the frozen P5.2 pipeline."""
from __future__ import annotations
import json, sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ml.train import load_dataset, build_models
from backend.app.services.prediction_service import predict_record, MODEL_VERSION_LABEL
from backend.app.services.xai_service import generate_shap_explanation, generate_lime_explanation, compare_explanations, LIME_NUM_SAMPLES, LIME_RANDOM_STATE

RESULTS = ROOT / "evaluation" / "results"
FIGURES = ROOT / "evaluation" / "figures"
ARTIFACT = ROOT / "ml" / "registry" / "logistic_regression_selected_v2.joblib"
FEATURES = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]

def main():
    pipeline = joblib.load(ARTIFACT)
    synthetic = SimpleNamespace(pregnancies=6, glucose=148, blood_pressure=72, skin_thickness=35,
        insulin=0, bmi=33.6, diabetes_pedigree_function=0.627, age=50)
    pred = predict_record(synthetic, pipeline)
    shap_a = generate_shap_explanation(synthetic, pipeline)
    lime_a = generate_lime_explanation(synthetic, pipeline)
    metrics = compare_explanations(shap_a, lime_a)
    shap_b = generate_shap_explanation(synthetic, pipeline)
    lime_b = generate_lime_explanation(synthetic, pipeline)
    reproducible = (np.allclose([f.contribution for f in shap_a.features], [f.contribution for f in shap_b.features], atol=1e-10, rtol=0)
                    and np.allclose([f.contribution for f in lime_a.features], [f.contribution for f in lime_b.features], atol=1e-10, rtol=0))
    def serialize(ex):
        return {"method": ex.method, "features": [f.__dict__ for f in ex.features], "output_space": ex.output_space,
                "base_value": ex.base_value, "contribution_sum": ex.contribution_sum,
                "model_output": ex.model_output, "additivity_error": ex.validation_error}
    (RESULTS / "xai_case_p52.json").write_text(json.dumps({"scope": "one synthetic research case; not a real patient",
      "model_version": MODEL_VERSION_LABEL, "input": {"Pregnancies":6,"Glucose":148,"BloodPressure":72,"SkinThickness":35,"Insulin":0,"BMI":33.6,"DiabetesPedigreeFunction":0.627,"Age":50},
      "prediction": {"class": pred.predicted_class, "probability": pred.predicted_probability}, "shap": serialize(shap_a),
      "lime": serialize(lime_a), "lime_configuration": {"seed": LIME_RANDOM_STATE,"num_samples": LIME_NUM_SAMPLES,"num_features":8},
      "repeated_xai_identical_within_1e-10": bool(reproducible), "personalized_explanation": {
        "summary": "For this synthetic input, the model's largest absolute feature contributions are listed below. Contributions describe model behavior only; they are not medical advice.",
        "patient_values_and_directions": [{"feature": f.feature,"value":f.feature_value,"contribution":f.contribution,"direction":f.direction} for f in shap_a.features[:5]],
        "recommendations": []}}, indent=2), encoding="utf-8")
    (RESULTS / "xai_consistency_p52.json").write_text(json.dumps({**metrics,"scope":"single synthetic case-level demonstration","model_version":MODEL_VERSION_LABEL}, indent=2), encoding="utf-8")

    # CV comparison (fold mean ± SD) figure.
    cv = pd.read_csv(RESULTS / "cv_model_selection_p52.csv")
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    ax.bar(cv.model.str.replace("_", " ").str.title(), cv.roc_auc_mean, yerr=cv.roc_auc_std, capsize=4, color=["#245b83" if m=="logistic_regression" else "#a9bbc8" for m in cv.model])
    ax.set_ylabel("Mean 5-fold ROC-AUC (training partition)"); ax.set_ylim(0.70,0.90); ax.set_title("Training-only cross-validation model comparison")
    ax.tick_params(axis="x", rotation=15); fig.tight_layout(); fig.savefig(FIGURES / "p52_cv_model_comparison.png", dpi=180); plt.close(fig)

    order_shap=[f.feature for f in shap_a.features]; order_lime=[f.feature for f in lime_a.features]
    names=list(reversed(FEATURES)); sv={f.feature:abs(f.contribution) for f in shap_a.features}; lv={f.feature:abs(f.contribution) for f in lime_a.features}
    fig,ax=plt.subplots(figsize=(8,5)); ypos=np.arange(len(names)); ax.barh(ypos-.18,[sv[n] for n in names],height=.35,label="SHAP absolute contribution"); ax.barh(ypos+.18,[lv[n] for n in names],height=.35,label="LIME absolute weight"); ax.set_yticks(ypos,names); ax.set_xlabel("Method-specific absolute contribution scale"); ax.set_title("SHAP and LIME for one synthetic case"); ax.legend(); fig.tight_layout(); fig.savefig(FIGURES / "p52_shap_lime_comparison.png",dpi=180); plt.close(fig)
    print(json.dumps({"prediction":pred.__dict__,"consistency":metrics,"xai_reproducible":reproducible,"figures":4},indent=2))

if __name__=="__main__": main()
