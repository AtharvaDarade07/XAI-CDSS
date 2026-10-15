"""
P2.1 - SHAP explanation engine

Purpose:
Generate a local, patient-specific SHAP explanation
for the selected XGBoost diabetes-risk model.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap


ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = ROOT / "ml" / "registry" / "xgboost_baseline.joblib"
RESULTS = ROOT / "ml" / "results"

RESULTS.mkdir(parents=True, exist_ok=True)


FEATURES = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]

ZERO_AS_MISSING = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
]


def prepare_patient(values):
    """
    Prepare one patient's raw feature values.

    Zero-coded medical measurements are converted to NaN,
    matching the training preprocessing.
    """

    patient = pd.DataFrame(
        [values],
        columns=FEATURES
    )

    for column in ZERO_AS_MISSING:
        if patient.loc[0, column] == 0:
            patient.loc[0, column] = np.nan

    return patient


def main():

    print("=" * 70)
    print("P2.1 - SHAP LOCAL EXPLANATION")
    print("=" * 70)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    model_pipeline = joblib.load(MODEL_PATH)

    # ---------------------------------------------------------
    # Example patient
    # ---------------------------------------------------------
    #
    # These values are ONLY a demonstration input.
    # They are not a real patient.
    #
    patient_values = [
        6,       # Pregnancies
        148,     # Glucose
        72,      # BloodPressure
        35,      # SkinThickness
        0,       # Insulin (zero treated as missing)
        33.6,    # BMI
        0.627,   # DiabetesPedigreeFunction
        50,      # Age
    ]

    patient = prepare_patient(patient_values)

    print("\nPatient input:")
    print(patient.to_string(index=False))

    # ---------------------------------------------------------
    # Get prediction
    # ---------------------------------------------------------

    probability = model_pipeline.predict_proba(
        patient
    )[0, 1]

    prediction = int(
        probability >= 0.5
    )

    print("\nPrediction:")
    print(f"Class: {prediction}")
    print(f"Probability: {probability:.4f}")

    # ---------------------------------------------------------
    # Transform input using the SAME preprocessing
    # used by the trained pipeline.
    # ---------------------------------------------------------

    preprocessor = model_pipeline.named_steps["preprocessor"]
    model = model_pipeline.named_steps["model"]

    transformed_patient = preprocessor.transform(
        patient
    )

    # ---------------------------------------------------------
    # Create SHAP explainer
    # ---------------------------------------------------------

    explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(
        transformed_patient
    )

    # XGBoost binary classification may return
    # either a 2D array or an array with an additional dimension
    # depending on SHAP version.
    shap_values = np.asarray(shap_values)

    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, 1]

    shap_values = shap_values.reshape(-1)

    # ---------------------------------------------------------
    # Base value
    # ---------------------------------------------------------

    expected_value = explainer.expected_value

    if np.asarray(expected_value).ndim > 0:
        expected_value = np.asarray(expected_value).reshape(-1)[-1]

    expected_value = float(expected_value)

    # ---------------------------------------------------------
    # Feature contribution table
    # ---------------------------------------------------------

    explanation = pd.DataFrame({
        "feature": FEATURES,
        "value": patient.iloc[0].values,
        "shap_value": shap_values,
    })

    explanation["absolute_shap"] = (
        explanation["shap_value"].abs()
    )

    explanation["direction"] = np.where(
        explanation["shap_value"] >= 0,
        "increases risk",
        "decreases risk",
    )

    explanation = explanation.sort_values(
        "absolute_shap",
        ascending=False
    )

    print("\nSHAP feature contributions:")
    print(
        explanation[
            [
                "feature",
                "value",
                "shap_value",
                "direction",
            ]
        ].to_string(index=False)
    )

    # ---------------------------------------------------------
    # Save explanation
    # ---------------------------------------------------------

    output_path = RESULTS / "sample_shap_explanation.csv"

    explanation.to_csv(
        output_path,
        index=False
    )

    # ---------------------------------------------------------
    # SHAP consistency check
    # ---------------------------------------------------------

    shap_sum = (
        expected_value +
        shap_values.sum()
    )

    print("\nSHAP validation:")
    print(f"Expected value: {expected_value:.6f}")
    print(f"SHAP sum:       {shap_sum:.6f}")

    print(
        "\nNote: SHAP values are additive in the model's "
        "output space. The exact interpretation of the "
        "base value depends on the model/SHAP configuration."
    )

    print("\nSaved:")
    print(output_path)

    print("\nP2.1 COMPLETE")


if __name__ == "__main__":
    main()
