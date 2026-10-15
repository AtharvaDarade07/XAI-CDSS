"""P2.3 - Reproducible local LIME explanation for the SHAP example patient."""

from pathlib import Path
import json

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lime.lime_tabular import LimeTabularExplainer

# Share P2.1's feature order, model location, patient normalization, and
# P1.1's raw-data loading/zero-as-missing rules.
from shap_explainer import FEATURES, MODEL_PATH, RESULTS, prepare_patient
from pipeline import load_data, prepare_features


RANDOM_STATE = 42
NUM_FEATURES = len(FEATURES)
NUM_SAMPLES = 5000


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    model_pipeline = joblib.load(MODEL_PATH)

    # The exact P2.1/P2.2 demonstration input. Zero Insulin becomes NaN via
    # the shared P2.1 helper, then the saved pipeline applies its fitted imputer.
    patient_values = [6, 148, 72, 35, 0, 33.6, 0.627, 50]
    patient = prepare_patient(patient_values)

    # LIME needs a complete numeric reference matrix for discretization and
    # perturbation. Reuse the training pipeline's fitted median statistics;
    # this does not fit or introduce a separate imputation rule.
    raw_data = load_data()
    training_features, _ = prepare_features(raw_data)
    fitted_imputer = (
        model_pipeline.named_steps["preprocessor"]
        .named_transformers_["numeric"]
        .named_steps["imputer"]
    )
    medians = pd.Series(fitted_imputer.statistics_, index=FEATURES)
    lime_training_data = training_features.fillna(medians)
    lime_training_data = lime_training_data[FEATURES].to_numpy(dtype=float)

    def predict_probabilities(raw_matrix):
        """Accept LIME's raw ordered feature matrix and use the saved pipeline."""
        frame = pd.DataFrame(np.asarray(raw_matrix), columns=FEATURES)
        # Keep zero-coded measurements consistent for perturbations that land
        # exactly on zero; NaNs are passed through to the fitted pipeline.
        for column in ("Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"):
            frame.loc[frame[column] == 0, column] = np.nan
        return model_pipeline.predict_proba(frame)

    patient_matrix = patient[FEATURES].to_numpy(dtype=float)
    probabilities = predict_probabilities(patient_matrix)
    prediction = int(np.argmax(probabilities[0]))
    positive_probability = float(probabilities[0, 1])

    # Feature order and names align with the prediction wrapper's raw matrix.
    explainer = LimeTabularExplainer(
        training_data=lime_training_data,
        feature_names=FEATURES,
        class_names=["negative", "positive"],
        mode="classification",
        discretize_continuous=True,
        random_state=RANDOM_STATE,
    )
    explanation = explainer.explain_instance(
        data_row=patient_matrix[0],
        predict_fn=predict_probabilities,
        labels=(1,),
        num_features=NUM_FEATURES,
        num_samples=NUM_SAMPLES,
    )

    # LIME terms are human-readable feature conditions; positive weights
    # support the positive class and negative weights oppose it.
    entries = explanation.as_list(label=1)
    factors = pd.DataFrame(entries, columns=["feature", "weight"])
    factors["direction"] = np.where(
        factors["weight"] >= 0,
        "supports positive prediction",
        "opposes positive prediction",
    )
    factors["absolute_weight"] = factors["weight"].abs()
    factors = factors.sort_values("absolute_weight", ascending=False).reset_index(drop=True)
    factors.insert(0, "rank", np.arange(1, len(factors) + 1))

    RESULTS.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS / "sample_lime_explanation.csv"
    image_path = RESULTS / "sample_lime_explanation.png"
    config_path = RESULTS / "lime_configuration.json"
    factors[["feature", "weight", "direction", "rank"]].to_csv(csv_path, index=False)
    configuration = {
        "random_state": RANDOM_STATE,
        "num_features": NUM_FEATURES,
        "num_samples": NUM_SAMPLES,
        "positive_class_label": 1,
        "feature_order": FEATURES,
        "model_path": str(MODEL_PATH),
        "patient_values": dict(zip(FEATURES, patient.iloc[0].tolist())),
    }
    config_path.write_text(json.dumps(configuration, indent=2), encoding="utf-8")

    # Red/blue bars distinguish support for and opposition to the positive class.
    chart = factors.sort_values("absolute_weight", ascending=True)
    colors = ["#b4473d" if weight >= 0 else "#3977a8" for weight in chart["weight"]]
    fig, ax = plt.subplots(figsize=(10, 6.2))
    bars = ax.barh(chart["feature"], chart["weight"], color=colors, edgecolor="white")
    ax.axvline(0, color="#333333", linewidth=0.9)
    extent = max(float(factors["absolute_weight"].max()), 0.01)
    ax.set_xlim(-extent * 1.3, extent * 1.3)
    for bar, weight in zip(bars, chart["weight"]):
        pad = extent * 0.025
        ax.text(weight + (pad if weight >= 0 else -pad), bar.get_y() + bar.get_height() / 2,
                f"{weight:+.3f}", va="center", ha="left" if weight >= 0 else "right", fontsize=9)
    ax.set_xlabel("LIME feature weight for positive class")
    ax.set_title("Local LIME Explanation for Example Patient", fontsize=14, pad=14)
    ax.text(0.5, -0.14,
            "Positive weights support the positive class  |  Negative weights oppose it\n"
            "Experimental risk prediction explanation; not a diagnosis or treatment recommendation.",
            transform=ax.transAxes, ha="center", va="top", fontsize=9, color="#444444")
    ax.grid(axis="x", linestyle=":", alpha=0.35)
    ax.set_axisbelow(True)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(image_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print("P2.3 - LIME LOCAL EXPLANATION")
    print(f"Predicted class: {prediction}")
    print(f"Predicted probability (positive class): {positive_probability:.4f}")
    print(f"LIME random seed: {RANDOM_STATE}")
    print(f"Configuration: num_features={NUM_FEATURES}, num_samples={NUM_SAMPLES}")
    print("\nTop 5 LIME factors:")
    for row in factors.head(5).itertuples(index=False):
        print(f"{row.feature}: {row.weight:+.6f} ({row.direction})")
    print("\nSaved:")
    print(csv_path)
    print(image_path)
    print(config_path)


if __name__ == "__main__":
    main()
