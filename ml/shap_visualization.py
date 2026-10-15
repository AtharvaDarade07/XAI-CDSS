"""P2.2 - Visualize local SHAP contributions for the example patient."""

from pathlib import Path

import joblib
import matplotlib

# Use a non-interactive backend so the chart can be generated in scripts/CI.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

# Reuse P2.1's patient preparation and feature definitions to keep behavior
# aligned with its explanation and the trained pipeline's preprocessing.
from shap_explainer import FEATURES, MODEL_PATH, RESULTS, prepare_patient


def _positive_class_shap(explainer, transformed_patient):
    """Normalize SHAP's legacy and Explanation return formats."""
    try:
        explanation = explainer(transformed_patient)
        values = np.asarray(explanation.values)
        base_values = np.asarray(explanation.base_values)
        if values.ndim == 3:
            values = values[:, :, 1]
        values = values.reshape(-1)
        base = float(base_values.reshape(-1)[-1])
        return values, base
    except (TypeError, AttributeError, NotImplementedError):
        # Older SHAP releases expose shap_values() but not callable explainers.
        values = np.asarray(explainer.shap_values(transformed_patient))
        if values.ndim == 3:
            values = values[:, :, 1]
        base = np.asarray(explainer.expected_value)
        return values.reshape(-1), float(base.reshape(-1)[-1])


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    # This is the exact illustrative input used by P2.1; zero Insulin is
    # converted to missing by its shared prepare_patient() helper.
    patient_values = [6, 148, 72, 35, 0, 33.6, 0.627, 50]
    patient = prepare_patient(patient_values)
    pipeline = joblib.load(MODEL_PATH)

    probability = float(pipeline.predict_proba(patient)[0, 1])
    prediction = int(probability >= 0.5)

    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]
    transformed_patient = preprocessor.transform(patient)
    explainer = shap.TreeExplainer(model)
    shap_values, base_value = _positive_class_shap(explainer, transformed_patient)

    factors = pd.DataFrame({
        "feature": FEATURES,
        "value": patient.iloc[0].to_numpy(),
        "shap_value": shap_values,
    })
    factors["absolute_shap"] = factors["shap_value"].abs()
    factors["direction"] = np.where(
        factors["shap_value"] >= 0, "increases model output", "decreases model output"
    )
    factors = factors.sort_values("absolute_shap", ascending=False).reset_index(drop=True)
    factors.insert(0, "rank", np.arange(1, len(factors) + 1))

    RESULTS.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS / "top_shap_factors.csv"
    image_path = RESULTS / "sample_shap_explanation.png"
    factors[["feature", "value", "shap_value", "absolute_shap", "direction", "rank"]].to_csv(
        csv_path, index=False
    )

    # Order bars by absolute contribution while retaining a zero reference:
    # right/positive bars increase the model output; left/negative bars reduce it.
    chart = factors.sort_values("absolute_shap", ascending=True)
    colors = ["#b4473d" if value > 0 else "#3977a8" for value in chart["shap_value"]]
    fig, ax = plt.subplots(figsize=(10, 6.2))
    bars = ax.barh(chart["feature"], chart["shap_value"], color=colors, edgecolor="white")
    ax.axvline(0, color="#333333", linewidth=0.9)
    span = max(float(factors["absolute_shap"].max()), 0.1)
    ax.set_xlim(-span * 1.28, span * 1.28)
    for bar, value in zip(bars, chart["shap_value"]):
        offset = span * 0.025
        ax.text(value + (offset if value >= 0 else -offset), bar.get_y() + bar.get_height() / 2,
                f"{value:+.3f}", va="center", ha="left" if value >= 0 else "right", fontsize=9)
    ax.set_xlabel("SHAP value (model output space)")
    ax.set_title("Local SHAP Contributions for Example Patient", fontsize=14, pad=14)
    ax.text(0.5, -0.14,
            "Positive values increase model output  |  Negative values decrease model output\n"
            "Experimental risk prediction explanation; not a diagnosis or treatment recommendation.",
            transform=ax.transAxes, ha="center", va="top", fontsize=9, color="#444444")
    ax.grid(axis="x", linestyle=":", alpha=0.35)
    ax.set_axisbelow(True)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(image_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    shap_sum = float(shap_values.sum())
    model_output = base_value + shap_sum
    print("P2.2 - SHAP VISUALIZATION")
    print(f"Prediction class: {prediction}")
    print(f"Prediction probability: {probability:.4f}")
    print("\nTop 5 contributing features:")
    for row in factors.head(5).itertuples(index=False):
        print(f"{row.feature}: {row.shap_value:+.6f} ({row.direction})")

    print("\nModel-output-space validation (not probability comparison):")
    print(f"SHAP base value: {base_value:.6f}")
    print(f"Sum of SHAP values: {shap_sum:.6f}")
    print(f"Base value + SHAP sum: {model_output:.6f}")
    print("This additive sum is in the explainer's model-output space; it is not directly compared with probability.")
    print(f"\nSaved visualization: {image_path}")
    print(f"Saved top-factor CSV: {csv_path}")


if __name__ == "__main__":
    main()
