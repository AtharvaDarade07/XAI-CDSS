"""P2.4 - Quantitative comparison of saved local SHAP and LIME results."""

import json
import re

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

from shap_explainer import FEATURES, MODEL_PATH, RESULTS, prepare_patient


SHAP_PATH = RESULTS / "sample_shap_explanation.csv"
LIME_PATH = RESULTS / "sample_lime_explanation.csv"


def normalize_feature_name(label):
    """Map SHAP names or LIME interval conditions to the canonical feature."""
    text = str(label).strip()
    for feature in sorted(FEATURES, key=len, reverse=True):
        if text == feature or text.startswith(feature + " "):
            return feature
    # Handle numeric interval terms LIME can emit, e.g. "3 < Age <= 8".
    for feature in sorted(FEATURES, key=len, reverse=True):
        if re.search(rf"\b{re.escape(feature)}\b", text):
            return feature
    raise ValueError(f"Could not normalize explanation feature: {label!r}")


def main():
    for path in (SHAP_PATH, LIME_PATH):
        if not path.exists():
            raise FileNotFoundError(f"Required saved explanation not found: {path}")

    # Read the explanation artifacts as generated; do not rerun either explainer.
    shap_df = pd.read_csv(SHAP_PATH)
    lime_df = pd.read_csv(LIME_PATH)
    shap_df["feature"] = shap_df["feature"].map(normalize_feature_name)
    lime_df["feature"] = lime_df["feature"].map(normalize_feature_name)
    if shap_df["feature"].duplicated().any() or lime_df["feature"].duplicated().any():
        raise ValueError("Duplicate canonical feature names found after normalization.")

    shap_df["shap_absolute"] = shap_df["shap_value"].abs()
    shap_df["shap_rank"] = rankdata(-shap_df["shap_absolute"], method="average").astype(float)
    shap_df["shap_direction"] = np.where(
        shap_df["shap_value"] >= 0, "positive", "negative"
    )
    lime_df["lime_absolute"] = lime_df["weight"].abs()
    lime_df["lime_rank"] = rankdata(-lime_df["lime_absolute"], method="average").astype(float)
    lime_df["lime_direction"] = np.where(
        lime_df["weight"] >= 0, "positive", "negative"
    )

    comparison = shap_df[["feature", "shap_value", "shap_absolute", "shap_rank", "shap_direction"]].merge(
        lime_df[["feature", "weight", "lime_absolute", "lime_rank", "lime_direction"]],
        on="feature", how="inner", validate="one_to_one",
    ).rename(columns={"weight": "lime_weight"})
    if len(comparison) != len(FEATURES):
        raise ValueError(f"Expected {len(FEATURES)} shared features; found {len(comparison)}.")
    comparison = comparison.sort_values("shap_rank", kind="stable").reset_index(drop=True)

    shap_order = shap_df.sort_values("shap_absolute", ascending=False, kind="stable")["feature"].tolist()
    lime_order = lime_df.sort_values("lime_absolute", ascending=False, kind="stable")["feature"].tolist()
    shap_top3, lime_top3 = shap_order[:3], lime_order[:3]
    common3 = set(shap_top3) & set(lime_top3)
    common5 = set(shap_order[:5]) & set(lime_order[:5])
    top3_percent = len(common3) / 3 * 100
    top5_percent = len(common5) / 5 * 100

    # Spearman compares ranks (descending absolute importance), never the raw
    # values from the two methods, which live on different scales.
    rho, p_value = spearmanr(comparison["shap_rank"], comparison["lime_rank"])
    agrees = comparison["shap_direction"] == comparison["lime_direction"]
    agreement_count = int(agrees.sum())
    disagreement_count = int((~agrees).sum())
    agreement_percent = agreement_count / len(comparison) * 100

    pipeline = joblib.load(MODEL_PATH)
    patient = prepare_patient([6, 148, 72, 35, 0, 33.6, 0.627, 50])
    prediction_probability = float(pipeline.predict_proba(patient)[0, 1])

    RESULTS.mkdir(parents=True, exist_ok=True)
    table_path = RESULTS / "shap_lime_consistency.csv"
    summary_path = RESULTS / "shap_lime_consistency_summary.json"
    rank_plot_path = RESULTS / "shap_lime_rank_comparison.png"
    direction_plot_path = RESULTS / "shap_lime_direction_comparison.png"
    columns = [
        "feature", "shap_value", "shap_absolute", "shap_rank", "shap_direction",
        "lime_weight", "lime_absolute", "lime_rank", "lime_direction",
    ]
    comparison[columns].to_csv(table_path, index=False)

    summary = {
        "prediction_probability": prediction_probability,
        "top3_shap": shap_top3,
        "top3_lime": lime_top3,
        "top3_overlap_count": len(common3),
        "top3_overlap_percentage": top3_percent,
        "top5_overlap_count": len(common5),
        "top5_overlap_percentage": top5_percent,
        "spearman_correlation": float(rho),
        "spearman_p_value": float(p_value),
        "direction_agreement_count": agreement_count,
        "direction_disagreement_count": disagreement_count,
        "direction_agreement_percentage": agreement_percent,
        "shared_feature_count": len(comparison),
        "interpretation": (
            "Metrics are reported without a qualitative agreement category because no defensible "
            "threshold has been established. SHAP and LIME are different methods; agreement in "
            "important features is evidence of consistency for this example, while disagreement "
            "does not by itself establish that either method is wrong. This is a single-patient "
            "analysis and cannot establish general explanation stability or clinical validity."
        ),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Compare ordinal ranks; lower rank means more important. No raw SHAP/LIME
    # magnitude is juxtaposed because their scales are method-specific.
    plot_data = comparison.sort_values("shap_rank", ascending=False)
    positions = np.arange(len(plot_data))
    width = 0.36
    fig, ax = plt.subplots(figsize=(10, 6.2))
    ax.barh(positions - width / 2, plot_data["shap_rank"], height=width,
            label="SHAP rank", color="#3977a8")
    ax.barh(positions + width / 2, plot_data["lime_rank"], height=width,
            label="LIME rank", color="#d48635")
    ax.set_yticks(positions, plot_data["feature"])
    ax.invert_yaxis()
    ax.set_xlabel("Importance rank (1 = highest; ranks compare ordering only)")
    ax.set_title("SHAP and LIME Feature Importance Ranks")
    ax.legend(frameon=False)
    ax.grid(axis="x", linestyle=":", alpha=0.35)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(rank_plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Per-feature binary direction agreement for the positive class.
    plot_data = comparison.sort_values("shap_rank", ascending=False)
    colors = ["#4c956c" if same else "#c44e52" for same in
              (plot_data["shap_direction"] == plot_data["lime_direction"])]
    fig, ax = plt.subplots(figsize=(10, 5.8))
    ax.barh(plot_data["feature"], np.ones(len(plot_data)), color=colors)
    ax.set_xlim(0, 1.35)
    ax.set_xticks([])
    ax.set_title("SHAP and LIME Direction Agreement by Feature")
    for idx, row in enumerate(plot_data.itertuples(index=False)):
        same = row.shap_direction == row.lime_direction
        ax.text(0.03, idx, f"SHAP: {row.shap_direction}; LIME: {row.lime_direction} — "
                f"{'agree' if same else 'disagree'}", va="center", color="white",
                fontsize=9, fontweight="bold")
    ax.text(0.5, -0.1, "Direction refers to contribution toward (+) or away from (−) the positive class.",
            transform=ax.transAxes, ha="center", va="top", fontsize=9, color="#444444")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(direction_plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print("========================================")
    print("P2.4 - SHAP vs LIME CONSISTENCY")
    print("========================================")
    print(f"\nPrediction probability: {prediction_probability:.4f}")
    print("\nTop 3 SHAP:")
    for feature in shap_top3:
        print(feature)
    print("\nTop 3 LIME:")
    for feature in lime_top3:
        print(feature)
    print(f"\nTop-3 overlap: {len(common3)}/3 ({top3_percent:.1f}%)")
    print(f"Top-5 overlap: {len(common5)}/5 ({top5_percent:.1f}%)")
    print(f"Spearman rank correlation: rho={rho:.4f}, p-value={p_value:.4g}")
    print(f"Direction agreement: {agreement_count} agree, {disagreement_count} disagree "
          f"({agreement_percent:.1f}%)")
    print("\nInterpretation: SHAP and LIME are different explanation methods. Agreement in important "
          "features provides evidence of explanation consistency for this example; disagreement "
          "does not automatically mean either method is wrong. This single-patient analysis cannot "
          "establish general explanation stability, clinical validity, or clinical trustworthiness.")
    print("\nSaved:")
    for path in (table_path, summary_path, rank_plot_path, direction_plot_path):
        print(path)


if __name__ == "__main__":
    main()
