"""P2.5 - Turn saved local explanations into a cautious readable report."""

import json

import joblib
import numpy as np
import pandas as pd

from pipeline import FEATURES, ZERO_AS_MISSING, load_data, prepare_features
from shap_explainer import MODEL_PATH, RESULTS, prepare_patient


SHAP_PATH = RESULTS / "sample_shap_explanation.csv"
CONSISTENCY_PATH = RESULTS / "shap_lime_consistency_summary.json"
PROBABILITY_TOLERANCE = 1e-6
DISCLAIMER = (
    "This output explains the behavior of an experimental machine-learning model. "
    "It is not a medical diagnosis or treatment recommendation. Final clinical decisions "
    "require qualified professional judgment. This prototype has not established "
    "prospective clinical validity."
)


def _value_text(value):
    return "missing" if pd.isna(value) else f"{float(value):g}"


def _direction_label(shap_value):
    return "higher" if shap_value > 0 else "lower"


def main():
    for path in (MODEL_PATH, SHAP_PATH, CONSISTENCY_PATH):
        if not path.exists():
            raise FileNotFoundError(f"Required input not found: {path}")

    # Same illustrative example as P2.1-P2.4; zero Insulin is normalized by
    # the existing SHAP helper and passed to the existing trained pipeline.
    patient = prepare_patient([6, 148, 72, 35, 0, 33.6, 0.627, 50])
    pipeline = joblib.load(MODEL_PATH)
    probability = float(pipeline.predict_proba(patient)[0, 1])
    predicted_class = int(probability >= 0.5)

    # Use saved SHAP and consistency artifacts as the explanation sources.
    shap_df = pd.read_csv(SHAP_PATH)
    consistency = json.loads(CONSISTENCY_PATH.read_text(encoding="utf-8"))
    if set(shap_df["feature"]) != set(FEATURES) or len(shap_df) != len(FEATURES):
        raise ValueError("Saved SHAP feature names do not match the project feature schema.")
    if "prediction_probability" not in consistency:
        raise ValueError("P2.4 summary does not contain prediction_probability.")
    stored_probability = float(consistency["prediction_probability"])
    if not np.isclose(probability, stored_probability, atol=PROBABILITY_TOLERANCE, rtol=0):
        raise ValueError(
            f"Model probability ({probability:.10f}) does not match P2.4 "
            f"({stored_probability:.10f}) within {PROBABILITY_TOLERANCE}."
        )

    # Calculate reference statistics after the project's established zero-as-
    # missing handling. No new imputation is used for cohort summaries.
    raw_reference = load_data()
    reference, _ = prepare_features(raw_reference)
    if list(reference.columns) != FEATURES:
        raise ValueError("Reference feature order does not match the project schema.")

    patient_values = patient.iloc[0]
    contributions = shap_df.copy()
    contributions["absolute_shap"] = contributions["shap_value"].abs()
    top = contributions.sort_values("absolute_shap", ascending=False, kind="stable").head(5)

    positive = top[top["shap_value"] > 0]
    negative = top[top["shap_value"] < 0]
    positive_features = []
    negative_features = []
    reference_context = []

    for row in top.itertuples(index=False):
        feature = row.feature
        value = patient_values[feature]
        shap_value = float(row.shap_value)
        direction = _direction_label(shap_value)
        value_label = _value_text(value)
        missing_insulin = feature == "Insulin" and pd.isna(value)
        sentence = (
            f"{feature} contributed toward a {direction} model prediction "
            f"(SHAP contribution {shap_value:+.6f})."
        )
        if missing_insulin:
            sentence += " The original Insulin value is missing and was handled through the project's preprocessing pipeline."

        item = {
            "feature": feature,
            "patient_value": None if pd.isna(value) else float(value),
            "patient_value_display": value_label,
            "shap_contribution": shap_value,
            "direction": f"toward a {direction} model prediction",
            "rank_by_absolute_shap": int(top.index.get_loc(row.Index) + 1) if hasattr(row, "Index") else None,
            "explanation": sentence,
        }
        # itertuples doesn't expose the DataFrame index consistently for named
        # tuple fields, so rank is assigned explicitly below from sorted order.
        item["rank_by_absolute_shap"] = len(positive_features) + len(negative_features) + 1
        (positive_features if shap_value > 0 else negative_features).append(item)

        if pd.isna(value):
            context_sentence = (
                f"{feature}: the original patient value is missing, so it cannot be compared "
                "with the reference interquartile range; preprocessing handled the missing value."
            )
            reference_context.append({
                "feature": feature,
                "patient_value": None,
                "reference_median": float(reference[feature].median()),
                "reference_25th_percentile": float(reference[feature].quantile(0.25)),
                "reference_75th_percentile": float(reference[feature].quantile(0.75)),
                "comparison": "missing patient value; no range comparison",
                "sentence": context_sentence,
            })
        else:
            median = float(reference[feature].median())
            q25 = float(reference[feature].quantile(0.25))
            q75 = float(reference[feature].quantile(0.75))
            if value < q25:
                comparison = "below reference interquartile range"
                phrase = "below"
            elif value > q75:
                comparison = "above reference interquartile range"
                phrase = "above"
            else:
                comparison = "within reference interquartile range"
                phrase = "within"
            context_sentence = (
                f"Compared with the reference dataset, the patient's {feature} value ({value_label}) "
                f"is {phrase} the interquartile range ({q25:.3g} to {q75:.3g})."
            )
            reference_context.append({
                "feature": feature,
                "patient_value": float(value),
                "reference_median": median,
                "reference_25th_percentile": q25,
                "reference_75th_percentile": q75,
                "comparison": comparison,
                "sentence": context_sentence,
            })

    # Guard the generated language against unsupported causal/clinical claims.
    claim_text = " ".join(
        item["explanation"] for item in positive_features + negative_features
    ).lower()
    forbidden_claims = ("causes diabetes", "caused diabetes", "has diabetes", "needs treatment")
    if any(claim in claim_text for claim in forbidden_claims):
        raise ValueError("Generated explanation contains an unsupported medical claim.")

    top_positive_lines = []
    for idx, item in enumerate(positive_features, 1):
        top_positive_lines.extend([
            f"{idx}. {item['feature']} - value: {item['patient_value_display']} - "
            f"contribution: {item['shap_contribution']:+.6f}",
            f"   {item['explanation']}",
        ])
    top_positive_lines = top_positive_lines or [
        "No positive contributors among the five highest absolute SHAP features."
    ]
    top_negative_lines = []
    for idx, item in enumerate(negative_features, 1):
        top_negative_lines.extend([
            f"{idx}. {item['feature']} - value: {item['patient_value_display']} - "
            f"contribution: {item['shap_contribution']:+.6f}",
            f"   {item['explanation']}",
        ])
    top_negative_lines = top_negative_lines or [
        "No negative contributors among the five highest absolute SHAP features."
    ]
    consistency_sentence = (
        f"SHAP and LIME identified {consistency['top3_overlap_count']} of the same 3 features "
        f"among their top three contributors ({consistency['top3_overlap_percentage']:.1f}%). "
        f"Their top-five overlap was {consistency['top5_overlap_count']} of 5 "
        f"({consistency['top5_overlap_percentage']:.1f}%). Across the "
        f"{consistency['shared_feature_count']} shared features, both methods agreed on "
        f"contribution direction for {consistency['direction_agreement_count']} "
        f"({consistency['direction_agreement_percentage']:.1f}%) in this example. "
        "This single-example comparison does not establish general explanation stability."
    )

    summary_parts = [
        f"Model-estimated probability of the positive class: {probability * 100:.2f}% (predicted class {predicted_class}).",
        "Top features contributing toward a higher model prediction: " + "; ".join(
            item["feature"] for item in positive_features
        ) + ".",
        "Top features contributing toward a lower model prediction: " + "; ".join(
            item["feature"] for item in negative_features
        ) + ".",
        " ".join(item["sentence"] for item in reference_context),
        consistency_sentence,
    ]
    human_summary = "\n\n".join(summary_parts)

    report_lines = [
        "PERSONALIZED MODEL EXPLANATION",
        "=" * 34,
        "",
        "MODEL OUTPUT",
        f"Model-estimated probability of the positive class: {probability * 100:.2f}%.",
        f"Predicted class: {predicted_class}.",
        "",
        "FEATURES CONTRIBUTING TOWARD A HIGHER MODEL PREDICTION",
        *top_positive_lines,
        "",
        "FEATURES CONTRIBUTING TOWARD A LOWER MODEL PREDICTION",
        *top_negative_lines,
        "",
        "REFERENCE-DATA CONTEXT",
        *[item["sentence"] for item in reference_context],
        "These are comparisons with this dataset's interquartile ranges, not medical normal/abnormal assessments.",
        "",
        "SHAP/LIME CONSISTENCY",
        consistency_sentence,
        "",
        "RESPONSIBLE-AI NOTE",
        DISCLAIMER,
        "",
        "SHAP contributions describe model behavior and are not on the same scale as LIME weights.",
    ]
    report = "\n".join(report_lines)

    result = {
        "prediction": {
            "class": predicted_class,
            "probability": probability,
            "probability_percent": probability * 100,
        },
        "patient": {key: None if pd.isna(value) else float(value) for key, value in patient_values.items()},
        "top_positive_features": positive_features,
        "top_negative_features": negative_features,
        "reference_context": reference_context,
        "explanation_consistency": {
            "top3_overlap_count": consistency["top3_overlap_count"],
            "top3_overlap_percentage": consistency["top3_overlap_percentage"],
            "top5_overlap_count": consistency["top5_overlap_count"],
            "top5_overlap_percentage": consistency["top5_overlap_percentage"],
            "direction_agreement_count": consistency["direction_agreement_count"],
            "direction_disagreement_count": consistency["direction_disagreement_count"],
            "direction_agreement_percentage": consistency["direction_agreement_percentage"],
            "shared_feature_count": consistency["shared_feature_count"],
            "summary": consistency_sentence,
        },
        "human_readable_summary": human_summary,
        "responsible_ai_note": DISCLAIMER,
    }

    json_path = RESULTS / "personalized_explanation.json"
    text_path = RESULTS / "personalized_explanation.txt"
    json_path.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    text_path.write_text(report, encoding="utf-8")

    print("========================================")
    print("P2.5 - PERSONALIZED EXPLANATION")
    print("========================================")
    print(f"\nModel probability: {probability:.4f} ({probability * 100:.2f}%)")
    print("\nTop positive contributors:")
    print("\n".join(top_positive_lines))
    print("\nTop negative contributors:")
    print("\n".join(top_negative_lines))
    print("\nReference-data context:")
    print("\n".join(item["sentence"] for item in reference_context))
    print("\nSHAP/LIME consistency:")
    print(consistency_sentence)
    print("\nResponsible-AI note:")
    print(DISCLAIMER)
    print("\nSaved:")
    print(json_path)
    print(text_path)


if __name__ == "__main__":
    main()
