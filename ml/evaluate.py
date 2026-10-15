"""
P1.3 - Final held-out test evaluation.

The test set is used exactly once for final model evaluation.
"""

from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "pima-indians-diabetes.csv"
REGISTRY = ROOT / "ml" / "registry"
RESULTS = ROOT / "ml" / "results"

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

TARGET = "Outcome"

ZERO_AS_MISSING = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
]


def load_data():
    df = pd.read_csv(DATA_PATH, header=None)
    df.columns = FEATURES + [TARGET]

    X = df[FEATURES].copy()
    y = df[TARGET].copy()

    for column in ZERO_AS_MISSING:
        X.loc[X[column] == 0, column] = np.nan

    return X, y


def bootstrap_auc(y_true, y_prob, n_bootstrap=2000, seed=42):
    rng = np.random.default_rng(seed)
    scores = []

    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    for _ in range(n_bootstrap):
        indices = rng.integers(0, len(y_true), len(y_true))

        if len(np.unique(y_true[indices])) < 2:
            continue

        score = roc_auc_score(
            y_true[indices],
            y_prob[indices]
        )

        scores.append(score)

    scores = np.array(scores)

    lower = np.percentile(scores, 2.5)
    upper = np.percentile(scores, 97.5)

    return lower, upper


def main():

    print("=" * 70)
    print("P1.3 - FINAL TEST EVALUATION")
    print("=" * 70)

    X, y = load_data()

    # EXACT same split as P1.2
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=42,
    )

    model_files = sorted(REGISTRY.glob("*_baseline.joblib"))

    if not model_files:
        raise FileNotFoundError(
            "No trained models found in ml/registry/"
        )

    results = []

    for model_path in model_files:

        model_name = model_path.stem.replace("_baseline", "")

        print("\n" + "-" * 70)
        print(f"MODEL: {model_name}")

        model = joblib.load(model_path)

        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions)
        recall = recall_score(y_test, predictions)
        f1 = f1_score(y_test, predictions)
        auc = roc_auc_score(y_test, probabilities)

        cm = confusion_matrix(y_test, predictions)

        ci_low, ci_high = bootstrap_auc(
            y_test,
            probabilities,
        )

        print(f"Accuracy : {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall   : {recall:.4f}")
        print(f"F1       : {f1:.4f}")
        print(f"ROC-AUC  : {auc:.4f}")
        print(f"95% AUC CI: [{ci_low:.4f}, {ci_high:.4f}]")

        print("\nConfusion Matrix:")
        print(cm)

        results.append({
            "model": model_name,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "roc_auc": auc,
            "auc_ci_lower": ci_low,
            "auc_ci_upper": ci_high,
            "true_negative": int(cm[0, 0]),
            "false_positive": int(cm[0, 1]),
            "false_negative": int(cm[1, 0]),
            "true_positive": int(cm[1, 1]),
        })

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        "roc_auc",
        ascending=False
    )

    output_path = RESULTS / "final_test_results.csv"

    results_df.to_csv(
        output_path,
        index=False
    )

    print("\n" + "=" * 70)
    print("FINAL TEST COMPARISON")
    print("=" * 70)

    print(
        results_df[
            [
                "model",
                "accuracy",
                "precision",
                "recall",
                "f1",
                "roc_auc",
                "auc_ci_lower",
                "auc_ci_upper",
            ]
        ].to_string(index=False)
    )

    best_model = results_df.iloc[0]["model"]

    with open(RESULTS / "final_model.json", "w") as f:
        json.dump(
            {
                "selected_model_by_test_roc_auc": best_model,
                "selection_note": (
                    "Preliminary selection was based on cross-validation. "
                    "This file records the highest held-out test ROC-AUC "
                    "for reporting comparison only."
                )
            },
            f,
            indent=4
        )

    print("\nResults saved to:")
    print(output_path)

    print("\nP1.3 COMPLETE")


if __name__ == "__main__":
    main()
