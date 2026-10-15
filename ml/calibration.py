"""
P1.4 - Probability calibration analysis.

Evaluates probability quality of the trained models.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import brier_score_loss
from sklearn.calibration import calibration_curve


ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "raw" / "pima-indians-diabetes.csv"
REGISTRY = ROOT / "ml" / "registry"
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

TARGET = "Outcome"

ZERO_AS_MISSING = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
]


def load_data():

    df = pd.read_csv(
        DATA_PATH,
        header=None
    )

    df.columns = FEATURES + [TARGET]

    X = df[FEATURES].copy()
    y = df[TARGET].copy()

    for column in ZERO_AS_MISSING:
        X.loc[X[column] == 0, column] = np.nan

    return X, y


def main():

    print("=" * 70)
    print("P1.4 - PROBABILITY CALIBRATION")
    print("=" * 70)

    X, y = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=42,
    )

    model_path = REGISTRY / "xgboost_baseline.joblib"

    if not model_path.exists():
        raise FileNotFoundError(
            "xgboost_baseline.joblib not found."
        )

    model = joblib.load(model_path)

    probabilities = model.predict_proba(X_test)[:, 1]

    brier = brier_score_loss(
        y_test,
        probabilities
    )

    fraction_positive, mean_predicted = calibration_curve(
        y_test,
        probabilities,
        n_bins=10,
        strategy="uniform",
    )

    print("\nXGBoost calibration results")
    print("-" * 50)

    print(f"Brier Score: {brier:.4f}")

    print(
        f"Actual positive rate: "
        f"{y_test.mean():.4f}"
    )

    print(
        f"Mean predicted probability: "
        f"{probabilities.mean():.4f}"
    )

    print("\nCalibration bins:")

    calibration_df = pd.DataFrame({
        "mean_predicted_probability": mean_predicted,
        "actual_positive_fraction": fraction_positive,
    })

    print(calibration_df.to_string(index=False))

    calibration_df.to_csv(
        RESULTS / "xgboost_calibration.csv",
        index=False
    )

    # Calibration plot
    plt.figure(figsize=(7, 7))

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Perfect calibration"
    )

    plt.plot(
        mean_predicted,
        fraction_positive,
        marker="o",
        label="XGBoost"
    )

    plt.xlabel("Mean predicted probability")
    plt.ylabel("Observed positive fraction")
    plt.title("XGBoost Calibration Curve")
    plt.legend()
    plt.grid(True)

    plot_path = RESULTS / "xgboost_calibration.png"

    plt.savefig(
        plot_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    print("\nSaved:")
    print(RESULTS / "xgboost_calibration.csv")
    print(plot_path)

    print("\nP1.4 COMPLETE")


if __name__ == "__main__":
    main()
