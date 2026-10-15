"""
P1.2 - Train and compare baseline ML models.

Models:
1. Logistic Regression
2. Random Forest
3. XGBoost
4. SVC

Evaluation:
- Stratified cross-validation
- Accuracy
- Precision
- Recall
- F1
- ROC-AUC

The final test set is kept untouched until final evaluation.
"""

from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "pima-indians-diabetes.csv"
REGISTRY = ROOT / "ml" / "registry"
RESULTS = ROOT / "ml" / "results"

REGISTRY.mkdir(parents=True, exist_ok=True)
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


def load_dataset():
    df = pd.read_csv(DATA_PATH, header=None)
    df.columns = FEATURES + [TARGET]

    X = df[FEATURES].copy()
    y = df[TARGET].copy()

    for column in ZERO_AS_MISSING:
        X.loc[X[column] == 0, column] = np.nan

    return X, y


def build_preprocessor():
    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                FEATURES,
            )
        ]
    )


def build_models():
    return {
        "logistic_regression": LogisticRegression(
            max_iter=2000,
            random_state=42
        ),

        "random_forest": RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            n_jobs=-1
        ),

        "xgboost": XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1
        ),

        "svc": SVC(
            probability=True,
            random_state=42
        ),
    }


def main():

    print("=" * 70)
    print("P1.2 — MODEL TRAINING & CROSS-VALIDATION")
    print("=" * 70)

    X, y = load_dataset()

    print(f"\nDataset: {X.shape[0]} records")
    print(f"Features: {X.shape[1]}")
    print(f"Positive cases: {y.sum()}")
    print(f"Negative cases: {(y == 0).sum()}")

    # Hold out final test set.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=42,
    )

    print(f"\nTraining records: {len(X_train)}")
    print(f"Final test records: {len(X_test)}")

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
    }

    models = build_models()

    results = []

    for name, model in models.items():

        print(f"\n{'-' * 70}")
        print(f"Training: {name}")

        pipeline = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                ("model", model),
            ]
        )

        cv_results = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
        )

        row = {
            "model": name,
            "accuracy_mean": cv_results["test_accuracy"].mean(),
            "precision_mean": cv_results["test_precision"].mean(),
            "recall_mean": cv_results["test_recall"].mean(),
            "f1_mean": cv_results["test_f1"].mean(),
            "roc_auc_mean": cv_results["test_roc_auc"].mean(),

            "accuracy_std": cv_results["test_accuracy"].std(),
            "precision_std": cv_results["test_precision"].std(),
            "recall_std": cv_results["test_recall"].std(),
            "f1_std": cv_results["test_f1"].std(),
            "roc_auc_std": cv_results["test_roc_auc"].std(),
        }

        results.append(row)

        print(f"Accuracy : {row['accuracy_mean']:.4f}")
        print(f"Precision: {row['precision_mean']:.4f}")
        print(f"Recall   : {row['recall_mean']:.4f}")
        print(f"F1       : {row['f1_mean']:.4f}")
        print(f"ROC-AUC  : {row['roc_auc_mean']:.4f}")

        # Train full pipeline on training data.
        pipeline.fit(X_train, y_train)

        model_path = REGISTRY / f"{name}_baseline.joblib"
        joblib.dump(pipeline, model_path)

        print(f"Saved: {model_path}")

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        by="roc_auc_mean",
        ascending=False
    )

    results_path = RESULTS / "cross_validation_results.csv"
    results_df.to_csv(results_path, index=False)

    print("\n" + "=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)

    print(
        results_df[
            [
                "model",
                "accuracy_mean",
                "precision_mean",
                "recall_mean",
                "f1_mean",
                "roc_auc_mean",
            ]
        ].to_string(index=False)
    )

    best_model = results_df.iloc[0]["model"]

    metadata = {
        "best_model_by_cv_roc_auc": best_model,
        "random_seed": 42,
        "test_size": 0.20,
        "cv_folds": 5,
        "selection_metric": "roc_auc_mean",
    }

    with open(RESULTS / "model_selection.json", "w") as f:
        json.dump(metadata, f, indent=4)

    print("\nBest preliminary model:", best_model)
    print("\nP1.2 COMPLETE")
    print("The final test set has NOT been used yet.")


if __name__ == "__main__":
    main()
