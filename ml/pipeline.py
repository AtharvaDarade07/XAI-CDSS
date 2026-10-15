"""
P1.1 - Data preprocessing pipeline

Important:
- Raw data is never modified.
- Zero-coded medical measurements are treated as missing.
- Pregnancies = 0 remains valid.
- Imputation and scaling are fitted only on training data.
"""

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "pima-indians-diabetes.csv"


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

# Pregnancies = 0 is valid.
ZERO_AS_MISSING = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
]


def load_data():
    """Load the untouched raw PIMA dataset."""
    df = pd.read_csv(DATA_PATH, header=None)

    df.columns = FEATURES + [TARGET]

    return df


def prepare_features(df):
    """
    Convert invalid zero-coded medical measurements to NaN.

    This is performed on a dataframe copy, so the raw CSV remains untouched.
    """
    X = df[FEATURES].copy()
    y = df[TARGET].copy()

    for column in ZERO_AS_MISSING:
        X.loc[X[column] == 0, column] = pd.NA

    return X, y


def build_preprocessor():
    """
    Build preprocessing pipeline.

    Median imputation and standardization are learned from training data
    only when used inside an sklearn Pipeline.
    """
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, FEATURES),
        ]
    )

    return preprocessor


if __name__ == "__main__":
    df = load_data()

    print("=" * 60)
    print("P1.1 PREPROCESSING PIPELINE CHECK")
    print("=" * 60)

    print(f"Dataset shape: {df.shape}")

    X, y = prepare_features(df)

    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")

    print("\nZero-coded values converted to missing:")
    print(X.isna().sum())

    print("\nTarget distribution:")
    print(y.value_counts())

    print("\nPreprocessor created successfully.")
