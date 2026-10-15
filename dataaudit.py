
import pandas as pd

FILE = "data/raw/pima-indians-diabetes.csv"

columns = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
    "Outcome"
]

df = pd.read_csv(FILE, header=None, names=columns)

print("=" * 60)
print("PIMA DIABETES DATASET AUDIT")
print("=" * 60)

print(f"\nRows: {len(df)}")
print(f"Columns: {len(df.columns)}")

print("\n--- DATA TYPES ---")
print(df.dtypes)

print("\n--- MISSING VALUES ---")
print(df.isnull().sum())

print("\n--- DUPLICATE ROWS ---")
print(df.duplicated().sum())

print("\n--- ZERO VALUES ---")
feature_columns = columns[:-1]

for col in feature_columns:
    count = (df[col] == 0).sum()
    percentage = count / len(df) * 100
    print(f"{col:30} {count:4} ({percentage:.2f}%)")

print("\n--- TARGET DISTRIBUTION ---")
print(df["Outcome"].value_counts().sort_index())

print("\nTarget percentages:")
print((df["Outcome"].value_counts(normalize=True).sort_index() * 100).round(2))

print("\n--- FEATURE RANGES ---")
print(df[feature_columns].describe().T[["min", "max", "mean", "std"]].round(2))

print("\n--- COMPLETE SUMMARY ---")
print(df.describe().round(2))

print("\n" + "=" * 60)
print("AUDIT COMPLETE")
print("=" * 60)
