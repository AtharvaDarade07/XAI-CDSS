# Data Card — PIMA Indians Diabetes Dataset

## 1. Dataset Overview

**Dataset:** PIMA Indians Diabetes Database

**Purpose:**  
Used for developing and evaluating a machine-learning based diabetes risk prediction component for the Explainable AI Personalized Clinical Decision Support System (XAI-Personalized CDSS).

**Records:** 768

**Input Features:** 8

**Target:** Outcome

---

## 2. Features

| Feature | Description |
|---|---|
| Pregnancies | Number of pregnancies |
| Glucose | Plasma glucose concentration |
| BloodPressure | Diastolic blood pressure |
| SkinThickness | Triceps skin fold thickness |
| Insulin | 2-hour serum insulin |
| BMI | Body mass index |
| DiabetesPedigreeFunction | Diabetes pedigree function |
| Age | Age in years |
| Outcome | Diabetes outcome: 0 = No, 1 = Yes |

---

## 3. Data Types

- Pregnancies: Integer
- Glucose: Integer
- BloodPressure: Integer
- SkinThickness: Integer
- Insulin: Integer
- BMI: Float
- DiabetesPedigreeFunction: Float
- Age: Integer
- Outcome: Integer

---

## 4. Dataset Audit

### Missing Values

No explicit missing/null values were found in the raw dataset.

### Duplicate Records

No duplicate rows were detected.

### Class Distribution

| Outcome | Records | Percentage |
|---|---:|---:|
| 0 — No Diabetes | 500 | 65.1% |
| 1 — Diabetes | 268 | 34.9% |

### Zero-Coded Values

Several medical measurements contain zero values that may represent missing or invalid measurements rather than genuine physiological measurements.

| Feature | Zero Values | Percentage |
|---|---:|---:|
| Pregnancies | 111 | 14.45% |
| Glucose | 5 | 0.65% |
| BloodPressure | 35 | 4.56% |
| SkinThickness | 227 | 29.56% |
| Insulin | 374 | 48.70% |
| BMI | 11 | 1.43% |
| DiabetesPedigreeFunction | 0 | 0% |
| Age | 0 | 0% |

**Important:** Zero is a valid value for Pregnancies and therefore will not be treated as missing during preprocessing.

For applicable medical measurement features, zero-coded values will be handled during the ML preprocessing pipeline.

---

## 5. Feature Ranges

| Feature | Minimum | Maximum |
|---|---:|---:|
| Pregnancies | 0 | 17 |
| Glucose | 0 | 199 |
| BloodPressure | 0 | 122 |
| SkinThickness | 0 | 99 |
| Insulin | 0 | 846 |
| BMI | 0 | 67.10 |
| DiabetesPedigreeFunction | 0.08 | 2.42 |
| Age | 21 | 81 |

---

## 6. Data Quality Decisions

The raw dataset will be preserved without modification.

Preprocessing will be performed only after train/test splitting.

The planned preprocessing pipeline will:

1. Split the dataset into training and test sets.
2. Identify zero-coded invalid measurements in applicable medical features.
3. Convert those values to missing values within the preprocessing pipeline.
4. Fit imputation only on the training data.
5. Apply the learned transformation to validation/test data.
6. Scale features where required by the selected ML model.

This approach is intended to avoid data leakage.

---

## 7. Known Limitations

- The dataset contains only 768 records.
- The dataset contains a moderate class imbalance.
- Several medical measurement columns contain substantial zero-coded values.
- The dataset represents a specific population and may not generalize directly to other populations.
- External validation may be limited if a suitable second dataset with compatible features cannot be identified.
- The dataset is intended for research and educational model development, not direct clinical diagnosis.

---

## 8. Intended Use

The dataset will be used to develop and evaluate a research prototype for:

**Explainable AI for Personalized Clinical Decision Support.**

The model output is intended to support interpretation and experimentation. It must not be presented as an autonomous medical diagnosis or treatment recommendation.

---

## 9. Preprocessing Status

**Status:** Audit completed.

The raw dataset has not been modified.

Detailed preprocessing and model development will be performed in Phase P1.

---

## 10. Dataset Decision

**Current decision: GO**

The PIMA dataset will be used as the primary dataset for the diabetes use case.

A second diabetes dataset with sufficiently overlapping features will be investigated for possible external validation before the P0 gate is closed.
