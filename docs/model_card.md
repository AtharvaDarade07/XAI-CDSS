# Model Card — XAI Personalized Clinical Decision Support System

## Model identity

- **Selected model:** Logistic Regression
- **Model version:** `logistic-regression-selected-v2`
- **Artifact:** `ml/registry/logistic_regression_selected_v2.joblib`
- **Dataset:** Pima Indians Diabetes dataset (768 rows; target `Outcome`)
- **Random seed:** 42
- **Selection protocol:** five-fold shuffled stratified CV on the training partition only; selection primary metric was mean ROC-AUC. The held-out partition did not influence selection.
- **Preprocessing:** zero-coded Glucose, BloodPressure, SkinThickness, Insulin, and BMI treated as missing; median imputation and StandardScaler fitted inside each CV fold and then on the complete training partition. Pregnancies zero is retained.

## Model selection results

| Model | ROC-AUC mean ± SD | Accuracy mean ± SD | Precision mean ± SD | Recall mean ± SD | F1 mean ± SD |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8434 ± 0.0185 | 0.7882 ± 0.0191 | 0.7638 ± 0.0598 | 0.5748 ± 0.0227 | 0.6545 ± 0.0244 |
| SVC | 0.8333 ± 0.0228 | 0.7801 ± 0.0136 | 0.7415 ± 0.0644 | 0.5842 ± 0.0572 | 0.6485 ± 0.0187 |
| Random Forest | 0.8209 ± 0.0209 | 0.7687 ± 0.0271 | 0.6995 ± 0.0636 | 0.6028 ± 0.0258 | 0.6458 ± 0.0298 |
| XGBoost | 0.8023 ± 0.0252 | 0.7508 ± 0.0218 | 0.6663 ± 0.0591 | 0.5888 ± 0.0401 | 0.6223 ± 0.0229 |

The predefined project selection metric is threshold-independent ROC-AUC because the prototype produces probability scores. This is a research evaluation criterion, not a clinical decision threshold. Logistic Regression was selected because it had the highest training-only mean CV ROC-AUC.

## Independent held-out evaluation

The untouched 20% stratified partition was evaluated after model selection and artifact freeze:

| Accuracy | Precision | Recall | F1 | ROC-AUC (bootstrap 95% CI) | Brier score |
|---:|---:|---:|---:|---:|---:|
| 0.7078 | 0.6000 | 0.5000 | 0.5455 | 0.8130 (0.7396–0.8729) | 0.1754 |

Confusion counts: TN=82, FP=18, FN=27, TP=27. The internal held-out calibration assessment had an observed positive rate of 0.3506 and mean predicted probability of 0.3515. These measurements do not establish clinical calibration.

## Intended use and limits

This model is for research and educational development of an explainable decision-support prototype. It predicts a diabetes class/probability from the Pima dataset. It is not intended to diagnose real patients, recommend treatment, replace clinical assessment, or operate without an authorized human reviewing the output.

The dataset contains 768 records from a specific population. External, prospective, subgroup, and clinical validation have not been performed. Generalization, demographic performance, bias, and clinical utility are unknown. False positives and false negatives occur. Probabilities are not clinically calibrated risk estimates. SHAP and LIME are post-hoc explanations of model behavior, not causal evidence or medical advice; their agreement was assessed on one synthetic example only.

## Explainability and reproducibility

The backend's P5.2 selected artifact is used by prediction, SHAP, LIME, and personalized explanation workflows. SHAP uses a linear explainer in raw model-output (log-odds) space and validates additivity against the decision function. LIME uses seed 42 and 5,000 samples. See `evaluation/results/` for versioned CV, independent test, XAI, and reproducibility records.

The old artifact `xgboost_baseline.joblib` and P5.1 records are retained for historical traceability; they are not the current selected inference model after the P5.2 code is loaded.
