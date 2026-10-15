# P5.1 System-Level Testing & Research Evaluation

**Evaluation date:** 2026-10-15  
**Overall result:** PARTIAL — the integrated technical workflow passed the checks performed, with important model-selection and validation limits described below.

## 1. Evaluation Objective

This evaluation measured reproducibility and technical behavior of the existing research prototype: held-out model metrics, five-fold cross-validation, probability calibration, local SHAP/LIME behavior, the current case-level consistency analysis, authentication/RBAC, data validation, frontend regression, and a live synthetic workflow. It did not assess clinical effectiveness or patient outcomes.

## 2. Experimental Setup

The evaluation reused the saved baseline pipelines under Python 3.11.9 and the existing FastAPI/SQLAlchemy and React/Vite implementation. It did not change model architecture, hyperparameters, database schema, API contracts, or application workflow. The P5.1 runner is [run_p51_evaluation.py](../run_p51_evaluation.py); generated tabular and JSON results are in `evaluation/results/` and figures in `evaluation/figures/`.

The existing artifacts identify `xgboost-baseline-v1` as the deployed model. The project contains no `.git` metadata or SRS file, so branch, commit, and tracked-file status were unavailable. A model card, data card, README, and P0 dataset decision were present and reviewed.

## 3. Dataset

The raw Pima Indians Diabetes CSV contains 768 rows, eight input features, and the `Outcome` target. The SHA-256 fingerprint is `6bfe5d0f379d17a0e0819b996407e3c09bf80febd4287f2ed212190dfff154af`. The locked stratified split produced 614 training rows and 154 held-out rows; the held-out positive fraction was 0.35065.

The dataset is small and population-specific. Its zero-coded measurements and class imbalance affect both learning and interpretation. It is not representative evidence for all demographic groups or care settings.

## 4. ML Evaluation Protocol

The existing protocol was retained: seed 42; stratified 80/20 split with `random_state=42`; five-fold shuffled stratified CV (`random_state=42`) on the training partition; feature order `Pregnancies`, `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI`, `DiabetesPedigreeFunction`, `Age`; and `Outcome` as target.

Zeroes in Glucose, BloodPressure, SkinThickness, Insulin, and BMI are treated as missing; zero pregnancies remain valid. The zero-to-missing mapping is deterministic. Median imputation and StandardScaler are fitted inside the saved training pipelines, avoiding learned preprocessing on the full dataset. Existing model definitions and hyperparameters in `ml/train.py` were used unchanged.

Held-out metrics were recomputed by loading the four existing trained pipelines. Five-fold CV was also rerun on the 614-row training portion using the existing estimators and preprocessing pipeline. The rerun results are in [cross_validation_recomputed.csv](../results/cross_validation_recomputed.csv).

## 5. Model Performance

Recomputed held-out metrics matched `ml/results/final_test_results.csv` to within `1e-9` for all four models.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.7078 | 0.6000 | 0.5000 | 0.5455 | 0.8130 |
| Random Forest | 0.7403 | 0.6591 | 0.5370 | 0.5918 | 0.8160 |
| SVC | 0.7403 | 0.6522 | 0.5556 | 0.6000 | 0.7964 |
| XGBoost | 0.7532 | 0.6429 | 0.6667 | 0.6545 | 0.8293 |

The long-form reproducible metric file is [model_comparison.csv](../results/model_comparison.csv). The ROC comparison is [model_roc_comparison.png](../figures/model_roc_comparison.png).

The current CV means rank Logistic Regression first by ROC-AUC (0.8434 ± 0.0185), followed by SVC (0.8333 ± 0.0228), Random Forest (0.8209 ± 0.0209), then XGBoost (0.8023 ± 0.0252). This differs from the existing CV result for XGBoost (0.8033 mean ROC-AUC; 0.7573 mean accuracy versus 0.7508 in this rerun). Other model means match. The currently installed XGBoost package is 3.2.0, and loading the serialized XGBoost artifact emits an older-serialization compatibility warning. Package/version drift is plausible, but the cause of the small CV difference was not isolated. The existing artifact was not retrained or silently replaced.

There is also a model-selection caveat: `model_selection.json` identifies Logistic Regression as the CV ROC-AUC winner, while `final_model.json` records XGBoost as the highest held-out-test ROC-AUC model. Using the held-out comparison to name/select the final model means those test metrics are not an independent, unbiased final estimate for the selected model. The displayed metrics remain useful as a comparison, but should be treated as exploratory pending a fresh independent evaluation.

## 6. Calibration

For XGBoost on the locked 154-row test partition:

- Brier score: **0.16861**
- Actual positive rate: **0.35065**
- Mean predicted probability: **0.33525**
- Calibration bins: **10 uniform-width bins**; empty bins are omitted from the plotted curve.

The bin-level values are in [calibration.json](../results/calibration.json); the plot is [calibration_curve.png](../figures/calibration_curve.png). These are probability-error measurements on one internal split. They do not establish clinical calibration or justify clinical risk bands.

## 7. Reproducibility

The evaluation recorded the dataset and model-artifact hashes, Python/package versions, feature order, missing-value rules, split configuration, model version, and LIME settings in [environment.json](../results/environment.json) and [reproducibility.json](../results/reproducibility.json).

For the fixed synthetic input, two predictions were identical (class 1; probability 0.80789435). Two SHAP runs matched within `1e-10`, and two LIME runs matched within `1e-10`. The reproducibility check is PASS for this environment. The cross-validation means differ slightly from the pre-existing XGBoost CV artifact, so reproducibility across package versions is not established. The saved XGBoost artifact emits a serialization compatibility warning when loaded.

## 8. SHAP Evaluation

The existing backend SHAP service returned all eight features in the expected order. Its output is the **raw tree-model margin**, not a probability change. For the synthetic case, the baseline was -0.60521346, the signed contribution sum was 2.0416000, and the raw model margin was 1.43638599. The absolute additivity error was `5.36e-7`, satisfying the service’s additivity check.

The largest absolute SHAP contributions were Age (+0.7547), Glucose (+0.6115), and Pregnancies (-0.3984). These values describe model behavior; they are not percentage-point effects or causal effects.

## 9. LIME Evaluation

The existing LIME service generated eight feature weights with seed 42 and 5,000 perturbation samples. The output labels them as local surrogate weights for positive-class probability; the weights themselves are not probabilities. The service’s class-1 prediction function returned 0.80789435, matching the XGBoost prediction to within `1e-8` for this case.

The three highest absolute LIME weights were Glucose (+0.4602), DiabetesPedigreeFunction (+0.1052), and Age (+0.0945). Their scale differs from SHAP’s raw-margin scale.

## 10. SHAP/LIME Consistency

For the evaluated case, top-three overlap was **2/3 (66.7%)**, top-five overlap was **4/5 (80%)**, Spearman rank correlation was **0.4524** (`p=0.2604`), and direction agreed for **8/8** shared features. See [xai_consistency.json](../results/xai_consistency.json) and [shap_lime_rank_comparison.png](../figures/shap_lime_rank_comparison.png).

SHAP and LIME showed partial agreement for the evaluated case. This is a single-case demonstration; it does not establish global explanation stability, trustworthiness, or correctness. The two methods use different contribution scales.

## 11. Personalized Explanation Evaluation

The existing generator was run against its synthetic example with output redirected to a temporary evaluation directory, which was removed afterward. It included the individual’s Age, Glucose, SkinThickness, Insulin, and Pregnancies values; preserved positive/negative SHAP directions; and compared non-missing values with reference-dataset interquartile ranges. It identified zero-coded Insulin as missing and did not turn the feature comparison into medical advice. The generated notice states that the output is not a diagnosis or treatment recommendation.

These comparisons are relative to the training/reference dataset, not clinical normal ranges. Only this example was checked.

## 12. Backend/API Testing

The complete backend suite passed: **41 passed, 0 failed, 0 skipped**, with five warnings (Starlette/httpx deprecation, three SHAP pending deprecations, and an XGBoost serialization warning). Existing tests cover health, auth, RBAC, patients, records, predictions, SHAP, LIME, consistency, validation, authorization failures, and missing resources.

Alembic reported revision `20261015_0001 (head)`. No schema change or migration was made or required.

## 13. Authentication and RBAC

The backend tests cover bcrypt password hashing, valid and invalid login, inactive accounts, bearer-token validation, admin/clinician roles, insufficient roles, and protected endpoints. In the live API check, admin login returned 200, `/auth/me` returned 200, `/auth/rbac/admin-check` returned 200, and the token type was `bearer`. Unauthenticated and malformed-token requests returned 401; inactive-account and insufficient-role cases returned 403 in the backend tests.

No credentials, hashes, tokens, or real patient information were included in evaluation artifacts.

## 14. Frontend Regression Testing

The frontend suite passed: **55 tests across 3 files**. The production build passed with 103 modules transformed. The configured development frontend returned HTTP 200 during the live check. No UI or workflow changes were made for evaluation.

## 15. End-to-End Workflow

The working backend at `http://127.0.0.1:8002` was used without starting or stopping a backend during P5.1. A disposable synthetic case passed login → `/auth/me` → patient creation → record creation → prediction → SHAP → LIME → consistency → patient and record retrieval. The prediction was class 1 with probability 0.80789435 and model version `xgboost-baseline-v1`.

The API returned eight SHAP and eight LIME features. SHAP additivity error was `2.98e-7`; the consistency endpoint reported top-three overlap 2/3, top-five overlap 4/5, Spearman 0.4524, and direction agreement 8/8. CORS preflight accepted `http://localhost:5173`. The two disposable patients, two records, one prediction, and 16 stored explanation rows were removed afterward. The sanitized outcome is in [live_e2e.json](../results/live_e2e.json).

## 16. Data Validation

Live API checks returned 422 for missing patient code and an out-of-bound Glucose value, 409 for a duplicate patient code, and 404 for a missing patient and for a record requested under a different patient. The API accepted a nullable measurement and preserved valid zero values for pregnancies, Glucose, and Insulin. Existing backend tests also cover missing resources and invalid feature boundaries.

These numerical limits are implementation boundaries based on the project dataset, not universal clinical reference ranges.

## 17. Security Review

This was a lightweight code/configuration review, not a penetration test. The review found bcrypt hashing through the existing utility; signed, expiring JWTs; protected routes and role checks; environment-configurable JWT secrets; `DEBUG` defaulting to false; narrow localhost CORS; SQLAlchemy ORM queries; safe response schemas without password hashes; and generic error details for expected authentication failures. `.gitignore` excludes local SQLite and `.env` files, but repository tracking could not be checked because `.git` metadata is absent. No static password was found in backend application modules; the existing local-only seed script contains its documented development credential and is not included in generated artifacts.

Limitations remain: the frontend stores JWTs in `localStorage`; tokens are not revocable before expiry; no rate limiting was found; the default JWT key is generated ephemerally when no configured secret is present; local development uses SQLite and localhost HTTP; production TLS is not configured; and no formal penetration test or deployment security review was performed. This prototype is not production-secure evidence.

## 18. Responsible AI Review

The intended use is research and education for an explainable diabetes classification/risk-prediction prototype, with qualified human oversight. Out-of-scope use includes diagnosing real patients, autonomous clinical decisions, or treatment recommendations. No prospective, external, or subgroup validation was conducted. Population coverage and demographic generalization are unestablished, and model bias remains possible. Post-hoc explanations describe model behavior, not causation; SHAP/LIME disagreement is possible; raw probabilities are not clinically calibrated; and demo records are synthetic.

## 19. Limitations

- One small, population-specific dataset and one internal random split were used.
- XGBoost is identified as selected from the held-out comparison although five-fold CV ranks Logistic Regression first; the test result is therefore not an independent estimate after selection.
- Recomputed XGBoost CV means differ slightly from the stored CV artifact, and the model emits an XGBoost serialization-version warning.
- SHAP/LIME consistency and personalized wording were assessed on one synthetic case only.
- No external dataset, prospective cohort, subgroup fairness analysis, clinical workflow study, or clinician usability study was conducted.
- Calibration is descriptive and does not establish clinically useful probabilities.
- The security review was limited to source/configuration inspection and automated tests.
- Git branch/commit and tracked-file status are unavailable because this directory has no `.git` metadata.

## 20. Research Interpretation

The measurements support technical feasibility of connecting an existing diabetes model with patient-specific SHAP/LIME explanations, case-level consistency calculations, authenticated API access, and a clinician-oriented frontend. They do not establish clinical performance, generalization, fairness, explanation trustworthiness, or clinical benefit. The observed CV/test selection discrepancy should be resolved before interpreting the held-out XGBoost numbers as final model performance.

## 21. Conclusion

The prototype demonstrates that patient-specific diabetes risk prediction can be integrated with post-hoc SHAP and LIME explanations, case-level explanation consistency analysis, authenticated API access, and a clinician-oriented interface. The evaluation supports technical feasibility of the integrated workflow. However, the current evidence is limited by the single Pima dataset, case-level XAI analysis, a held-out-test model-selection caveat, slight XGBoost CV reproduction drift, absence of external/prospective validation, and prototype deployment conditions. No claim of clinical effectiveness or validity is supported.
