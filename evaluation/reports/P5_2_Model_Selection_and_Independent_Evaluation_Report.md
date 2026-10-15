# P5.2 Model Selection and Independent Evaluation

## 1. Problem Identified in P5.1

P5.1 was marked PARTIAL because the earlier model-selection record relied on held-out test comparison while cross-validation produced a different ranking. P5.2 resolves the selection rule at the analysis level by selecting the model using only five-fold cross-validation on the training partition and reserving the held-out partition for independent final evaluation. The held-out split was never used to rank models or tune parameters.

## 2. Evaluation Protocol

The existing feature definitions, models, hyperparameters, and preprocessing were retained. Seed 42, stratification, fold-local median imputation/standardization, zero-as-missing rules, and the existing 80/20 split were used. No hyperparameter tuning was introduced. Metrics were ROC-AUC, accuracy, precision, recall, and F1.

## 3. Dataset Split

The public Pima Indians Diabetes dataset contains 768 rows. The unchanged 80/20 stratified split (`random_state=42`) produced 614 training rows and 154 held-out rows. Dataset SHA-256: `6bfe5d0f379d17a0e0819b996407e3c09bf80febd4287f2ed212190dfff154af`. Features were kept in the established order. Invalid zero-coded medical measurements were mapped to missing before partitioning; fitted imputation and scaling were inside each pipeline fold. Pregnancies zero remained valid.

## 4. Cross-Validation Method

`StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` ran only on the 614-row training partition. Each fold independently fitted its preprocessor and estimator on four folds and scored the remaining fold. Standard deviations below are population SD across the five fold scores (`ddof=0`). Model definitions come from `ml/train.py`; no defaults or hyperparameters were adjusted for P5.2.

## 5. Primary Model-Selection Criterion

Mean ROC-AUC was predefined as the primary criterion because this prototype emits probability scores and ROC-AUC evaluates ranking across thresholds. It is a project research criterion, not a clinical decision threshold. Tie-breaks were lower ROC-AUC SD, higher mean F1, then higher mean recall. Accuracy, precision, recall, and F1 were secondary descriptive measures.

## 6. Cross-Validation Results

| Model | ROC-AUC mean ± SD | Accuracy mean ± SD | Precision mean ± SD | Recall mean ± SD | F1 mean ± SD |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8434 ± 0.0185 | 0.7882 ± 0.0191 | 0.7638 ± 0.0598 | 0.5748 ± 0.0227 | 0.6545 ± 0.0244 |
| SVC | 0.8333 ± 0.0228 | 0.7801 ± 0.0136 | 0.7415 ± 0.0644 | 0.5842 ± 0.0572 | 0.6485 ± 0.0187 |
| Random Forest | 0.8209 ± 0.0209 | 0.7687 ± 0.0271 | 0.6995 ± 0.0636 | 0.6028 ± 0.0258 | 0.6458 ± 0.0298 |
| XGBoost | 0.8023 ± 0.0252 | 0.7508 ± 0.0218 | 0.6663 ± 0.0591 | 0.5888 ± 0.0401 | 0.6223 ± 0.0229 |

Full precision is in `evaluation/results/cv_model_selection_p52.csv`.

## 7. Model Selection

Logistic Regression was selected because its mean training-partition CV ROC-AUC (0.84336) was highest, followed by SVC (0.83331), Random Forest (0.82095), and XGBoost (0.80228). The choice was made before the held-out metrics were calculated and did not depend on the earlier test ranking.

## 8. Frozen Model

The fitted pipeline was saved as `ml/registry/logistic_regression_selected_v2.joblib`, version `logistic-regression-selected-v2`; its SHA-256 is in `evaluation/results/model_selection_p52.json`. It was fitted once on the training partition with the existing Logistic Regression parameters and preprocessing. No schema migration was added. P1/P5.1 artifacts, including `xgboost_baseline.joblib`, remain present for traceability.

## 9. Independent Held-Out Test Results

The frozen Logistic Regression pipeline produced: accuracy 0.7078, precision 0.6000, recall 0.5000, F1 0.5455, and ROC-AUC 0.8130. Bootstrap 95% ROC-AUC interval: 0.7396–0.8729 (2,000 resamples, seed 42). Confusion counts: TN=82, FP=18, FN=27, TP=27. These results are a final descriptive evaluation and did not alter the selection. Results are in `final_independent_test.json/.csv`.

**Protocol execution note:** the frozen model and training-only selection were unchanged during artifact finalization, but the held-out calculation was rerun while adding calibration/ROC outputs and finishing the result files. Thus the final values are stable, but the implementation history did not satisfy a literal single computation of held-out metrics. No selection or tuning decision used those metrics. Future evaluation runs should treat this split as consumed and must not be used for further model choices.

## 10. Calibration

Internal held-out calibration assessment: Brier score 0.1754, actual positive rate 0.3506, and mean predicted probability 0.3515. The calibration curve uses eight quantile bins. This is not evidence of clinical calibration; no recalibration was performed.

## 11. Reproducibility

Repeated five-fold scores were identical in the same project environment; the selected model’s fixed synthetic example prediction/probability repeated identically. Dataset and artifact hashes, versions, and seed are recorded in `reproducibility_p52.json`. Environment: Python 3.11.9, NumPy 2.4.6, pandas 3.0.6, scikit-learn 1.9.1, XGBoost 3.2.0, SciPy 1.17.1, joblib 1.6.0.

The historical XGBoost stored CV mean was 0.8032766, while the current/P5.1 recomputation is 0.8022771 (difference -0.0009994). P5.1’s recomputed scores match this P5.2 run to output precision. Available environments have different XGBoost/SciPy/SHAP versions, but the historical runtime/package record is incomplete; **Cause not conclusively established.** See `cv_reproducibility_comparison.json`.

## 12. SHAP Evaluation

The selected model uses SHAP LinearExplainer. For the evaluated synthetic example, all eight features were returned and signed contributions summed to the Logistic Regression decision function in raw log-odds space within the service additivity tolerance. Contributions are not changes in probability or percentage points. The example predicted class 1 with probability 0.7058.

## 13. LIME Evaluation

LIME produced eight feature weights for the same synthetic example using random state 42 and 5,000 samples. Weights are local surrogate weights, not probabilities. Repeated SHAP and LIME outputs matched within 1e-10 for this case.

## 14. SHAP/LIME Consistency

For the one synthetic case, top-3 overlap was 2/3 (66.7%), top-5 overlap 4/5 (80.0%), Spearman correlation 0.6429 (p=0.0856), direction agreement 7/8, and shared feature count 8. SHAP and LIME showed partial agreement for the evaluated case. This case-level result does not establish global XAI stability or trustworthiness.

## 15. Personalized Explanation

The P5.2 evaluation summary refers to the synthetic input values and reports the selected model’s five largest absolute SHAP contributions and their directions. It contains no recommendations or medical advice. Output is recorded in `xai_case_p52.json`.

## 16. Comparison with P5.1

P5.1’s XGBoost artifact, report, system summary, and case-level XAI results were preserved. P5.1 remains historically PARTIAL. P5.2 changes the selected inference artifact to Logistic Regression based on CV-only selection. The P5.1 XGBoost example probability and XAI values are not reused as P5.2 results.

## 17. Limitations

One small dataset and one synthetic case do not support population-level or clinical conclusions. No external, prospective, subgroup, or clinical validation was done. The confidence interval is broad and applies only to this held-out split. Calibration is internal and descriptive. SHAP/LIME are local post-hoc model explanations and may disagree. Threshold, fairness, clinical utility, and human-factors evaluation remain future work.

## 18. Research Interpretation

What was measured: training-only cross-validation for four fixed models; one frozen pipeline’s held-out metrics/calibration; repeatability under the recorded software environment; SHAP additivity; and one-case LIME/consistency behavior. Backend tests exercised the selected artifact in isolated test databases. Security and responsible-AI topics were reviewed at prototype scope, not penetration-tested or clinically evaluated.

## 19. Conclusion

**Model-selection methodology corrected.** The evidence supports selecting Logistic Regression by training-partition CV and reporting its separate held-out measurements. It does not establish clinical effectiveness, clinical calibration, external validity, or global explanation consistency. The code and isolated tests use the frozen P5.2 pipeline; the live service still reports P5.1 XGBoost because it is an already-running process that was left untouched, and its configured SQLite database still marks XGBoost v1 active. Full live P5.2 verification requires a coordinated switch of that local model-version row and reload of the existing service on port 8002, then repeating the synthetic workflow against the selected artifact. No database schema migration is needed.
