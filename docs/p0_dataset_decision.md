# P0 Dataset Decision

## Primary Dataset

**PIMA Indians Diabetes Dataset**

Status: **GO**

The PIMA dataset will be used as the primary dataset for the diabetes prediction and XAI prototype.

## Second Dataset Investigation

Several publicly available diabetes datasets were investigated for possible external validation.

Potential datasets included:
- Diabetes Prediction Dataset
- DiaBD — Diabetes Dataset for Enhanced Risk Analysis and Research in Bangladesh

These datasets contain overlapping clinical variables such as glucose, blood pressure, BMI and age. However, their feature definitions, populations and available variables differ from the PIMA dataset.

## External Validation Decision

**Decision: NO-GO for direct external validation in the initial implementation.**

A second dataset will not be used as a direct external test set because the available candidates do not provide sufficiently identical feature definitions and target semantics for a clean frozen-model evaluation.

Forcing feature mappings could introduce additional assumptions and make the external-validation result difficult to interpret.

## Alternative Validation Strategy

The initial implementation will use:

1. Stratified cross-validation
2. Held-out test set
3. Bootstrap 95% confidence intervals
4. Training-size sensitivity analysis
5. Reproducibility using fixed random seeds

The test set will be used only for final evaluation.

## Future Scope

External validation on an independently collected and feature-compatible diabetes dataset will remain a future research direction.

Future work may also investigate:
- Multi-population validation
- Cross-hospital validation
- Prospective clinical validation
- Dataset shift and generalization
- Fairness across demographic groups

## P0 Status

Dataset audit: COMPLETE

Data Card: COMPLETE

Second dataset investigation: COMPLETE

External validation decision: COMPLETE

**P0 GATE: READY TO CLOSE**
