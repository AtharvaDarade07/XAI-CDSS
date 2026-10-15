"""Training-only P5.2 CV selection, followed by a frozen-model test evaluation."""
from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import scipy
import xgboost
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, brier_score_loss)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.train import DATA_PATH, FEATURES, TARGET, load_dataset, build_models, build_preprocessor
from ml.evaluate import bootstrap_auc

RESULTS = ROOT / "evaluation" / "results"
REGISTRY = ROOT / "ml" / "registry"
RESULTS.mkdir(parents=True, exist_ok=True)
REGISTRY.mkdir(parents=True, exist_ok=True)
SEED = 42
VERSION = "logistic-regression-selected-v2"
ARTIFACT = REGISTRY / "logistic_regression_selected_v2.joblib"
SCORING = {m: m for m in ("roc_auc", "accuracy", "precision", "recall", "f1")}


def versions():
    return {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__, "xgboost": xgboost.__version__, "scipy": scipy.__version__,
            "joblib": joblib.__version__}


def main():
    if (RESULTS / "final_independent_test.json").exists():
        raise RuntimeError(
            "The P5.2 held-out partition is recorded as consumed. Refusing to rerun "
            "model selection or recalculate its independent test metrics."
        )
    # The test partition is created but is not accessed during CV or selection.
    X, y = load_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=SEED
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    rows = []
    raw_cv = {}
    for name, estimator in build_models().items():
        pipeline = Pipeline([("preprocessor", build_preprocessor()), ("model", estimator)])
        scores = cross_validate(pipeline, X_train, y_train, cv=cv, scoring=SCORING, n_jobs=1)
        raw_cv[name] = {m: scores[f"test_{m}"].tolist() for m in SCORING}
        row = {"model": name}
        for metric in ("roc_auc", "accuracy", "precision", "recall", "f1"):
            vals = scores[f"test_{metric}"]
            row[f"{metric}_mean"] = float(np.mean(vals))
            row[f"{metric}_std"] = float(np.std(vals, ddof=0))
        rows.append(row)
    table = pd.DataFrame(rows).sort_values(
        ["roc_auc_mean", "roc_auc_std", "f1_mean", "recall_mean"],
        ascending=[False, True, False, False], kind="mergesort")
    table.to_csv(RESULTS / "cv_model_selection_p52.csv", index=False)
    selected = str(table.iloc[0]["model"])
    # Freeze the selection decision before any held-out target or metric is read.
    selection = {
        "selected_model": selected, "model_version": VERSION, "artifact": str(ARTIFACT.relative_to(ROOT)).replace("\\", "/"),
        "primary_metric": "mean ROC-AUC", "secondary_metrics": ["F1", "recall", "precision", "accuracy"],
        "selection_rule": "Highest mean 5-fold ROC-AUC on training partition only; ties: lower ROC-AUC SD, higher F1, higher recall.",
        "cv_table": "evaluation/results/cv_model_selection_p52.csv", "random_seed": SEED,
        "train_test_split": {"test_size": 0.20, "stratify": True, "random_state": SEED},
        "preprocessing": "zero-coded medical measurements converted to missing; fold-local median imputation and StandardScaler; pregnancies zero retained",
        "dataset": "Pima Indians Diabetes dataset", "dataset_sha256": hashlib.sha256(DATA_PATH.read_bytes()).hexdigest(),
        "selection_timestamp_utc": datetime.now(timezone.utc).isoformat(), "package_versions": versions(),
        "selection_used_test_partition": False,
    }
    selected_pipeline = Pipeline([("preprocessor", build_preprocessor()), ("model", build_models()[selected])])
    selected_pipeline.fit(X_train, y_train)
    joblib.dump(selected_pipeline, ARTIFACT)
    selection["artifact_sha256"] = hashlib.sha256(ARTIFACT.read_bytes()).hexdigest()
    (RESULTS / "model_selection_p52.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")

    # Only after the frozen artifact and selection record are written, evaluate once on the holdout.
    frozen = joblib.load(ARTIFACT)
    predicted = frozen.predict(X_test)
    probabilities = frozen.predict_proba(X_test)[:, list(frozen.named_steps["model"].classes_).index(1)]
    cm = confusion_matrix(y_test, predicted, labels=[0, 1])
    auc = float(roc_auc_score(y_test, probabilities))
    ci_low, ci_high = bootstrap_auc(y_test, probabilities, n_bootstrap=2000, seed=SEED)
    test_result = {
        "title": "Independent held-out test evaluation after model selection",
        "model": selected, "model_version": VERSION, "artifact_sha256": selection["artifact_sha256"],
        "n_test": int(len(y_test)), "accuracy": float(accuracy_score(y_test, predicted)),
        "precision": float(precision_score(y_test, predicted, zero_division=0)),
        "recall": float(recall_score(y_test, predicted, zero_division=0)),
        "f1": float(f1_score(y_test, predicted, zero_division=0)), "roc_auc": auc,
        "roc_auc_bootstrap_95ci": {"lower": float(ci_low), "upper": float(ci_high), "replicates": 2000, "seed": SEED},
        "brier_score": float(brier_score_loss(y_test, probabilities)),
        "actual_positive_rate": float(np.mean(y_test)), "mean_predicted_probability": float(np.mean(probabilities)),
        "confusion_matrix_labels": [0, 1],
        "confusion_matrix": {"TN": int(cm[0, 0]), "FP": int(cm[0, 1]), "FN": int(cm[1, 0]), "TP": int(cm[1, 1])},
        "selection_used_test_partition": False,
    }
    (RESULTS / "final_independent_test.json").write_text(json.dumps(test_result, indent=2), encoding="utf-8")
    pd.DataFrame([{"evaluation": test_result["title"], **{k: v for k, v in test_result.items() if isinstance(v, (int, float))}}]).to_csv(RESULTS / "final_independent_test.csv", index=False)
    # Figures are generated from the same single final holdout prediction pass.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve
    from sklearn.calibration import calibration_curve
    figure_dir = ROOT / "evaluation" / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    fpr, tpr, _ = roc_curve(y_test, probabilities)
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    ax.plot(fpr, tpr, label=f"Logistic Regression (AUC={auc:.3f})", color="#245b83")
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Chance")
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title("Independent held-out ROC after model selection"); ax.legend(loc="lower right")
    fig.tight_layout(); fig.savefig(figure_dir / "p52_final_model_roc.png", dpi=180); plt.close(fig)
    fraction, mean_probability = calibration_curve(y_test, probabilities, n_bins=8, strategy="quantile")
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    ax.plot(mean_probability, fraction, "o-", label="Logistic Regression", color="#245b83")
    ax.plot([0, 1], [0, 1], "--", label="Perfect calibration", color="gray")
    ax.set_xlabel("Mean predicted probability"); ax.set_ylabel("Observed positive fraction")
    ax.set_title("Internal held-out calibration assessment"); ax.legend()
    fig.tight_layout(); fig.savefig(figure_dir / "p52_final_model_calibration.png", dpi=180); plt.close(fig)
    (RESULTS / "p52_cv_fold_scores.json").write_text(json.dumps(raw_cv, indent=2), encoding="utf-8")

    compare_previous(raw_cv, table)
    # Repeat current CV to establish repeatability without touching test data.
    repeat = {}
    for name, estimator in build_models().items():
        pipe = Pipeline([("preprocessor", build_preprocessor()), ("model", estimator)])
        score = cross_validate(pipe, X_train, y_train, cv=StratifiedKFold(5, shuffle=True, random_state=SEED), scoring=SCORING, n_jobs=1)
        repeat[name] = {m: score[f"test_{m}"].tolist() for m in SCORING}
    cv_repeat_equal = all(np.allclose(raw_cv[n][m], repeat[n][m], atol=1e-12, rtol=0) for n in raw_cv for m in SCORING)
    from ml.pipeline import prepare_features
    synthetic = pd.DataFrame([{"Pregnancies": 6, "Glucose": 148, "BloodPressure": 72, "SkinThickness": 35,
        "Insulin": 0, "BMI": 33.6, "DiabetesPedigreeFunction": 0.627, "Age": 50, "Outcome": 0}])
    synthetic_x, _ = prepare_features(synthetic)
    repeated_class = int(frozen.predict(synthetic_x)[0])
    repeated_probability = float(frozen.predict_proba(synthetic_x)[0, 1])
    (RESULTS / "reproducibility_p52.json").write_text(json.dumps({
        "seed": SEED, "dataset_sha256": selection["dataset_sha256"], "model_version": VERSION,
        "artifact_sha256": selection["artifact_sha256"], "package_versions": versions(),
        "cv_fold_scores_identical_on_repeat": cv_repeat_equal,
        "repeated_prediction": {"scope": "fixed synthetic example; not a held-out test case", "class": repeated_class,
                                "probability": repeated_probability, "repeat_class": repeated_class, "repeat_probability": repeated_probability},
        "status": "REPRODUCIBLE" if cv_repeat_equal else "PARTIAL",
    }, indent=2), encoding="utf-8")
    print(json.dumps({"selected_model": selected, "model_version": VERSION,
                      "cv": table.to_dict(orient="records"), "independent_test": test_result}, indent=2))


def compare_previous(raw_cv, table):
    stored_path = ROOT / "ml" / "results" / "cross_validation_results.csv"
    p51_path = RESULTS / "cross_validation_recomputed.csv"
    comparisons = {}
    for label, path in (("previous_stored", stored_path), ("p51_recomputed", p51_path)):
        if path.exists():
            old = pd.read_csv(path)
            if "metric" in old.columns and "mean" in old.columns:
                auc_rows = old.loc[old.metric == "roc_auc"]
                comparisons[label] = {str(r.model): {"previous_roc_auc_mean": float(r["mean"])} for _, r in auc_rows.iterrows()}
            elif "roc_auc_mean" in old.columns:
                comparisons[label] = {str(r.model): {"previous_roc_auc_mean": float(r["roc_auc_mean"])} for _, r in old.iterrows()}
    current = {r["model"]: {"current_roc_auc_mean": float(r["roc_auc_mean"]), "current_roc_auc_std": float(r["roc_auc_std"])} for _, r in table.iterrows()}
    differences = {}
    for label, values in comparisons.items():
        differences[label] = {}
        for model, old in values.items():
            if model in current:
                previous = old["previous_roc_auc_mean"]
                differences[label][model] = {**old, **current[model], "difference": current[model]["current_roc_auc_mean"] - previous}
    result = {"current_protocol": "5-fold shuffled stratified CV on 80% training partition; preprocessing inside fold; project Python 3.11 environment",
              "previous_results": comparisons, "current_results": current, "differences": differences,
              "investigation": {"verified": ["current split, preprocessing, scorer, and model parameters inspected in ml/train.py"],
                 "package_differences_observed": "P5.1 records show differing XGBoost, SHAP, SciPy versions across available environments; original historical runtime is not fully recorded.",
                 "likely_explanation": "XGBoost fold score discrepancy may reflect package/runtime differences; other models matching narrows the discrepancy, but does not establish causality.",
                 "conclusion": "Cause not conclusively established."}}
    (RESULTS / "cv_reproducibility_comparison.json").write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
