"""Reproduce P5.1 metrics from the locked split and existing model artifacts."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import platform
import subprocess
import sys

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DATA_PATH = ROOT / "data" / "raw" / "pima-indians-diabetes.csv"
REGISTRY = ROOT / "ml" / "registry"
EXISTING_RESULTS = ROOT / "ml" / "results"
RESULTS = ROOT / "evaluation" / "results"
FIGURES = ROOT / "evaluation" / "figures"
SEED = 42
TEST_SIZE = 0.20
MODEL_NAMES = ("logistic_regression", "random_forest", "svc", "xgboost")
FEATURES = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin",
    "BMI", "DiabetesPedigreeFunction", "Age",
]
ZERO_AS_MISSING = ("Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI")
METRIC_NAMES = ("accuracy", "precision", "recall", "f1", "roc_auc")


def fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def load_locked_data() -> tuple[pd.DataFrame, pd.Series]:
    raw = pd.read_csv(DATA_PATH, header=None)
    raw.columns = FEATURES + ["Outcome"]
    x = raw[FEATURES].copy()
    y = raw["Outcome"].astype(int).copy()
    # Deterministic data interpretation precedes splitting; learned imputation
    # and scaling remain inside each saved training pipeline.
    for feature in ZERO_AS_MISSING:
        x.loc[x[feature] == 0, feature] = np.nan
    return x, y


def package_versions() -> dict[str, str]:
    names = {
        "numpy": "numpy", "pandas": "pandas", "scikit_learn": "scikit-learn",
        "xgboost": "xgboost", "shap": "shap", "lime": "lime",
        "matplotlib": "matplotlib", "fastapi": "fastapi", "sqlalchemy": "sqlalchemy",
        "pytest": "pytest", "pydantic": "pydantic", "alembic": "alembic",
        "scipy": "scipy", "joblib": "joblib", "passlib": "passlib",
        "bcrypt": "bcrypt", "python_jose": "python-jose",
    }
    result = {}
    for key, package in names.items():
        try:
            result[key] = metadata.version(package)
        except metadata.PackageNotFoundError:
            result[key] = "not-installed"
    return result


def frontend_package_versions() -> dict[str, str]:
    lock_path = ROOT / "frontend" / "package-lock.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    packages = lock.get("packages", {})
    names = ("react", "react-dom", "react-router-dom", "axios", "vite", "typescript", "vitest")
    return {
        name: packages.get(f"node_modules/{name}", {}).get("version", "not-locked")
        for name in names
    }


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    x, y = load_locked_data()
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=TEST_SIZE, stratify=y, random_state=SEED,
    )
    model_outputs: dict[str, dict[str, object]] = {}
    metric_rows: list[dict[str, object]] = []
    roc_data: list[tuple[str, np.ndarray, np.ndarray, float]] = []
    confusions: dict[str, dict[str, int]] = {}
    warnings_seen: list[str] = []

    for name in MODEL_NAMES:
        path = REGISTRY / f"{name}_baseline.joblib"
        if not path.is_file():
            raise FileNotFoundError(f"Required trained pipeline is missing: {path}")
        import warnings
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            pipeline = joblib.load(path)
        warnings_seen.extend(f"{name}: {item.message}" for item in caught)
        names = list(getattr(pipeline, "feature_names_in_", []))
        if names and names != FEATURES:
            raise ValueError(f"Feature order mismatch in {name} artifact")

        predicted = np.asarray(pipeline.predict(x_test), dtype=int)
        probability = np.asarray(pipeline.predict_proba(x_test)[:, 1], dtype=float)
        metrics = {
            "accuracy": float(accuracy_score(y_test, predicted)),
            "precision": float(precision_score(y_test, predicted, zero_division=0)),
            "recall": float(recall_score(y_test, predicted, zero_division=0)),
            "f1": float(f1_score(y_test, predicted, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, probability)),
        }
        for metric, value in metrics.items():
            metric_rows.append({"model": name, "metric": metric, "value": value})
        matrix = confusion_matrix(y_test, predicted, labels=[0, 1])
        confusions[name] = {
            "tn": int(matrix[0, 0]), "fp": int(matrix[0, 1]),
            "fn": int(matrix[1, 0]), "tp": int(matrix[1, 1]),
        }
        fpr, tpr, _ = roc_curve(y_test, probability)
        roc_data.append((name, fpr, tpr, metrics["roc_auc"]))
        model_outputs[name] = {
            "metrics": metrics,
            "confusion_matrix": confusions[name],
            "predicted_class": predicted,
            "probability": probability,
            "pipeline": pipeline,
        }

    pd.DataFrame(metric_rows).to_csv(RESULTS / "model_comparison.csv", index=False, float_format="%.12g")
    write_json(RESULTS / "confusion_matrices.json", confusions)

    fig, ax = plt.subplots(figsize=(8.4, 6.4), constrained_layout=True)
    display_names = {
        "logistic_regression": "Logistic Regression", "random_forest": "Random Forest",
        "svc": "SVC", "xgboost": "XGBoost",
    }
    for name, fpr, tpr, auc in roc_data:
        ax.plot(fpr, tpr, linewidth=2, label=f"{display_names[name]} (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], color="0.5", linestyle="--", linewidth=1, label="Chance reference")
    ax.set(xlim=(0, 1), ylim=(0, 1.02), xlabel="False positive rate", ylabel="True positive rate",
           title="Held-out ROC comparison (research dataset)")
    ax.legend(loc="lower right", frameon=True)
    ax.grid(alpha=0.2)
    fig.savefig(FIGURES / "model_roc_comparison.png", dpi=180)
    plt.close(fig)

    selected = model_outputs["xgboost"]
    matrix = np.array([[confusions["xgboost"]["tn"], confusions["xgboost"]["fp"]],
                       [confusions["xgboost"]["fn"], confusions["xgboost"]["tp"]]])
    fig, ax = plt.subplots(figsize=(6.2, 5.2), constrained_layout=True)
    heatmap = ax.imshow(matrix, cmap="Blues")
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["0", "1"], yticklabels=["0", "1"],
           xlabel="Predicted class", ylabel="Observed class", title="XGBoost held-out confusion matrix")
    for row in range(2):
        for col in range(2):
            ax.text(col, row, str(matrix[row, col]), ha="center", va="center", fontsize=14,
                    color="white" if matrix[row, col] > matrix.max() / 2 else "black")
    fig.colorbar(heatmap, ax=ax, label="Number of records")
    fig.savefig(FIGURES / "model_confusion_matrix.png", dpi=180)
    plt.close(fig)

    xgb_probability = np.asarray(selected["probability"])
    brier = float(brier_score_loss(y_test, xgb_probability))
    positive_rate = float(y_test.mean())
    mean_probability = float(xgb_probability.mean())
    observed, predicted = calibration_curve(y_test, xgb_probability, n_bins=10, strategy="uniform")
    bin_counts, bin_edges = np.histogram(xgb_probability, bins=np.linspace(0, 1, 11))
    calibration_bins = [
        {"bin_lower": float(bin_edges[i]), "bin_upper": float(bin_edges[i + 1]),
         "count": int(bin_counts[i])}
        for i in range(len(bin_counts))
    ]
    calibration = {
        "model": "xgboost-baseline-v1", "brier_score": brier,
        "actual_positive_rate": positive_rate, "mean_predicted_probability": mean_probability,
        "bin_strategy": "10 uniform-width bins; empty bins omitted by sklearn calibration_curve",
        "observed_positive_fraction": [float(v) for v in observed],
        "mean_predicted_probability_by_nonempty_bin": [float(v) for v in predicted],
        "bin_counts": calibration_bins,
        "interpretation": "Brier score and reliability bins describe held-out probability errors; they do not establish clinical calibration.",
    }
    write_json(RESULTS / "calibration.json", calibration)

    fig, (ax, hist_ax) = plt.subplots(2, 1, figsize=(7.4, 8.0), sharex=True,
                                      gridspec_kw={"height_ratios": [3, 1]}, constrained_layout=True)
    ax.plot([0, 1], [0, 1], linestyle="--", color="0.5", label="Perfect calibration reference")
    ax.plot(predicted, observed, marker="o", linewidth=2, label="XGBoost")
    ax.set(ylabel="Observed positive fraction", title="Held-out XGBoost calibration (research dataset)", xlim=(0, 1), ylim=(0, 1))
    ax.legend(loc="best")
    ax.grid(alpha=0.2)
    hist_ax.hist(xgb_probability, bins=np.linspace(0, 1, 11), color="#4c78a8", edgecolor="white")
    hist_ax.set(xlabel="Mean predicted probability", ylabel="Records")
    fig.savefig(FIGURES / "calibration_curve.png", dpi=180)
    plt.close(fig)

    # Repeated inference and explanations use the project's frozen P2 service.
    from backend.app.db.models import PatientRecord
    from backend.app.services.prediction_service import predict_record
    from backend.app.services.xai_service import (
        LIME_NUM_FEATURES, LIME_NUM_SAMPLES, LIME_RANDOM_STATE,
        compare_explanations, generate_lime_explanation, generate_shap_explanation,
    )
    from backend.app.services.xai_service import _lime_predict_fn
    record = PatientRecord(
        pregnancies=6, glucose=148, blood_pressure=72, skin_thickness=35,
        insulin=0, bmi=33.6, diabetes_pedigree_function=0.627, age=50,
    )
    pipeline = selected["pipeline"]
    prediction_a = predict_record(record, pipeline=pipeline)
    prediction_b = predict_record(record, pipeline=pipeline)
    shap_a = generate_shap_explanation(record, pipeline=pipeline)
    shap_b = generate_shap_explanation(record, pipeline=pipeline)
    lime_a = generate_lime_explanation(record, pipeline=pipeline)
    lime_b = generate_lime_explanation(record, pipeline=pipeline)
    agreement = compare_explanations(shap_a, lime_a)
    raw_case = np.array([[6, 148, 72, 35, 0, 33.6, 0.627, 50]], dtype=float)
    lime_prediction_probability = float(_lime_predict_fn(pipeline)(raw_case)[0, 1])
    lime_probability_matches = bool(np.isclose(
        lime_prediction_probability, prediction_a.predicted_probability, atol=1e-8, rtol=0,
    ))
    shap_values_a = {item.feature: item.contribution for item in shap_a.features}
    shap_values_b = {item.feature: item.contribution for item in shap_b.features}
    lime_values_a = {item.feature: item.contribution for item in lime_a.features}
    lime_values_b = {item.feature: item.contribution for item in lime_b.features}
    predictions_same = prediction_a == prediction_b
    shap_same = all(np.isclose(shap_values_a[name], shap_values_b[name], atol=1e-10, rtol=0) for name in FEATURES)
    lime_same = all(np.isclose(lime_values_a[name], lime_values_b[name], atol=1e-10, rtol=0) for name in FEATURES)
    expected_features = all(item.feature in FEATURES for item in shap_a.features + lime_a.features)
    dataset_hash = fingerprint(DATA_PATH)
    artifact_path = REGISTRY / "xgboost_baseline.joblib"
    case_values = dict(zip(FEATURES, [6, 148, 72, 35, 0, 33.6, 0.627, 50]))
    reproducibility = {
        "random_seed": SEED,
        "dataset_file": str(DATA_PATH.relative_to(ROOT)),
        "dataset_sha256": dataset_hash,
        "model_version": "xgboost-baseline-v1",
        "model_artifact_sha256": fingerprint(artifact_path),
        "python_version": platform.python_version(),
        "package_versions": package_versions(),
        "preprocessing": {
            "zero_as_missing": list(ZERO_AS_MISSING),
            "pregnancies_zero_valid": True,
            "imputation": "median fitted inside training pipeline",
            "scaling": "StandardScaler fitted inside training pipeline",
            "feature_order": FEATURES,
        },
        "split": {"test_size": TEST_SIZE, "stratified": True, "random_state": SEED},
        "cross_validation": {
            "folds": 5, "stratified": True, "shuffle": True, "random_state": SEED,
            "source": "Recomputed in P5.1 using ml/train.py estimators and preprocessor on the training partition only",
        },
        "lime": {"random_state": LIME_RANDOM_STATE, "num_features": LIME_NUM_FEATURES,
                 "num_samples": LIME_NUM_SAMPLES},
        "repeated_prediction_results": {
            "input": case_values,
            "prediction_1": {"class": prediction_a.predicted_class, "probability": prediction_a.predicted_probability},
            "prediction_2": {"class": prediction_b.predicted_class, "probability": prediction_b.predicted_probability},
            "prediction_identical": predictions_same,
            "shap_values_identical_within_1e-10": shap_same,
            "lime_weights_identical_within_1e-10": lime_same,
        },
        "model_load_warnings": warnings_seen,
        "reproducibility_status": "PASS" if predictions_same and shap_same and lime_same and expected_features else "PARTIAL",
    }
    write_json(RESULTS / "reproducibility.json", reproducibility)

    shap_map = {item.feature: item for item in shap_a.features}
    lime_map = {item.feature: item for item in lime_a.features}
    ranking = {
        "case_level_only": True,
        "patient_input": {name: (None if name == "Insulin" else float(value)) for name, value in zip(FEATURES, [6, 148, 72, 35, None, 33.6, 0.627, 50])},
        "predicted_class": prediction_a.predicted_class,
        "predicted_probability": prediction_a.predicted_probability,
        "shap": {
            "feature_count": len(shap_a.features), "feature_order": FEATURES,
            "contribution_output_space": shap_a.output_space,
            "base_value": shap_a.base_value, "contribution_sum": shap_a.contribution_sum,
            "raw_model_margin": shap_a.model_output, "additivity_absolute_error": shap_a.validation_error,
            "additivity_verified": bool(shap_a.validation_error is not None and shap_a.validation_error <= 2e-4),
            "contributions": [{"feature": name, "value": shap_map[name].contribution,
                               "direction": shap_map[name].direction, "rank": shap_map[name].rank}
                              for name in FEATURES],
        },
        "lime": {
            "feature_count": len(lime_a.features), "feature_order": FEATURES,
            "configuration": {"random_state": LIME_RANDOM_STATE, "num_samples": LIME_NUM_SAMPLES,
                              "num_features": LIME_NUM_FEATURES},
            "output_space": lime_a.output_space,
            "weights_are_probabilities": False,
            "predict_fn_probability_class1": lime_prediction_probability,
            "prediction_probability_matches_xgboost": lime_probability_matches,
            "contributions": [{"feature": name, "weight": lime_map[name].contribution,
                               "direction": lime_map[name].direction, "rank": lime_map[name].rank}
                              for name in FEATURES],
        },
        "consistency": agreement,
        "interpretation": "SHAP and LIME showed partial agreement for the evaluated synthetic case. A single case does not establish global explanation stability or trustworthiness.",
    }
    write_json(RESULTS / "xai_consistency.json", ranking)

    # Rank plot displays ranks only; SHAP and LIME contribution units differ.
    shap_order = sorted(FEATURES, key=lambda n: shap_map[n].rank)
    lime_order = sorted(FEATURES, key=lambda n: lime_map[n].rank)
    fig, ax = plt.subplots(figsize=(8, 5.5), constrained_layout=True)
    y_positions = np.arange(len(FEATURES))
    ax.scatter([shap_map[n].rank for n in shap_order], y_positions, label="SHAP rank", marker="o", s=45)
    ax.scatter([lime_map[n].rank for n in shap_order], y_positions, label="LIME rank", marker="s", s=42)
    ax.set(yticks=y_positions, yticklabels=shap_order, xticks=range(1, 9), xlim=(0.5, 8.5),
           xlabel="Absolute contribution rank (1 = highest)", title="Case-level SHAP/LIME feature ranking")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.2)
    ax.legend()
    fig.savefig(FIGURES / "shap_lime_rank_comparison.png", dpi=180)
    plt.close(fig)

    existing_cv = pd.read_csv(EXISTING_RESULTS / "cross_validation_results.csv")
    prior_selection = json.loads((EXISTING_RESULTS / "model_selection.json").read_text(encoding="utf-8"))
    prior_final_selection = json.loads((EXISTING_RESULTS / "final_model.json").read_text(encoding="utf-8"))
    prior_test = pd.read_csv(EXISTING_RESULTS / "final_test_results.csv")
    current_test = pd.DataFrame(metric_rows).pivot(index="model", columns="metric", values="value")
    previous_test = prior_test.set_index("model")
    compared_metrics = list(METRIC_NAMES)
    prior_match = all(
        np.isclose(current_test.loc[name, metric], previous_test.loc[name, metric], atol=1e-9, rtol=0)
        for name in MODEL_NAMES for metric in compared_metrics
    )

    # Re-run five-fold CV only inside the 80% training set with the existing
    # estimator/preprocessor definitions. Sequential fold scheduling affects
    # execution only; estimator hyperparameters remain unchanged.
    from ml.train import build_models, build_preprocessor
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    scoring = {name: name for name in METRIC_NAMES}
    cv_rows = []
    for name, estimator in build_models().items():
        cv_result = cross_validate(
            Pipeline([("preprocessor", build_preprocessor()), ("model", estimator)]),
            x_train, y_train, cv=cv, scoring=scoring, n_jobs=1,
        )
        for metric in METRIC_NAMES:
            values = cv_result[f"test_{metric}"]
            cv_rows.append({"model": name, "metric": metric,
                            "mean": float(np.mean(values)), "std": float(np.std(values))})
    current_cv = pd.DataFrame(cv_rows)
    current_cv.to_csv(RESULTS / "cross_validation_recomputed.csv", index=False, float_format="%.12g")
    old_cv_long = existing_cv.melt(
        id_vars="model", value_vars=[f"{metric}_mean" for metric in METRIC_NAMES],
        var_name="metric", value_name="mean",
    )
    old_cv_long["metric"] = old_cv_long["metric"].str.removesuffix("_mean")
    cv_reference_match = all(
        np.isclose(
            current_cv.loc[(current_cv.model == row.model) & (current_cv.metric == row.metric), "mean"].iloc[0],
            row.mean, atol=1e-9, rtol=0,
        )
        for row in old_cv_long.itertuples(index=False)
    )
    reproducibility["cross_validation"]["matches_existing_means_within_1e-9"] = bool(cv_reference_match)
    write_json(RESULTS / "reproducibility.json", reproducibility)

    environment = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "os": platform.platform(), "python_version": platform.python_version(),
        "package_versions": package_versions(),
        "frontend_node_version": subprocess.run(["node", "--version"], capture_output=True,
                                                 text=True, check=False).stdout.strip() or "unavailable",
        "frontend_npm_version": subprocess.run(["npm.cmd", "--version"], capture_output=True,
                                                 text=True, check=False).stdout.strip() or "unavailable",
        "frontend_package_versions": frontend_package_versions(),
        "model_version": "xgboost-baseline-v1",
        "dataset_filename": DATA_PATH.name, "dataset_sha256": dataset_hash,
        "git_commit_hash": None,
        "git_status": "unavailable: project directory has no .git metadata",
        "ml_artifact_hashes": {
            f"{name}_baseline.joblib": fingerprint(REGISTRY / f"{name}_baseline.joblib")
            for name in MODEL_NAMES
        },
    }
    write_json(RESULTS / "environment.json", environment)
    write_json(RESULTS / "cross_validation_reference.json", {
        "protocol": prior_selection,
        "results": json.loads(current_cv.to_json(orient="records")),
        "existing_cv_means_match_within_1e-9": bool(cv_reference_match),
        "recomputed_during_p51": True,
        "selection_note": "CV results are training-partition estimates. The CV winner is determined by the recomputed table; final_model.json records XGBoost as highest on the same held-out test set, so the test set was used for selection and is not an independent unbiased estimate for that selected model.",
        "final_model_record": prior_final_selection,
    })
    write_json(RESULTS / "metric_reproduction_check.json", {
        "prior_held_out_metrics_match_within_1e-9": bool(prior_match),
        "comparison_source": "existing ml/results/final_test_results.csv",
        "evaluated_test_rows": int(len(y_test)),
        "test_positive_rate": positive_rate,
        "xgboost_brier_score": brier,
    })
    print(json.dumps({
        "dataset_rows": len(y), "train_rows": len(y_train), "test_rows": len(y_test),
        "test_positive_rate": positive_rate, "held_out_metrics_match_existing": bool(prior_match),
        "xgboost_metrics": selected["metrics"], "xgboost_confusion": confusions["xgboost"],
        "brier_score": brier, "mean_predicted_probability": mean_probability,
        "reproducibility_status": reproducibility["reproducibility_status"],
        "shap_additivity_error": shap_a.validation_error,
        "lime_config": ranking["lime"]["configuration"],
        "consistency": agreement,
        "warning_count": len(warnings_seen),
    }, indent=2))


if __name__ == "__main__":
    main()
