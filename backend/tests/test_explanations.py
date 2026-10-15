"""Integration tests for authenticated SHAP/LIME explanations on isolated SQLite."""

import numpy as np
import pandas as pd
import pytest
import json
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import sessionmaker

from backend.app.api.dependencies import get_db
from backend.app.core.security import create_access_token
from backend.app.db.base import Base
from backend.app.db.models import Explanation, ModelVersion, Patient, PatientRecord, Prediction, Role, User
from backend.app.db.session import build_engine
from backend.app.main import app
from backend.app.services.prediction_service import (
    MODEL_ARTIFACT_RELATIVE, MODEL_NAME, MODEL_VERSION, load_model_pipeline, predict_record,
)
from backend.app.services.xai_service import (
    LIME_NUM_FEATURES, LIME_NUM_SAMPLES, LIME_RANDOM_STATE,
    normalize_lime_feature,
)
from ml.pipeline import FEATURES


@pytest.fixture
def xai_api(tmp_path):
    engine = build_engine(f"sqlite:///{(tmp_path / 'xai_test.db').as_posix()}")
    Base.metadata.create_all(bind=engine)
    sessions = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
    with sessions() as db:
        roles = {name: Role(name=name) for name in ("admin", "clinician", "researcher")}
        db.add_all(roles.values())
        db.flush()
        users = {name: User(username=f"{name}-xai", email=f"{name}@xai.example",
                            password_hash="test-hash", role=role)
                 for name, role in roles.items()}
        db.add_all(users.values())
        patient_a, patient_b = Patient(patient_code="xai-synthetic-a"), Patient(patient_code="xai-synthetic-b")
        db.add_all([patient_a, patient_b])
        db.flush()
        # Synthetic research fixture mirrors the P2 example; Insulin zero is
        # retained in storage and treated by the existing P1 rules as missing.
        record_a = PatientRecord(
            patient_id=patient_a.id, pregnancies=6, glucose=148, blood_pressure=72,
            skin_thickness=35, insulin=0, bmi=33.6,
            diabetes_pedigree_function=0.627, age=50,
        )
        record_b = PatientRecord(
            patient_id=patient_b.id, pregnancies=1, glucose=120, blood_pressure=70,
            skin_thickness=20, insulin=None, bmi=25.0,
            diabetes_pedigree_function=0.4, age=30,
        )
        version = ModelVersion(name=MODEL_NAME, version=MODEL_VERSION, algorithm="Logistic Regression",
                               artifact_path=MODEL_ARTIFACT_RELATIVE, is_active=True)
        db.add_all([record_a, record_b, version])
        db.flush()
        probability = predict_record(record_a, pipeline=load_model_pipeline()).predicted_probability
        prediction_a = Prediction(patient_id=patient_a.id, patient_record_id=record_a.id,
                                  model_version_id=version.id, predicted_class=1,
                                  predicted_probability=probability)
        prediction_b = Prediction(patient_id=patient_b.id, patient_record_id=record_b.id,
                                  model_version_id=version.id, predicted_class=0,
                                  predicted_probability=0.2)
        db.add_all([prediction_a, prediction_b])
        db.commit()
        ids = {"patient_a": patient_a.id, "patient_b": patient_b.id,
               "record_a": record_a.id, "record_b": record_b.id,
               "prediction_a": prediction_a.id, "prediction_b": prediction_b.id}
        tokens = {name: create_access_token(user.id) for name, user in users.items()}

    def override_get_db():
        with sessions() as db:
            yield db

    app.dependency_overrides[get_db] = override_get_db
    try:
        with TestClient(app) as client:
            yield {"client": client, "ids": ids, "tokens": tokens, "sessions": sessions,
                   "engine": engine}
    finally:
        app.dependency_overrides.pop(get_db, None)
        Base.metadata.drop_all(bind=engine)
        engine.dispose()


def endpoint(context, kind="shap", patient="patient_a", record="record_a", prediction="prediction_a"):
    ids = context["ids"]
    return (f"/patients/{ids[patient]}/records/{ids[record]}/predictions/"
            f"{ids[prediction]}/explanation/{kind}")


def auth(context, role="clinician"):
    return {"Authorization": f"Bearer {context['tokens'][role]}"}


def test_authentication_and_rbac(xai_api):
    assert xai_api["client"].post(endpoint(xai_api, "shap")).status_code == 401
    assert xai_api["client"].post(endpoint(xai_api, "lime")).status_code == 401
    assert xai_api["client"].get(endpoint(xai_api, "consistency")).status_code == 401
    assert xai_api["client"].post(endpoint(xai_api, "shap"), headers=auth(xai_api, "researcher")).status_code == 403
    for role in ("clinician", "admin"):
        response = xai_api["client"].post(endpoint(xai_api, "shap"), headers=auth(xai_api, role))
        assert response.status_code == 201, response.text


def test_ownership_chain_returns_404(xai_api):
    client, headers = xai_api["client"], auth(xai_api)
    base = endpoint(xai_api, "shap")
    assert client.post(base.replace(f"/{xai_api['ids']['patient_a']}/", "/99999/"), headers=headers).status_code == 404
    assert client.post(base.replace(f"/records/{xai_api['ids']['record_a']}/", "/records/99999/"), headers=headers).status_code == 404
    assert client.post(endpoint(xai_api, "shap", record="record_b"), headers=headers).status_code == 404
    assert client.post(base.replace(f"/predictions/{xai_api['ids']['prediction_a']}/", "/predictions/99999/"), headers=headers).status_code == 404
    assert client.post(endpoint(xai_api, "shap", prediction="prediction_b"), headers=headers).status_code == 404


def test_shap_values_validation_persistence_and_repeat_requests(xai_api):
    response = xai_api["client"].post(endpoint(xai_api), headers=auth(xai_api))
    assert response.status_code == 201, response.text
    result = response.json()
    assert result["method"] == "SHAP"
    assert result["prediction_id"] == xai_api["ids"]["prediction_a"]
    assert [item["feature"] for item in sorted(result["features"], key=lambda x: x["rank"])] == sorted(FEATURES, key=lambda f: next(x["rank"] for x in result["features"] if x["feature"] == f))
    assert {item["feature"] for item in result["features"]} == set(FEATURES)
    assert len(result["features"]) == len(FEATURES)
    by_feature = {item["feature"]: item for item in result["features"]}
    assert by_feature["Pregnancies"]["feature_value"] == 6
    assert by_feature["Insulin"]["feature_value"] == 0
    assert all(item["rank"] in range(1, 9) and np.isfinite(item["contribution"]) for item in result["features"])
    assert {item["direction"] for item in result["features"]} <= {"positive", "negative"}
    assert "log-odds" in result["output_space"]
    assert result["shap_additivity_error"] < 2e-4

    second = xai_api["client"].post(endpoint(xai_api), headers=auth(xai_api))
    assert second.status_code == 201
    with xai_api["sessions"]() as db:
        rows = db.scalars(select(Explanation).where(Explanation.prediction_id == result["prediction_id"])).all()
        assert len(rows) == 16
        assert {row.method for row in rows} == {"SHAP"}
        assert all(row.prediction_id == result["prediction_id"] for row in rows)


def test_lime_uses_p2_configuration_and_preserves_values(xai_api):
    response = xai_api["client"].post(endpoint(xai_api, "lime"), headers=auth(xai_api))
    assert response.status_code == 201, response.text
    result = response.json()
    assert len(result["features"]) == 8
    assert {item["feature"] for item in result["features"]} == set(FEATURES)
    by_feature = {item["feature"]: item for item in result["features"]}
    assert by_feature["Insulin"]["feature_value"] == 0
    assert result["lime_configuration"] == {
        "random_state": LIME_RANDOM_STATE,
        "num_features": LIME_NUM_FEATURES,
        "num_samples": LIME_NUM_SAMPLES,
    }
    assert result["lime_configuration"] == {"random_state": 42, "num_features": 8, "num_samples": 5000}


def test_consistency_endpoint_is_case_level_for_selected_model(xai_api):
    response = xai_api["client"].get(endpoint(xai_api, "consistency"), headers=auth(xai_api))
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["prediction_id"] == xai_api["ids"]["prediction_a"]
    assert result["top3_overlap_count"] in range(4)
    assert result["top5_overlap_count"] in range(6)
    assert -1 <= result["spearman_correlation"] <= 1
    assert result["direction_agreement_count"] + result["direction_disagreement_count"] == 8
    assert len(result["shap_top_features"]) == len(result["lime_top_features"]) == 3
    assert "this explanation instance" in result["interpretation"]

    assert result["shared_feature_count"] == 8
    assert result["direction_agreement_count"] + result["direction_disagreement_count"] == 8


def test_record_changed_after_prediction_is_rejected(xai_api):
    with xai_api["sessions"]() as db:
        record = db.get(PatientRecord, xai_api["ids"]["record_a"])
        record.glucose = 90
        db.commit()
    response = xai_api["client"].post(endpoint(xai_api), headers=auth(xai_api))
    assert response.status_code == 409


def test_incompatible_prediction_model_is_not_silently_explained(xai_api):
    with xai_api["sessions"]() as db:
        version = db.get(ModelVersion, 1)
        version.artifact_path = "ml/registry/another-model.joblib"
        db.commit()
    response = xai_api["client"].post(endpoint(xai_api), headers=auth(xai_api))
    assert response.status_code == 503


def test_missing_xai_artifact_returns_503_without_path_leak(xai_api, monkeypatch, tmp_path):
    from backend.app.api.routes import explanations as explanation_route
    from backend.app.services.prediction_service import ModelArtifactUnavailable

    def unavailable():
        raise ModelArtifactUnavailable(f"missing file: {tmp_path / 'private-model.joblib'}")

    monkeypatch.setattr(explanation_route, "load_model_pipeline", unavailable)
    response = xai_api["client"].post(endpoint(xai_api), headers=auth(xai_api))
    assert response.status_code == 503
    assert "private-model" not in response.text
