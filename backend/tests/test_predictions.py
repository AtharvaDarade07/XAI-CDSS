"""Prediction API tests with isolated SQLite and the real P1 artifact."""

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select
from sqlalchemy.orm import sessionmaker

from ml.pipeline import FEATURES, TARGET, prepare_features

from backend.app.api.dependencies import get_db
from backend.app.core.security import create_access_token
from backend.app.db.base import Base
from backend.app.db.models import ModelVersion, Patient, PatientRecord, Prediction, Role, User
from backend.app.db.session import build_engine
from backend.app.main import app
from backend.app.services.prediction_service import (
    ModelArtifactUnavailable,
    load_model_pipeline,
    patient_record_to_features,
)


@pytest.fixture
def prediction_api(tmp_path):
    db_path = (tmp_path / "prediction_test.db").as_posix()
    engine = build_engine(f"sqlite:///{db_path}")
    Base.metadata.create_all(bind=engine)
    sessions = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)

    with sessions() as database:
        roles = {name: Role(name=name) for name in ("admin", "clinician", "researcher")}
        database.add_all(roles.values())
        database.flush()
        users = {
            name: User(
                username=f"{name}-prediction-test",
                email=f"{name}@prediction-test.example",
                password_hash="$test-hash-only$",
                role=role,
            )
            for name, role in roles.items()
        }
        database.add_all(users.values())
        patient_a = Patient(patient_code="prediction-subject-a")
        patient_b = Patient(patient_code="prediction-subject-b")
        database.add_all([patient_a, patient_b])
        database.flush()
        # This mirrors the known P2 example, including zero-coded inputs.
        record_a = PatientRecord(
            patient_id=patient_a.id,
            pregnancies=0,
            glucose=0,
            blood_pressure=72,
            skin_thickness=35,
            insulin=0,
            bmi=33.6,
            diabetes_pedigree_function=0.627,
            age=50,
        )
        record_b = PatientRecord(
            patient_id=patient_b.id,
            pregnancies=1,
            glucose=120,
            blood_pressure=70,
            skin_thickness=20,
            insulin=None,
            bmi=25.0,
            diabetes_pedigree_function=0.4,
            age=30,
        )
        database.add_all([record_a, record_b])
        database.commit()
        ids = {
            "patient_a": patient_a.id,
            "patient_b": patient_b.id,
            "record_a": record_a.id,
            "record_b": record_b.id,
        }
        tokens = {
            name: create_access_token(user.id)
            for name, user in users.items()
        }

    def override_get_db():
        with sessions() as database:
            yield database

    app.dependency_overrides[get_db] = override_get_db
    try:
        with TestClient(app) as client:
            yield {
                "client": client,
                "ids": ids,
                "tokens": tokens,
                "engine": engine,
                "sessions": sessions,
            }
    finally:
        app.dependency_overrides.pop(get_db, None)
        Base.metadata.drop_all(bind=engine)
        engine.dispose()


def headers(context, role="clinician"):
    return {"Authorization": f"Bearer {context['tokens'][role]}"}


def predict(context, role="clinician", patient_key="patient_a", record_key="record_a"):
    return context["client"].post(
        f"/patients/{context['ids'][patient_key]}/records/{context['ids'][record_key]}/predict",
        headers=headers(context, role),
    )


def test_prediction_requires_authentication_and_allowed_role(prediction_api):
    url = (
        f"/patients/{prediction_api['ids']['patient_a']}"
        f"/records/{prediction_api['ids']['record_a']}/predict"
    )
    assert prediction_api["client"].post(url).status_code == 401
    assert predict(prediction_api, "researcher").status_code == 403
    assert predict(prediction_api, "clinician").status_code == 201
    assert predict(prediction_api, "admin").status_code == 201


def test_missing_patient_record_and_wrong_patient_ownership_are_404(prediction_api):
    client = prediction_api["client"]
    auth = headers(prediction_api)
    record_id = prediction_api["ids"]["record_a"]

    assert client.post(f"/patients/99999/records/{record_id}/predict", headers=auth).status_code == 404
    assert client.post(
        f"/patients/{prediction_api['ids']['patient_a']}/records/99999/predict", headers=auth
    ).status_code == 404
    assert client.post(
        f"/patients/{prediction_api['ids']['patient_a']}/records/"
        f"{prediction_api['ids']['record_b']}/predict",
        headers=auth,
    ).status_code == 404


def test_real_pipeline_prediction_persistence_and_repeated_requests(prediction_api):
    response = predict(prediction_api)
    assert response.status_code == 201, response.text
    result = response.json()
    assert result["id"] > 0
    assert result["patient_id"] == prediction_api["ids"]["patient_a"]
    assert result["patient_record_id"] == prediction_api["ids"]["record_a"]
    assert result["predicted_class"] in (0, 1)
    assert 0 <= result["predicted_probability"] <= 1
    assert result["model_version"] == "logistic-regression-selected-v2"

    # Compare the endpoint with direct inference through the exact saved P1
    # pipeline after applying P1's shared raw-data preparation function.
    with prediction_api["sessions"]() as database:
        record = database.get(PatientRecord, prediction_api["ids"]["record_a"])
        service_features = patient_record_to_features(record)
        assert service_features.iloc[0]["Pregnancies"] == 0
        assert pd.isna(service_features.iloc[0]["Glucose"])
        assert pd.isna(service_features.iloc[0]["Insulin"])
        raw = pd.DataFrame([{
            "Pregnancies": record.pregnancies,
            "Glucose": record.glucose,
            "BloodPressure": record.blood_pressure,
            "SkinThickness": record.skin_thickness,
            "Insulin": record.insulin,
            "BMI": record.bmi,
            "DiabetesPedigreeFunction": record.diabetes_pedigree_function,
            "Age": record.age,
            TARGET: 0,
        }], columns=FEATURES + [TARGET])
        expected_features, _ = prepare_features(raw)
        pipeline = load_model_pipeline()
        direct_class = int(pipeline.predict(expected_features)[0])
        direct_probability = float(pipeline.predict_proba(expected_features)[0, 1])

    assert result["predicted_class"] == direct_class
    assert np.isclose(result["predicted_probability"], direct_probability, atol=1e-8, rtol=0)

    second = predict(prediction_api)
    assert second.status_code == 201
    assert second.json()["id"] != result["id"]
    with prediction_api["sessions"]() as database:
        predictions = database.scalars(select(Prediction)).all()
        versions = database.scalars(select(ModelVersion)).all()
        assert len(predictions) == 2
        assert len(versions) == 1
        assert versions[0].is_active is True
        assert versions[0].artifact_path == "ml/registry/logistic_regression_selected_v2.joblib"
        for row in predictions:
            assert row.patient_id == prediction_api["ids"]["patient_a"]
            assert row.patient_record_id == prediction_api["ids"]["record_a"]
            assert row.model_version_id == versions[0].id


def test_missing_artifact_returns_503_without_registry_row(prediction_api, monkeypatch, tmp_path):
    from backend.app.api.routes import predictions as prediction_route

    missing_path = tmp_path / "not-present.joblib"

    def missing_model():
        return load_model_pipeline(missing_path)

    monkeypatch.setattr(prediction_route, "load_model_pipeline", missing_model)
    response = predict(prediction_api)
    assert response.status_code == 503
    assert "not-present.joblib" not in response.text
    with pytest.raises(ModelArtifactUnavailable):
        load_model_pipeline(missing_path)
    with prediction_api["sessions"]() as database:
        assert database.scalar(select(func.count()).select_from(ModelVersion)) == 0


def test_conflicting_active_model_returns_503(prediction_api):
    with prediction_api["sessions"]() as database:
        database.add(ModelVersion(
            name="different-model",
            version="v9",
            algorithm="other",
            artifact_path="ml/registry/other.joblib",
            is_active=True,
        ))
        database.commit()

    response = predict(prediction_api)
    assert response.status_code == 503
    assert response.json()["detail"] == "Prediction model is currently unavailable"
