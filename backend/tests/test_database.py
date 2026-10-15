"""Database integration tests using an isolated temporary SQLite file."""

import pytest
from sqlalchemy import inspect
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from backend.app.db.base import Base
from backend.app.db.models import (
    AuditLog,
    Explanation,
    ModelVersion,
    Patient,
    PatientRecord,
    Prediction,
    Role,
    User,
)
from backend.app.db.session import build_engine


@pytest.fixture
def db_session(tmp_path):
    database_path = (tmp_path / "test_cdss.db").as_posix()
    test_engine = build_engine(f"sqlite:///{database_path}")
    Base.metadata.create_all(bind=test_engine)
    with Session(test_engine) as session:
        yield session
    Base.metadata.drop_all(bind=test_engine)
    test_engine.dispose()


def test_metadata_creates_all_expected_tables(db_session):
    expected = {
        "roles", "users", "patients", "patient_records", "model_versions",
        "predictions", "explanations", "audit_logs",
    }
    assert expected.issubset(set(inspect(db_session.get_bind()).get_table_names()))
    assert expected.issubset(Base.metadata.tables.keys())


def test_entities_and_relationships_persist(db_session):
    role = Role(name="researcher", description="Research role")
    patient = Patient(patient_code="study-subject-001")
    model = ModelVersion(
        name="xgboost-baseline",
        version="1.0.0",
        algorithm="XGBoost",
        artifact_path="ml/registry/xgboost_baseline.joblib",
        metrics_json={"roc_auc": 0.80},
        is_active=True,
    )
    db_session.add_all([role, patient, model])
    db_session.flush()

    user = User(
        username="researcher-1",
        email="researcher@example.test",
        password_hash="$test-hash$",
        role=role,
    )
    record = PatientRecord(patient=patient, glucose=148, insulin=None, pregnancies=6)
    db_session.add_all([user, record])
    db_session.flush()

    prediction = Prediction(
        patient=patient,
        patient_record=record,
        model_version=model,
        predicted_class=1,
        predicted_probability=0.8079,
    )
    db_session.add(prediction)
    db_session.flush()
    explanation = Explanation(
        prediction=prediction,
        method="SHAP",
        feature="Glucose",
        feature_value=148,
        contribution=0.61,
        direction="positive",
        rank=1,
    )
    audit_log = AuditLog(
        user=user,
        action="read",
        entity_type="patient_record",
        entity_id=str(record.id),
        details_json={"purpose": "test"},
    )
    db_session.add_all([explanation, audit_log])
    db_session.commit()

    assert user.role.name == "researcher"
    assert record.patient.patient_code == "study-subject-001"
    assert prediction.patient_record.id == record.id
    assert prediction.model_version.name == "xgboost-baseline"
    assert prediction.explanations[0].method == "SHAP"
    assert audit_log.user.username == "researcher-1"
    assert record.insulin is None


def test_unique_role_username_email_and_patient_code(db_session):
    role = Role(name="clinician")
    db_session.add(role)
    db_session.flush()
    db_session.add(User(
        username="clinician-1",
        email="clinician@example.test",
        password_hash="$test-hash$",
        role=role,
    ))
    db_session.add(Patient(patient_code="patient-unique"))
    db_session.commit()

    db_session.add(Role(name="clinician"))
    with pytest.raises(IntegrityError):
        db_session.commit()
    db_session.rollback()

    db_session.add(User(
        username="clinician-1",
        email="another@example.test",
        password_hash="$test-hash$",
        role_id=role.id,
    ))
    with pytest.raises(IntegrityError):
        db_session.commit()
    db_session.rollback()

    db_session.add(User(
        username="clinician-2",
        email="clinician@example.test",
        password_hash="$test-hash$",
        role_id=role.id,
    ))
    with pytest.raises(IntegrityError):
        db_session.commit()
    db_session.rollback()

    db_session.add(Patient(patient_code="patient-unique"))
    with pytest.raises(IntegrityError):
        db_session.commit()
