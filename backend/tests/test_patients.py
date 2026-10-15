"""Authenticated patient/record API tests using isolated temporary SQLite."""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import sessionmaker

from backend.app.api.dependencies import get_db
from backend.app.core.security import create_access_token
from backend.app.db.base import Base
from backend.app.db.models import Role, User
from backend.app.db.session import build_engine
from backend.app.main import app


@pytest.fixture
def patient_api(tmp_path):
    db_path = (tmp_path / "patients_test.db").as_posix()
    engine = build_engine(f"sqlite:///{db_path}")
    Base.metadata.create_all(bind=engine)
    test_sessions = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)

    tokens = {}
    with test_sessions() as database:
        roles = {name: Role(name=name) for name in ("admin", "clinician", "researcher")}
        database.add_all(roles.values())
        database.flush()
        users = {
            name: User(
                username=f"{name}-patient-test",
                email=f"{name}@patient-test.example",
                password_hash="$test-hash-only$",
                role=role,
            )
            for name, role in roles.items()
        }
        database.add_all(users.values())
        database.commit()
        tokens = {
            name: create_access_token(user.id)
            for name, user in users.items()
        }

    def override_get_db():
        with test_sessions() as database:
            yield database

    app.dependency_overrides[get_db] = override_get_db
    try:
        with TestClient(app) as client:
            yield {"client": client, "tokens": tokens}
    finally:
        app.dependency_overrides.pop(get_db, None)
        Base.metadata.drop_all(bind=engine)
        engine.dispose()


def auth_headers(patient_api, role="clinician"):
    return {"Authorization": f"Bearer {patient_api['tokens'][role]}"}


def create_patient(client, headers, code="subject-001"):
    return client.post("/patients", json={"patient_code": code}, headers=headers)


def test_patient_routes_require_authentication(patient_api):
    client = patient_api["client"]
    assert client.post("/patients", json={"patient_code": "unauthorized"}).status_code == 401
    assert client.get("/patients").status_code == 401
    assert client.get("/patients/1").status_code == 401
    assert client.post(
        "/patients/1/records", json={"glucose": 100}
    ).status_code == 401
    assert client.get("/patients/1/records").status_code == 401
    assert client.get("/patients/1/records/1").status_code == 401


def test_patient_create_roles_duplicate_and_code_validation(patient_api):
    client = patient_api["client"]
    for role, code in (("clinician", "  clinician-subject  "), ("admin", "admin-subject")):
        response = create_patient(client, auth_headers(patient_api, role), code)
        assert response.status_code == 201
        body = response.json()
        assert body["patient_code"] == code  # Submitted value is preserved verbatim.
        assert set(body) == {"id", "patient_code", "created_at", "updated_at"}

    duplicate = create_patient(client, auth_headers(patient_api), "admin-subject")
    assert duplicate.status_code == 409

    for invalid_code in ("", "   ", "x" * 101):
        response = create_patient(client, auth_headers(patient_api), invalid_code)
        assert response.status_code == 422


def test_patient_get_list_pagination_and_not_found(patient_api):
    client = patient_api["client"]
    headers = auth_headers(patient_api)
    ids = []
    for number in range(3):
        response = create_patient(client, headers, f"page-subject-{number}")
        assert response.status_code == 201
        ids.append(response.json()["id"])

    got = client.get(f"/patients/{ids[1]}", headers=headers)
    assert got.status_code == 200
    assert got.json()["patient_code"] == "page-subject-1"
    assert client.get("/patients/99999", headers=headers).status_code == 404

    first_page = client.get("/patients?skip=0&limit=2", headers=headers)
    second_page = client.get("/patients?skip=2&limit=2", headers=headers)
    assert first_page.status_code == second_page.status_code == 200
    assert [row["id"] for row in first_page.json()["items"]] == ids[:2]
    assert [row["id"] for row in second_page.json()["items"]] == ids[2:]
    assert first_page.json()["total"] == 3
    assert client.get("/patients?limit=101", headers=headers).status_code == 422
    assert client.get("/patients?skip=-1", headers=headers).status_code == 422


def test_record_create_list_and_get_preserves_zero_values(patient_api):
    client = patient_api["client"]
    headers = auth_headers(patient_api)
    patient = create_patient(client, headers).json()
    record_payload = {
        "pregnancies": 0,
        "glucose": 0,
        "blood_pressure": 72,
        "skin_thickness": 0,
        "insulin": 0,
        "bmi": 33.625,
        "diabetes_pedigree_function": 0.078,
        "age": 50,
    }
    created = client.post(
        f"/patients/{patient['id']}/records",
        json=record_payload,
        headers=headers,
    )
    assert created.status_code == 201
    body = created.json()
    assert body["pregnancies"] == 0
    assert body["glucose"] == 0
    assert body["insulin"] == 0
    assert body["bmi"] == 33.625
    assert body["diabetes_pedigree_function"] == 0.078
    assert body["patient_id"] == patient["id"]

    listed = client.get(f"/patients/{patient['id']}/records", headers=headers)
    detail = client.get(
        f"/patients/{patient['id']}/records/{body['id']}", headers=headers
    )
    assert listed.status_code == 200
    assert len(listed.json()) == 1
    assert detail.status_code == 200
    assert detail.json() == body

    second = client.post(
        f"/patients/{patient['id']}/records",
        json={"glucose": 120},
        headers=headers,
    ).json()
    third = client.post(
        f"/patients/{patient['id']}/records",
        json={"glucose": 130},
        headers=headers,
    ).json()
    all_records = client.get(
        f"/patients/{patient['id']}/records?skip=0&limit=3", headers=headers
    )
    page = client.get(
        f"/patients/{patient['id']}/records?skip=1&limit=1", headers=headers
    )
    assert [item["id"] for item in all_records.json()] == [body["id"], second["id"], third["id"]]
    assert [item["id"] for item in page.json()] == [second["id"]]
    assert client.get(
        f"/patients/{patient['id']}/records?limit=101", headers=headers
    ).status_code == 422


@pytest.mark.parametrize(
    "field,value",
    [
        ("pregnancies", -1),
        ("pregnancies", 18),
        ("glucose", 200),
        ("blood_pressure", 123),
        ("skin_thickness", 100),
        ("insulin", 847),
        ("bmi", 67.2),
        ("diabetes_pedigree_function", 0.07),
        ("diabetes_pedigree_function", 2.43),
        ("age", 20),
        ("age", 82),
    ],
)
def test_invalid_record_values_rejected(patient_api, field, value):
    client = patient_api["client"]
    headers = auth_headers(patient_api)
    patient = create_patient(client, headers).json()
    response = client.post(
        f"/patients/{patient['id']}/records",
        json={field: value},
        headers=headers,
    )
    assert response.status_code == 422


def test_record_routes_not_found_and_enforce_patient_ownership(patient_api):
    client = patient_api["client"]
    headers = auth_headers(patient_api)
    patient_a = create_patient(client, headers, "record-owner-a").json()
    patient_b = create_patient(client, headers, "record-owner-b").json()
    record = client.post(
        f"/patients/{patient_a['id']}/records",
        json={"glucose": 148},
        headers=headers,
    ).json()

    assert client.post(
        "/patients/99999/records", json={"glucose": 148}, headers=headers
    ).status_code == 404
    assert client.get("/patients/99999/records", headers=headers).status_code == 404
    assert client.get(
        f"/patients/{patient_a['id']}/records/99999", headers=headers
    ).status_code == 404
    assert client.get(
        f"/patients/{patient_b['id']}/records/{record['id']}", headers=headers
    ).status_code == 404


def test_unrecognized_role_cannot_access_patient_api(patient_api):
    response = create_patient(
        patient_api["client"], auth_headers(patient_api, "researcher"), "researcher-subject"
    )
    assert response.status_code == 403
