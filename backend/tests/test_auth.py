"""Authentication and role authorization tests against isolated SQLite."""

from datetime import timedelta

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import sessionmaker

from backend.app.api.dependencies import get_db
from backend.app.core.security import (
    JWTError,
    create_access_token,
    decode_access_token,
    hash_password,
    verify_password,
)
from backend.app.db.base import Base
from backend.app.db.models import Role, User
from backend.app.db.session import build_engine
from backend.app.main import app


TEST_PASSWORD = "test-only-password-123"
TEST_PASSWORD_HASH = hash_password(TEST_PASSWORD)


@pytest.fixture
def auth_client(tmp_path):
    db_path = (tmp_path / "auth_test.db").as_posix()
    engine = build_engine(f"sqlite:///{db_path}")
    Base.metadata.create_all(bind=engine)
    test_sessions = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)

    with test_sessions() as database:
        admin_role = Role(name="admin")
        clinician_role = Role(name="clinician")
        database.add_all([admin_role, clinician_role])
        database.flush()
        database.add_all([
            User(
                username="admin-test",
                email="admin@example.test",
                password_hash=TEST_PASSWORD_HASH,
                role=admin_role,
            ),
            User(
                username="clinician-test",
                email="clinician@example.test",
                password_hash=TEST_PASSWORD_HASH,
                role=clinician_role,
            ),
            User(
                username="inactive-test",
                email="inactive@example.test",
                password_hash=TEST_PASSWORD_HASH,
                role=clinician_role,
                is_active=False,
            ),
        ])
        database.commit()

    def override_get_db():
        with test_sessions() as database:
            yield database

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as client:
        yield client
    app.dependency_overrides.pop(get_db, None)
    Base.metadata.drop_all(bind=engine)
    engine.dispose()


def login(client: TestClient, username: str) -> str:
    response = client.post(
        "/auth/login",
        json={"username": username, "password": TEST_PASSWORD},
    )
    assert response.status_code == 200
    return response.json()["access_token"]


def test_password_hashing_and_verification():
    password_hash = hash_password("another-test-password")

    assert password_hash != "another-test-password"
    assert verify_password("another-test-password", password_hash)
    assert not verify_password("incorrect-password", password_hash)
    assert not verify_password("x" * 73, password_hash)
    with pytest.raises(ValueError):
        hash_password("x" * 73)


def test_login_valid_invalid_and_inactive_users(auth_client):
    response = auth_client.post(
        "/auth/login",
        json={"username": "clinician-test", "password": TEST_PASSWORD},
    )
    assert response.status_code == 200
    token_response = response.json()
    assert token_response["token_type"] == "bearer"
    assert token_response["expires_in"] > 0
    assert decode_access_token(token_response["access_token"])["sub"]
    assert "password_hash" not in token_response

    invalid_username = auth_client.post(
        "/auth/login",
        json={"username": "missing-user", "password": TEST_PASSWORD},
    )
    invalid_password = auth_client.post(
        "/auth/login",
        json={"username": "clinician-test", "password": "wrong-password"},
    )
    inactive = auth_client.post(
        "/auth/login",
        json={"username": "inactive-test", "password": TEST_PASSWORD},
    )
    assert invalid_username.status_code == 401
    assert invalid_password.status_code == 401
    assert inactive.status_code == 403


def test_jwt_rejects_malformed_and_expired_tokens():
    with pytest.raises(JWTError):
        decode_access_token("not-a-jwt")

    expired_token = create_access_token("1", expires_delta=timedelta(seconds=-1))
    with pytest.raises(JWTError):
        decode_access_token(expired_token)


def test_protected_me_requires_valid_token_and_returns_safe_user(auth_client):
    assert auth_client.get("/auth/me").status_code == 401
    assert auth_client.get(
        "/auth/me", headers={"Authorization": "Bearer malformed"}
    ).status_code == 401
    assert auth_client.get(
        "/auth/me", headers={"Authorization": f"Bearer {create_access_token('99999')}"}
    ).status_code == 401
    expired_token = create_access_token("1", expires_delta=timedelta(seconds=-1))
    assert auth_client.get(
        "/auth/me", headers={"Authorization": f"Bearer {expired_token}"}
    ).status_code == 401

    for username, role in (("clinician-test", "clinician"), ("admin-test", "admin")):
        token = login(auth_client, username)
        response = auth_client.get(
            "/auth/me", headers={"Authorization": f"Bearer {token}"}
        )
        assert response.status_code == 200
        body = response.json()
        assert body["username"] == username
        assert body["role"] == role
        assert "password_hash" not in body


def test_admin_role_guard(auth_client):
    clinician_token = login(auth_client, "clinician-test")
    admin_token = login(auth_client, "admin-test")

    denied = auth_client.get(
        "/auth/rbac/admin-check",
        headers={"Authorization": f"Bearer {clinician_token}"},
    )
    allowed = auth_client.get(
        "/auth/rbac/admin-check",
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert denied.status_code == 403
    assert allowed.status_code == 200
    assert allowed.json() == {"status": "authorized", "role": "admin"}
