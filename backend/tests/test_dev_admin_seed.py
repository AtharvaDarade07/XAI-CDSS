"""Tests for the explicit, idempotent development-admin seed script."""

from sqlalchemy import func, select
from sqlalchemy.orm import sessionmaker
from fastapi.testclient import TestClient

from backend.app.api.dependencies import get_db
from backend.app.core.security import verify_password
from backend.app.db.base import Base
from backend.app.db.models import Role, User
from backend.app.db.session import build_engine
from backend.app.main import app
from scripts.create_dev_admin import (
    DEV_ADMIN_EMAIL,
    DEV_ADMIN_PASSWORD,
    DEV_ADMIN_USERNAME,
    ensure_dev_admin,
    reset_dev_admin_password,
)


def test_dev_admin_seed_hashes_password_and_is_idempotent(tmp_path):
    engine = build_engine(f"sqlite:///{(tmp_path / 'seed_test.db').as_posix()}")
    Base.metadata.create_all(bind=engine)
    test_sessions = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)

    def override_get_db():
        with test_sessions() as database:
            yield database

    app.dependency_overrides[get_db] = override_get_db

    try:
        with test_sessions() as database:
            created_user, was_created = ensure_dev_admin(database)
            assert was_created is True
            assert created_user.username == DEV_ADMIN_USERNAME
            assert created_user.email == DEV_ADMIN_EMAIL
            assert created_user.role.name == "admin"
            assert created_user.is_active is True
            assert created_user.password_hash != DEV_ADMIN_PASSWORD
            assert verify_password(DEV_ADMIN_PASSWORD, created_user.password_hash)
            original_hash = created_user.password_hash

            # A normal rerun must preserve a deliberately changed password.
            created_user.password_hash = "not-a-valid-bcrypt-hash"
            database.commit()

            existing_user, was_created_again = ensure_dev_admin(database)
            assert was_created_again is False
            assert existing_user.id == created_user.id
            assert existing_user.password_hash == "not-a-valid-bcrypt-hash"
            assert database.scalar(select(func.count()).select_from(User)) == 1
            assert database.scalar(select(func.count()).select_from(Role)) == 1

            reset_user = reset_dev_admin_password(database)
            assert reset_user.id == existing_user.id
            assert verify_password(DEV_ADMIN_PASSWORD, reset_user.password_hash)
            assert reset_user.password_hash != original_hash

        with TestClient(app) as client:
            login = client.post(
                "/auth/login",
                json={"username": DEV_ADMIN_USERNAME, "password": DEV_ADMIN_PASSWORD},
            )
            assert login.status_code == 200
            token = login.json()["access_token"]
            profile = client.get("/auth/me", headers={"Authorization": f"Bearer {token}"})
            assert profile.status_code == 200
            assert profile.json()["role"] == "admin"
            assert profile.json()["username"] == DEV_ADMIN_USERNAME
            admin_check = client.get(
                "/auth/rbac/admin-check", headers={"Authorization": f"Bearer {token}"}
            )
            assert admin_check.status_code == 200
            assert admin_check.json() == {"status": "authorized", "role": "admin"}
    finally:
        app.dependency_overrides.pop(get_db, None)
        Base.metadata.drop_all(bind=engine)
        engine.dispose()
