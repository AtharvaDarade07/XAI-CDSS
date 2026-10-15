"""Create the documented local-only development administrator account."""

from __future__ import annotations

import sys
import argparse
from pathlib import Path

# Running a script by path puts scripts/ rather than the repository root first
# on sys.path. Add the root so the existing backend package imports reliably.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from backend.app.core.security import hash_password
from backend.app.db.models import Role, User
from backend.app.db.session import SessionLocal


DEV_ADMIN_USERNAME = "admin"
DEV_ADMIN_EMAIL = "admin@example.com"
# This credential is intentionally limited to the one-off local development seed.
DEV_ADMIN_PASSWORD = "Admin@123"
DEV_ADMIN_ROLE = "admin"


def ensure_dev_admin(database: Session) -> tuple[User, bool]:
    """Create the admin role/account once, without changing an existing account."""
    existing_users = database.scalars(
        select(User).where(
            or_(
                User.username == DEV_ADMIN_USERNAME,
                User.email == DEV_ADMIN_EMAIL,
            )
        )
    ).all()

    if existing_users:
        if len(existing_users) != 1:
            raise RuntimeError("The development admin username/email conflicts with existing accounts.")
        existing = existing_users[0]
        if existing.username != DEV_ADMIN_USERNAME or existing.email != DEV_ADMIN_EMAIL:
            raise RuntimeError("The development admin username/email conflicts with an existing account.")
        if existing.role.name != DEV_ADMIN_ROLE or not existing.is_active:
            raise RuntimeError(
                "The existing admin account is inactive or has a different role; it was not modified."
            )
        return existing, False

    role = database.scalar(select(Role).where(Role.name == DEV_ADMIN_ROLE))
    if role is None:
        role = Role(name=DEV_ADMIN_ROLE, description="Local development administrator role")
        database.add(role)
        database.flush()

    user = User(
        username=DEV_ADMIN_USERNAME,
        email=DEV_ADMIN_EMAIL,
        password_hash=hash_password(DEV_ADMIN_PASSWORD),
        role_id=role.id,
        is_active=True,
    )
    database.add(user)
    try:
        database.commit()
        database.refresh(user)
    except Exception:
        database.rollback()
        raise
    return user, True


def reset_dev_admin_password(database: Session) -> User:
    """Explicitly reset only the known local development admin password."""
    user, _created = ensure_dev_admin(database)
    user.password_hash = hash_password(DEV_ADMIN_PASSWORD)
    try:
        database.commit()
        database.refresh(user)
    except Exception:
        database.rollback()
        raise
    return user


def main() -> int:
    parser = argparse.ArgumentParser(description="Create or explicitly reset the local development admin account.")
    parser.add_argument(
        "--reset-password",
        action="store_true",
        help="reset the known local development admin password to its documented value",
    )
    args = parser.parse_args()
    from backend.app.core.config import settings
    from sqlalchemy.engine import make_url

    url = make_url(settings.DATABASE_URL)
    database_path = url.database
    if url.drivername.startswith("sqlite") and database_path and database_path != ":memory:":
        database_path = str(Path(database_path).resolve())
    print(f"LOCAL DEVELOPMENT database: {database_path if database_path else url.drivername}")
    print("The password and stored hash will not be displayed.")
    database = SessionLocal()
    try:
        if args.reset_password:
            user = reset_dev_admin_password(database)
            print(f"LOCAL DEVELOPMENT password reset completed for username={user.username}.")
        else:
            user, created = ensure_dev_admin(database)
            state = "created" if created else "already existed; password unchanged"
            print(f"Development admin {state}: username={user.username}, email={user.email}, role={user.role.name}.")
        return 0
    except Exception as error:
        database.rollback()
        print(f"Could not seed the development admin account: {error}", file=sys.stderr)
        return 1
    finally:
        database.close()


if __name__ == "__main__":
    raise SystemExit(main())
