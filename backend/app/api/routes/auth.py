"""Login and minimal authenticated/RBAC verification endpoints."""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.api.dependencies import get_current_user, get_db, require_roles
from backend.app.core.config import settings
from backend.app.core.security import create_access_token, verify_password
from backend.app.db.models import User
from backend.app.schemas.auth import CurrentUserResponse, LoginRequest, TokenResponse


router = APIRouter(prefix="/auth", tags=["authentication"])


@router.post("/login", response_model=TokenResponse)
def login(credentials: LoginRequest, database: Session = Depends(get_db)) -> TokenResponse:
    """Authenticate by username/password and return a short-lived bearer token."""
    user = database.scalar(select(User).where(User.username == credentials.username))
    if user is None or not verify_password(
        credentials.password.get_secret_value(),
        user.password_hash if user is not None else "",
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Inactive user account",
        )

    return TokenResponse(
        access_token=create_access_token(subject=user.id),
        expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
    )


@router.get("/me", response_model=CurrentUserResponse)
def read_current_user(user: User = Depends(get_current_user)) -> CurrentUserResponse:
    """Return safe profile details for the authenticated user."""
    return CurrentUserResponse(
        id=user.id,
        username=user.username,
        email=user.email,
        role=user.role.name,
        is_active=user.is_active,
    )


@router.get("/rbac/admin-check")
def admin_rbac_check(user: User = Depends(require_roles("admin"))) -> dict[str, str]:
    """Minimal admin-only endpoint for authentication/RBAC verification."""
    return {"status": "authorized", "role": user.role.name}
