"""Reusable password hashing and JWT helpers."""

from datetime import datetime, timedelta, timezone
from typing import Any

from jose import JWTError, jwt
from passlib.context import CryptContext

from backend.app.core.config import settings


password_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(password: str) -> str:
    """Return a bcrypt hash suitable for storage; never store the input."""
    if len(password.encode("utf-8")) > 72:
        raise ValueError("bcrypt passwords must not exceed 72 UTF-8 bytes")
    return password_context.hash(password)


def verify_password(plain_password: str, password_hash: str) -> bool:
    """Verify a password against a stored bcrypt hash safely."""
    if len(plain_password.encode("utf-8")) > 72:
        return False
    try:
        return password_context.verify(plain_password, password_hash)
    except (ValueError, TypeError):
        return False


def create_access_token(
    subject: str | int,
    expires_delta: timedelta | None = None,
    claims: dict[str, Any] | None = None,
) -> str:
    """Create a signed JWT with a subject and expiration claim."""
    now = datetime.now(timezone.utc)
    expires_at = now + (expires_delta or timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES))
    payload: dict[str, Any] = {"sub": str(subject), "iat": now, "exp": expires_at}
    if claims:
        # Reserved claims stay controlled by this helper.
        payload.update({key: value for key, value in claims.items() if key not in {"sub", "iat", "exp"}})
    return jwt.encode(
        payload,
        settings.JWT_SECRET.get_secret_value(),
        algorithm=settings.JWT_ALGORITHM,
    )


def decode_access_token(token: str) -> dict[str, Any]:
    """Decode and validate a signed, non-expired access token.

    Raises jose.jwt.JWTError for malformed, invalid-signature, or expired tokens.
    """
    return jwt.decode(
        token,
        settings.JWT_SECRET.get_secret_value(),
        algorithms=[settings.JWT_ALGORITHM],
    )


__all__ = ["JWTError", "create_access_token", "decode_access_token", "hash_password", "verify_password"]
