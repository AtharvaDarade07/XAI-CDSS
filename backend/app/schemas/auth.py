"""Public authentication request and response schemas."""

from pydantic import BaseModel, ConfigDict, SecretStr


class LoginRequest(BaseModel):
    username: str
    password: SecretStr


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int


class CurrentUserResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    username: str
    email: str
    role: str
    is_active: bool
