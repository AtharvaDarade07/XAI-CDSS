"""Environment-backed settings for the backend application."""

from pathlib import Path
import secrets

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATABASE_URL = f"sqlite:///{(PROJECT_ROOT / 'backend' / 'cdss.db').as_posix()}"


class Settings(BaseSettings):
    APP_NAME: str = "XAI Personalized Clinical Decision Support System"
    APP_VERSION: str = "0.1.0"
    DEBUG: bool = False
    API_PREFIX: str = "/api/v1"
    DATABASE_URL: str = DEFAULT_DATABASE_URL
    # A fresh ephemeral key permits local startup without committing a secret.
    # Set CDSS_JWT_SECRET to a stable strong value in every deployed environment.
    JWT_SECRET: SecretStr = Field(
        default_factory=lambda: SecretStr(secrets.token_urlsafe(48))
    )
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

    model_config = SettingsConfigDict(
        env_prefix="CDSS_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
