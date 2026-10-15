"""Root and health endpoints for service checks."""

from fastapi import APIRouter

from backend.app.core.config import settings
from backend.app.schemas.health import HealthResponse, RootResponse


router = APIRouter()


@router.get("/", response_model=RootResponse, tags=["health"])
def root() -> RootResponse:
    """Return basic service identity and runtime status."""
    return RootResponse(
        name=settings.APP_NAME,
        version=settings.APP_VERSION,
        status="running",
    )


@router.get("/health", response_model=HealthResponse, tags=["health"])
def health() -> HealthResponse:
    """Return a lightweight liveness response."""
    return HealthResponse(status="healthy", service="backend")
