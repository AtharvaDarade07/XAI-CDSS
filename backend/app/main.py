"""FastAPI entry point for the initial backend foundation."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.api.routes.auth import router as auth_router
from backend.app.api.routes.health import router as health_router
from backend.app.api.routes.explanations import router as explanations_router
from backend.app.api.routes.patients import router as patients_router
from backend.app.api.routes.predictions import router as predictions_router
from backend.app.core.config import settings


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description=(
        "An academic clinical decision-support prototype for research and "
        "education. It is not a clinically validated or deployed system."
    ),
)

# Local Vite origins only; production origins must be configured separately
# when a deployment environment is introduced.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
)

app.include_router(health_router)
app.include_router(auth_router)
app.include_router(patients_router)
app.include_router(predictions_router)
app.include_router(explanations_router)
