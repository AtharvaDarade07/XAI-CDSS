"""Response schemas for root and health endpoints."""

from pydantic import BaseModel


class RootResponse(BaseModel):
    name: str
    version: str
    status: str


class HealthResponse(BaseModel):
    status: str
    service: str
