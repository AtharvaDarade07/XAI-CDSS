"""Fast, service-independent tests for the initial health API."""

from fastapi.testclient import TestClient

from backend.app.main import app


client = TestClient(app)


def test_root_returns_service_identity_and_running_status():
    response = client.get("/")

    assert response.status_code == 200
    body = response.json()
    assert body["name"] == "XAI Personalized Clinical Decision Support System"
    assert body["version"] == "0.1.0"
    assert body["status"] == "running"


def test_health_returns_healthy_backend_status():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "healthy", "service": "backend"}
