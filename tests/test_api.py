import pytest
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from app.main import app
from app.api.endpoints import get_experiment_service

client = TestClient(app)

def test_read_root() -> None:
    response = client.get("/")
    # root now redirects to dashboard
    assert response.status_code == 200
    assert "AI Experiment Dashboard" in response.text

def test_health_check() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_metadata_endpoint() -> None:
    response = client.get("/metadata")
    assert response.status_code == 200
    data = response.json()
    assert "models" in data
    assert "experiment_types" in data
    assert "claude-sonnet-4-6" in data["models"]
    assert "Stress Test" in data["experiment_types"]

@pytest.fixture
def mock_service():
    service = MagicMock()
    # Default mock behavior
    service.analyze_text.return_value = {
        "response": "Hello! I am an AI.",
        "log_analysis": {
            "model": "gpt-4o",
            "experiment_type": "Baseline",
            "input_tokens": 5,
            "output_tokens": 10,
            "cost_usd": 0.0001,
            "status": "Success"
        }
    }
    return service

def test_chat_endpoint(mock_service: MagicMock) -> None:
    app.dependency_overrides[get_experiment_service] = lambda: mock_service
    try:
        response = client.post("/chat", json={
            "prompt": "Hello", 
            "model": "gpt-4o",
            "experiment_type": "Decoding Strategy",
            "temperature": 0.5,
            "top_p": 0.9,
            "top_k": 40
        })
        assert response.status_code == 200
        data = response.json()
        assert "request_id" in data
        assert data["response"] == "Hello! I am an AI."
        assert mock_service.analyze_text.called
    finally:
        app.dependency_overrides.clear()

def test_chat_endpoint_invalid_model() -> None:
    response = client.post("/chat", json={"prompt": "Hello", "model": "invalid-model"})
    assert response.status_code == 422

def test_chat_endpoint_empty_prompt(mock_service: MagicMock) -> None:
    app.dependency_overrides[get_experiment_service] = lambda: mock_service
    try:
        response = client.post("/chat", json={"prompt": "", "model": "gpt-4o"})
        assert response.status_code in [200, 422]
    finally:
        app.dependency_overrides.clear()

def test_get_results() -> None:
    # Use a real service instance to test the logic (filesystem dependent)
    # Note: This might fail if experiment_results.csv is empty or missing, 
    # but the endpoint handles missing files.
    response = client.get("/results")
    assert response.status_code == 200
    assert isinstance(response.json(), list)
