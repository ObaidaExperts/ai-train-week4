from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

def test_read_root() -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to the Tokenization Chat Analysis API"}

def test_health_check() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

@patch("app.api.endpoints.get_experiment_service")
def test_chat_endpoint(mock_get_service: MagicMock) -> None:
    # Setup mock service
    mock_service = MagicMock()
    mock_get_service.return_value = mock_service
    mock_service.analyze_text.return_value = {
        "response": "Hello! I am an AI.",
        "log_analysis": {
            "model": "gpt-4o",
            "input_tokens": 5,
            "output_tokens": 10,
            "cost_usd": 0.0001,
            "status": "Success"
        }
    }

    response = client.post("/chat", json={"prompt": "Hello"})
    assert response.status_code == 200
    data = response.json()
    assert "request_id" in data
    assert data["response"] == "Hello! I am an AI."
    assert "log_analysis" in data

def test_get_results() -> None:
    # Use a real service instance to test the logic (filesystem dependent)
    # Note: This might fail if experiment_results.csv is empty or missing, 
    # but the endpoint handles missing files.
    response = client.get("/results")
    assert response.status_code == 200
    assert isinstance(response.json(), list)
