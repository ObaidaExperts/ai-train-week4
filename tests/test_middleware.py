import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from app.api.middleware import ExceptionHandlerMiddleware

def test_exception_handler_middleware_catches_error():
    app = FastAPI()
    app.add_middleware(ExceptionHandlerMiddleware)

    @app.get("/trigger-error")
    async def trigger_error():
        raise ValueError("Something went wrong")

    client = TestClient(app)
    response = client.get("/trigger-error")

    assert response.status_code == 500
    data = response.json()
    assert data["error"] == "Internal Server Error"
    # In non-debug mode (default for tests here unless set), detail should be generic
    # If app.debug is False, detail is "An unexpected error occurred."
    assert "detail" in data

def test_exception_handler_middleware_debug_mode():
    app = FastAPI(debug=True)
    app.add_middleware(ExceptionHandlerMiddleware)

    @app.get("/trigger-error")
    async def trigger_error():
        raise ValueError("Debugging error")

    client = TestClient(app)
    response = client.get("/trigger-error")

    assert response.status_code == 500
    data = response.json()
    assert data["detail"] == "Debugging error"
def test_exception_handler_middleware_openai_error():
    import openai
    app = FastAPI()
    app.add_middleware(ExceptionHandlerMiddleware)

    @app.get("/trigger-openai-error")
    async def trigger_error():
        # Simulate an OpenAI error with a body
        error = openai.OpenAIError("Original Message Quota Exceeded")
        error.body = {"error": {"message": "Readable Quota Message"}}
        raise error

    client = TestClient(app)
    response = client.get("/trigger-openai-error")

    assert response.status_code == 429
    data = response.json()
    assert data["detail"] == "Readable Quota Message"
