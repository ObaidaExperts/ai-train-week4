"""Unit tests for ToolCallingService (Task 4)."""
import json
import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

from app.main import app
from app.services.tool_service import (
    ToolCallingService,
    ToolArgumentError,
    _validate_and_execute_tool,
    TOOLS,
)

client = TestClient(app)


# ── Tool schema tests ─────────────────────────────────────────────────────
class TestToolSchemas:
    def test_all_tools_present(self):
        names = [t["function"]["name"] for t in TOOLS]
        assert "get_weather" in names
        assert "calculate" in names
        assert "get_stock_price" in names

    def test_tools_have_required_fields(self):
        for tool in TOOLS:
            assert tool["type"] == "function"
            fn = tool["function"]
            assert "name" in fn
            assert "description" in fn
            assert "parameters" in fn
            assert "required" in fn["parameters"]


# ── Argument validation tests ─────────────────────────────────────────────
class TestArgumentValidation:

    def test_get_weather_valid(self):
        result = json.loads(_validate_and_execute_tool("get_weather", '{"location": "Paris", "unit": "celsius"}'))
        assert result["location"] == "Paris"
        assert result["unit"] == "celsius"
        assert isinstance(result["temperature"], (int, float))

    def test_get_weather_missing_location(self):
        with pytest.raises(ToolArgumentError, match="location"):
            _validate_and_execute_tool("get_weather", '{"unit": "celsius"}')

    def test_get_weather_invalid_unit(self):
        with pytest.raises(ToolArgumentError, match="unit"):
            _validate_and_execute_tool("get_weather", '{"location": "Paris", "unit": "kelvin"}')

    def test_calculate_valid(self):
        result = json.loads(_validate_and_execute_tool("calculate", '{"expression": "2 + 2"}'))
        assert result["result"] == 4

    def test_calculate_sqrt(self):
        result = json.loads(_validate_and_execute_tool("calculate", '{"expression": "sqrt(144)"}'))
        assert result["result"] == 12.0

    def test_calculate_missing_expression(self):
        with pytest.raises(ToolArgumentError, match="expression"):
            _validate_and_execute_tool("calculate", '{}')

    def test_calculate_invalid_expression(self):
        with pytest.raises(ToolArgumentError):
            _validate_and_execute_tool("calculate", '{"expression": "import os"}')

    def test_get_stock_price_valid(self):
        result = json.loads(_validate_and_execute_tool("get_stock_price", '{"ticker": "AAPL"}'))
        assert result["ticker"] == "AAPL"
        assert "price_usd" in result

    def test_get_stock_price_missing_ticker(self):
        with pytest.raises(ToolArgumentError, match="ticker"):
            _validate_and_execute_tool("get_stock_price", '{}')

    def test_malformed_json_args(self):
        with pytest.raises(ToolArgumentError, match="malformed JSON"):
            _validate_and_execute_tool("get_weather", "not-valid-json")

    def test_unknown_tool(self):
        with pytest.raises(ToolArgumentError, match="Unknown tool"):
            _validate_and_execute_tool("non_existent_tool", '{}')


# ── ToolCallingService loop tests ─────────────────────────────────────────
class TestToolCallingService:

    def _make_direct_response(self, text: str) -> MagicMock:
        """Build a mock OpenAI response that answers directly (no tool call)."""
        msg = MagicMock()
        msg.content = text
        msg.tool_calls = None
        choice = MagicMock()
        choice.finish_reason = "stop"
        choice.message = msg
        resp = MagicMock()
        resp.choices = [choice]
        return resp

    def _make_tool_call_response(self, tool_name: str, args: dict) -> MagicMock:
        """Build a mock that triggers a tool call."""
        tool_call = MagicMock()
        tool_call.id = "call_abc123"
        tool_call.function.name = tool_name
        tool_call.function.arguments = json.dumps(args)
        msg = MagicMock()
        msg.tool_calls = [tool_call]
        choice = MagicMock()
        choice.finish_reason = "tool_calls"
        choice.message = msg
        resp = MagicMock()
        resp.choices = [choice]
        return resp

    def test_direct_answer_no_tool(self):
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = self._make_direct_response("The answer is 42.")
        service = ToolCallingService(openai_client=mock_client)
        result = service.run_tool_loop("What is the meaning of life?")
        assert result["final_response"] == "The answer is 42."
        assert result["tool_called"] is None
        assert any(s["step"] == "model_direct_answer" for s in result["steps"])

    def test_successful_tool_call(self):
        mock_client = MagicMock()
        # First call → trigger tool; second call → final answer
        mock_client.chat.completions.create.side_effect = [
            self._make_tool_call_response("get_weather", {"location": "Tokyo", "unit": "celsius"}),
            self._make_direct_response("The weather in Tokyo is 18°C and partly cloudy."),
        ]
        service = ToolCallingService(openai_client=mock_client)
        result = service.run_tool_loop("What is the weather in Tokyo?")
        assert result["tool_called"] == "get_weather"
        assert result["tool_error"] is None
        assert result["tool_result"]["location"] == "Tokyo"
        assert "final_response" in result

    def test_tool_call_with_invalid_args(self):
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = [
            self._make_tool_call_response("get_weather", {"unit": "kelvin"}),  # missing location + bad unit
            self._make_direct_response("I could not retrieve weather due to an error."),
        ]
        service = ToolCallingService(openai_client=mock_client)
        result = service.run_tool_loop("Weather?")
        # The error is caught and returned back to the model
        assert result["tool_error"] is not None
        assert "error" in result["tool_result"]
        assert any(s["step"] == "tool_error" for s in result["steps"])

    def test_enabled_tools_filter(self):
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = self._make_direct_response("Noted.")
        service = ToolCallingService(openai_client=mock_client)
        service.run_tool_loop("x", enabled_tools=["calculate"])
        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        passed_tools = call_kwargs.get("tools", [])
        assert len(passed_tools) == 1
        assert passed_tools[0]["function"]["name"] == "calculate"


# ── API endpoint tests ────────────────────────────────────────────────────
class TestToolCallEndpoint:

    def test_tool_schemas_endpoint(self):
        res = client.get("/tools/schemas")
        assert res.status_code == 200
        data = res.json()
        assert "tools" in data
        assert data["count"] == 3

    def test_tool_call_force_error(self):
        res = client.post("/tool-call", json={
            "prompt": "test",
            "force_error": True
        })
        assert res.status_code == 200
        data = res.json()
        assert data["tool_error"] is not None
        assert data["tool_called"] == "get_weather"
        assert any(s["step"] == "tool_error" for s in data["steps"])

    def test_tool_call_invalid_payload(self):
        res = client.post("/tool-call", json={})
        assert res.status_code == 422  # missing required 'prompt'
