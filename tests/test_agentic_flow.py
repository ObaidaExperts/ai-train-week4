"""Unit tests for Single vs Agentic Flow (trip planning)."""
import json
import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

from app.main import app
from app.services.single_prompt_service import SinglePromptService, TRIP_PLANNING_PROMPT
from app.services.agentic_service import AgenticService

client = TestClient(app)


# ── Single-prompt service tests ─────────────────────────────────────────────
class TestSinglePromptService:
    def test_prompt_template_includes_user_request(self):
        req = "Plan a 3-day trip to Paris with $500 budget"
        prompt = TRIP_PLANNING_PROMPT.format(user_request=req)
        assert "Paris" in prompt
        assert "$500" in prompt

    @patch("app.services.single_prompt_service.OpenAI")
    def test_run_returns_expected_structure(self, mock_openai_class):
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Day 1: Visit Louvre..."
        mock_response.usage = MagicMock()
        mock_response.usage.prompt_tokens = 100
        mock_response.usage.completion_tokens = 200

        mock_client.chat.completions.create.return_value = mock_response

        service = SinglePromptService(openai_client=mock_client)
        result = service.run(user_request="Trip to Paris", model="gpt-4o")

        assert result["response"] == "Day 1: Visit Louvre..."
        assert result["input_tokens"] == 100
        assert result["output_tokens"] == 200
        assert result["approach"] == "single_prompt"
        assert result["model"] == "gpt-4o"

    @patch("app.services.single_prompt_service.Anthropic")
    def test_run_uses_claude_when_model_starts_with_claude(self, mock_anthropic_class):
        mock_client = MagicMock()
        mock_anthropic_class.return_value = mock_client

        mock_message = MagicMock()
        mock_message.content = [MagicMock(text="Claude trip plan")]
        mock_message.usage = MagicMock(input_tokens=50, output_tokens=80)

        mock_client.messages.create.return_value = mock_message

        service = SinglePromptService(anthropic_client=mock_client)
        result = service.run(user_request="Paris 3 days", model="claude-sonnet-4-6")

        assert result["response"] == "Claude trip plan"
        assert result["input_tokens"] == 50
        assert result["output_tokens"] == 80
        mock_client.messages.create.assert_called_once()

    @patch("app.services.single_prompt_service.OpenAI")
    def test_run_passes_user_request_to_llm(self, mock_openai_class):
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Plan"
        mock_response.usage = MagicMock(prompt_tokens=10, completion_tokens=5)

        mock_client.chat.completions.create.return_value = mock_response

        service = SinglePromptService(openai_client=mock_client)
        service.run(user_request="3 days in Tokyo, $300")

        call_args = mock_client.chat.completions.create.call_args
        messages = call_args.kwargs["messages"]
        assert any("Tokyo" in str(m.get("content", "")) for m in messages)
        assert any("300" in str(m.get("content", "")) for m in messages)


# ── Agentic service tests ──────────────────────────────────────────────────
class TestAgenticService:
    def test_parse_plan_extracts_json(self):
        service = AgenticService()
        text = 'Here is the plan: {"steps": [{"id": 1, "action": "get_weather"}]}'
        plan = service._parse_plan(text)
        assert "steps" in plan
        assert len(plan["steps"]) == 1
        assert plan["steps"][0]["action"] == "get_weather"

    def test_parse_plan_fallback_on_invalid_json(self):
        service = AgenticService()
        plan = service._parse_plan("No JSON here at all")
        assert "steps" in plan
        assert plan["steps"][0]["action"] == "synthesize"

    @patch("app.services.agentic_service.OpenAI")
    def test_run_planning_phase_called(self, mock_openai_class):
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client

        # Planner returns a plan
        plan_response = MagicMock()
        plan_response.choices = [MagicMock()]
        plan_response.choices[0].message.content = '{"steps": [{"id": 1, "action": "get_weather", "reason": "x"}, {"id": 2, "action": "synthesize", "reason": "y"}]}'
        plan_response.usage = MagicMock(prompt_tokens=50, completion_tokens=30)

        # Executor returns final answer (no tool calls)
        exec_response = MagicMock()
        exec_response.choices = [MagicMock()]
        exec_response.choices[0].finish_reason = "stop"
        exec_response.choices[0].message.content = "Final trip plan"
        exec_response.choices[0].message.tool_calls = None
        exec_response.usage = MagicMock(prompt_tokens=100, completion_tokens=150)

        mock_client.chat.completions.create.side_effect = [plan_response, exec_response]

        service = AgenticService(openai_client=mock_client)
        result = service.run(user_request="Paris 3 days $500", model="gpt-4o")

        assert result["response"] == "Final trip plan"
        assert result["approach"] == "agentic"
        assert "plan" in result
        assert "steps" in result
        assert result["plan"]["steps"][0]["action"] == "get_weather"

    @patch("app.services.agentic_service.OpenAI")
    def test_run_includes_tool_calls_in_trace(self, mock_openai_class):
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client

        plan_response = MagicMock()
        plan_response.choices = [MagicMock()]
        plan_response.choices[0].message.content = '{"steps": [{"id": 1, "action": "get_weather"}]}'
        plan_response.usage = MagicMock(prompt_tokens=50, completion_tokens=30)

        # First exec call: tool call
        tool_call = MagicMock()
        tool_call.id = "call_123"
        tool_call.function.name = "get_weather"
        tool_call.function.arguments = '{"location": "Paris"}'

        exec_response_1 = MagicMock()
        exec_response_1.choices = [MagicMock()]
        exec_response_1.choices[0].finish_reason = "tool_calls"
        exec_response_1.choices[0].message.content = None
        exec_response_1.choices[0].message.tool_calls = [tool_call]
        exec_response_1.usage = MagicMock(prompt_tokens=100, completion_tokens=20)

        # Second exec call: final answer
        exec_response_2 = MagicMock()
        exec_response_2.choices = [MagicMock()]
        exec_response_2.choices[0].finish_reason = "stop"
        exec_response_2.choices[0].message.content = "Here is your plan"
        exec_response_2.choices[0].message.tool_calls = None
        exec_response_2.usage = MagicMock(prompt_tokens=150, completion_tokens=80)

        mock_client.chat.completions.create.side_effect = [
            plan_response,
            exec_response_1,
            exec_response_2,
        ]

        service = AgenticService(openai_client=mock_client)
        result = service.run(user_request="Paris", model="gpt-4o")

        assert result["response"] == "Here is your plan"
        tool_steps = [s for s in result["steps"] if s.get("phase") == "tool_call"]
        assert len(tool_steps) >= 1
        assert tool_steps[0]["tool"] == "get_weather"


# ── API endpoint tests ────────────────────────────────────────────────────
class TestAgenticFlowEndpoints:
    @patch("app.services.single_prompt_service.OpenAI")
    def test_agentic_flow_single_endpoint(self, mock_openai_class):
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Trip plan"
        mock_response.usage = MagicMock(prompt_tokens=50, completion_tokens=100)

        mock_client.chat.completions.create.return_value = mock_response

        resp = client.post(
            "/agentic-flow/single",
            json={"user_request": "3 days in Rome, $400"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["response"] == "Trip plan"
        assert data["approach"] == "single_prompt"

    @patch("app.services.agentic_service.OpenAI")
    def test_agentic_flow_agentic_endpoint(self, mock_openai_class):
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client

        plan_resp = MagicMock()
        plan_resp.choices = [MagicMock()]
        plan_resp.choices[0].message.content = '{"steps": [{"id": 1, "action": "synthesize"}]}'
        plan_resp.usage = MagicMock(prompt_tokens=50, completion_tokens=30)

        exec_resp = MagicMock()
        exec_resp.choices = [MagicMock()]
        exec_resp.choices[0].finish_reason = "stop"
        exec_resp.choices[0].message.content = "Agentic plan"
        exec_resp.choices[0].message.tool_calls = None
        exec_resp.usage = MagicMock(prompt_tokens=100, completion_tokens=80)

        mock_client.chat.completions.create.side_effect = [plan_resp, exec_resp]

        resp = client.post(
            "/agentic-flow/agentic",
            json={"user_request": "Paris 3 days"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["response"] == "Agentic plan"
        assert data["approach"] == "agentic"
        assert "plan" in data
        assert "steps" in data
