"""Unit tests for Multi-SDK Model Execution (Task 6)."""
import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

from app.main import app
from app.services.multi_sdk_service import MultiSDKService

client = TestClient(app)


# ── Structure tests ────────────────────────────────────────────────────────
class TestMultiSDKService:
    def _make_openai_stream(self, content: str, input_tokens: int = 100, output_tokens: int = 200):
        """Build mock streaming response chunks."""
        chunk1 = MagicMock()
        chunk1.choices = [MagicMock()]
        chunk1.choices[0].delta = MagicMock(content=content)
        chunk1.usage = None
        chunk2 = MagicMock()
        chunk2.choices = []
        chunk2.usage = MagicMock(prompt_tokens=input_tokens, completion_tokens=output_tokens)
        return iter([chunk1, chunk2])

    @patch("app.services.multi_sdk_service.OpenAI")
    def test_run_passes_user_request_to_model_as_is(self, mock_openai_class):
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client

        mock_client.chat.completions.create.return_value = self._make_openai_stream("Plan", 10, 5)

        service = MultiSDKService(openai_client=mock_client)
        service.run(user_request="Plan a 3-day trip to Paris with $500", provider="openai")

        call_args = mock_client.chat.completions.create.call_args
        messages = call_args.kwargs["messages"]
        assert messages[0]["content"] == "Plan a 3-day trip to Paris with $500"

    @patch("app.services.multi_sdk_service.OpenAI")
    def test_run_openai_returns_expected_structure(self, mock_openai_class):
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client

        mock_client.chat.completions.create.return_value = self._make_openai_stream(
            "Day 1: Visit Louvre...", 100, 200
        )

        service = MultiSDKService(openai_client=mock_client)
        result = service.run(user_request="Trip to Paris", provider="openai")

        assert result["response"] == "Day 1: Visit Louvre..."
        assert result["provider"] == "openai"
        assert result["model"] == "gpt-4o-mini"
        assert result["input_tokens"] == 100
        assert result["output_tokens"] == 200
        assert "duration_ms" in result
        assert result["ttft_ms"] is not None
        assert result["cost_usd"] is not None
        assert result["error"] is None

    @patch("app.services.multi_sdk_service.Anthropic")
    def test_run_anthropic_returns_expected_structure(self, mock_anthropic_class):
        mock_client = MagicMock()
        mock_anthropic_class.return_value = mock_client

        mock_message = MagicMock()
        mock_message.content = [MagicMock(text="Claude trip plan")]
        mock_message.usage = MagicMock(input_tokens=50, output_tokens=80)

        mock_client.messages.create.return_value = mock_message

        service = MultiSDKService(anthropic_client=mock_client)
        result = service.run(user_request="Paris 3 days", provider="anthropic")

        assert result["response"] == "Claude trip plan"
        assert result["provider"] == "anthropic"
        assert result["input_tokens"] == 50
        assert result["output_tokens"] == 80
        assert "duration_ms" in result

    @patch("google.generativeai.GenerativeModel")
    @patch("google.generativeai.configure")
    def test_run_gemini_returns_expected_structure(self, mock_configure, mock_model_class):
        mock_response = MagicMock()
        mock_response.text = "Gemini trip plan"
        mock_response.usage_metadata = MagicMock()
        mock_response.usage_metadata.prompt_token_count = 60
        mock_response.usage_metadata.candidates_token_count = 90

        mock_model = MagicMock()
        mock_model.generate_content.return_value = mock_response
        mock_model_class.return_value = mock_model

        with patch("app.services.multi_sdk_service.settings") as mock_settings:
            mock_settings.GOOGLE_API_KEY = "test-key"

            service = MultiSDKService()
            result = service.run(user_request="Tokyo 2 days", provider="gemini")

        assert result["response"] == "Gemini trip plan"
        assert result["provider"] == "gemini"
        assert result["input_tokens"] == 60
        assert result["output_tokens"] == 90

    def test_run_unknown_provider_returns_error(self):
        service = MultiSDKService()
        result = service.run(user_request="Tokyo", provider="unknown_provider")

        assert result["provider"] == "unknown_provider"
        assert result["error"] is not None
        assert "Unknown provider" in result["error"]
        assert result["response"] == ""

    @patch("app.services.multi_sdk_service.settings")
    def test_run_vllm_without_config_returns_error(self, mock_settings):
        mock_settings.VLLM_BASE_URL = None
        mock_settings.VLLM_MODEL = "default"

        service = MultiSDKService()
        result = service.run(user_request="Tokyo", provider="vllm")

        assert result["provider"] == "vllm"
        assert result["error"] is not None
        assert "VLLM_BASE_URL" in result["error"]

    @patch("app.services.multi_sdk_service.settings")
    def test_run_llamacpp_without_config_returns_error(self, mock_settings):
        mock_settings.LLAMA_CPP_BASE_URL = None

        service = MultiSDKService()
        result = service.run(user_request="Tokyo", provider="llamacpp")

        assert result["provider"] == "llamacpp"
        assert result["error"] is not None
        assert "LLAMA_CPP_BASE_URL" in result["error"]

    @patch("app.services.multi_sdk_service.OpenAI")
    @patch("app.services.multi_sdk_service.settings")
    def test_run_all_returns_results_for_each_provider(
        self, mock_settings, mock_openai_class
    ):
        mock_settings.VLLM_BASE_URL = None

        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Plan"
        mock_response.usage = MagicMock(prompt_tokens=10, completion_tokens=5)

        mock_client.chat.completions.create.return_value = mock_response

        service = MultiSDKService(openai_client=mock_client)
        result = service.run_all(user_request="Paris", providers=["openai"])

        assert "results" in result
        assert len(result["results"]) == 1
        assert result["results"][0]["provider"] == "openai"


# ── API endpoint tests ─────────────────────────────────────────────────────
class TestMultiSDKAPI:
    @patch("app.services.multi_sdk_service.OpenAI")
    def test_multi_sdk_run_endpoint(self, mock_openai_class):
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client

        def make_stream(content, in_tok, out_tok):
            c1 = MagicMock()
            c1.choices = [MagicMock()]
            c1.choices[0].delta = MagicMock(content=content)
            c1.usage = None
            c2 = MagicMock()
            c2.choices = []
            c2.usage = MagicMock(prompt_tokens=in_tok, completion_tokens=out_tok)
            return iter([c1, c2])

        mock_client.chat.completions.create.return_value = make_stream("API plan", 20, 30)

        res = client.post(
            "/multi-sdk/run",
            json={"user_request": "Paris 3 days", "provider": "openai"},
        )

        assert res.status_code == 200
        data = res.json()
        assert data["provider"] == "openai"
        assert data["response"] == "API plan"
        assert "input_tokens" in data
        assert "output_tokens" in data
        assert "duration_ms" in data
        assert "cost_usd" in data
        assert "ttft_ms" in data

    def test_multi_sdk_run_requires_provider(self):
        res = client.post(
            "/multi-sdk/run",
            json={"user_request": "Paris"},
        )
        assert res.status_code == 422

    def test_metadata_includes_multi_sdk_providers(self):
        res = client.get("/metadata")
        assert res.status_code == 200
        data = res.json()
        assert "multi_sdk_providers" in data
        assert "openai" in data["multi_sdk_providers"]
        assert "anthropic" in data["multi_sdk_providers"]
        assert "gemini" in data["multi_sdk_providers"]
        assert "vllm" in data["multi_sdk_providers"]
        assert "llamacpp" in data["multi_sdk_providers"]

    @patch("app.services.multi_sdk_service.OpenAI")
    @patch("app.services.multi_sdk_service.settings")
    def test_multi_sdk_run_all_endpoint(self, mock_settings, mock_openai_class):
        mock_settings.VLLM_BASE_URL = None

        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client

        def make_stream(content, in_tok, out_tok):
            c1 = MagicMock()
            c1.choices = [MagicMock()]
            c1.choices[0].delta = MagicMock(content=content)
            c1.usage = None
            c2 = MagicMock()
            c2.choices = []
            c2.usage = MagicMock(prompt_tokens=in_tok, completion_tokens=out_tok)
            return iter([c1, c2])

        mock_client.chat.completions.create.return_value = make_stream("Plan", 10, 5)

        res = client.post(
            "/multi-sdk/run-all",
            json={"user_request": "Paris 3 days", "providers": ["openai"]},
        )

        assert res.status_code == 200
        data = res.json()
        assert "results" in data
        assert len(data["results"]) == 1
        assert data["results"][0]["provider"] == "openai"
        assert "cost_usd" in data["results"][0]
