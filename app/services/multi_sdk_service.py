"""
Multi-SDK Model Execution — Task 6 + Task 7

Runs the same task across providers with metrics: latency (TTFT, total),
token usage, cost, and quality scoring.
"""

import logging
import time
from typing import Any

from openai import OpenAI
from anthropic import Anthropic

from app.core.config import settings

logger = logging.getLogger("multi_sdk_service")


# ---------------------------------------------------------------------------
# Pricing (USD per 1M tokens) for cost calculation
# ---------------------------------------------------------------------------

MODEL_PRICING: dict[str, dict[str, float]] = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "claude-sonnet-4-6": {"input": 3.00, "output": 15.00},
    "claude-opus-4-6": {"input": 5.00, "output": 25.00},
    "claude-haiku-4-5-20251001": {"input": 1.00, "output": 5.00},
    "gemini-2.0-flash": {"input": 0.075, "output": 0.30},
    "gemini-1.5-flash": {"input": 0.075, "output": 0.30},
    "gemini-1.5-pro": {"input": 3.50, "output": 10.50},
}


def _calc_cost(provider: str, model: str, input_tokens: int, output_tokens: int) -> float:
    """Calculate cost in USD. Local providers return 0."""
    if provider in ("vllm", "llamacpp"):
        return 0.0
    prices = MODEL_PRICING.get(model, {"input": 0.0, "output": 0.0})
    in_cost = (input_tokens / 1_000_000) * prices["input"]
    out_cost = (output_tokens / 1_000_000) * prices["output"]
    return round(in_cost + out_cost, 6)


# ---------------------------------------------------------------------------
# Normalized output schema (Task 7: + ttft_ms, cost_usd)
# ---------------------------------------------------------------------------

def _normalized_result(
    response: str,
    provider: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
    duration_ms: float,
    error: str | None = None,
    ttft_ms: float | None = None,
) -> dict[str, Any]:
    cost_usd = _calc_cost(provider, model, input_tokens, output_tokens)
    return {
        "response": response,
        "provider": provider,
        "model": model,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "ttft_ms": round(ttft_ms, 2) if ttft_ms is not None else None,
        "duration_ms": round(duration_ms, 2),
        "cost_usd": cost_usd,
        "error": error,
    }


# ---------------------------------------------------------------------------
# MultiSDKService
# ---------------------------------------------------------------------------

class MultiSDKService:
    """
    Runs the same task across OpenAI, Anthropic, Gemini, and local (vLLM) providers.
    Outputs are captured consistently.
    """

    def __init__(
        self,
        openai_client: OpenAI | None = None,
        anthropic_client: Anthropic | None = None,
    ) -> None:
        self.openai_client = openai_client or OpenAI(
            api_key=settings.OPENAI_API_KEY
        )
        self.anthropic_client = anthropic_client or Anthropic(
            api_key=settings.ANTHROPIC_API_KEY
        )

    def run(
        self,
        user_request: str,
        provider: str,
        model: str | None = None,
    ) -> dict[str, Any]:
        """
        Run trip planning task on the specified provider.

        provider: "openai" | "anthropic" | "gemini" | "vllm" | "llamacpp"
        model: provider-specific model name (optional, uses default per provider)
        """
        providers = {
            "openai": self._run_openai,
            "anthropic": self._run_anthropic,
            "gemini": self._run_gemini,
            "vllm": self._run_vllm,
            "llamacpp": self._run_llamacpp,
        }
        runner = providers.get(provider)
        if not runner:
            return _normalized_result(
                response="",
                provider=provider,
                model=model or "unknown",
                input_tokens=0,
                output_tokens=0,
                duration_ms=0,
                error=f"Unknown provider: {provider}",
                ttft_ms=None,
            )
        return runner(user_request, model)

    def run_all(
        self,
        user_request: str,
        providers: list[str] | None = None,
    ) -> dict[str, list[dict[str, Any]]]:
        """
        Run the task on multiple providers. Returns results keyed by provider.
        """
        providers = providers or ["openai", "anthropic", "gemini", "vllm", "llamacpp"]
        results: list[dict[str, Any]] = []
        for p in providers:
            try:
                r = self.run(user_request, provider=p)
                results.append(r)
            except Exception as e:
                logger.exception(f"Provider {p} failed")
                results.append(
                    _normalized_result(
                        response="",
                        provider=p,
                        model="",
                        input_tokens=0,
                        output_tokens=0,
                        duration_ms=0,
                        error=str(e),
                        ttft_ms=None,
                    )
                )
        return {"results": results}

    def _run_openai(
        self, prompt: str, model: str | None
    ) -> dict[str, Any]:
        model = model or "gpt-4o-mini"
        start = time.perf_counter()
        ttft_ms: float | None = None
        try:
            stream = self.openai_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                stream=True,
                stream_options={"include_usage": True},
            )
            content_parts: list[str] = []
            input_tokens = 0
            output_tokens = 0
            for chunk in stream:
                if ttft_ms is None:
                    ttft_ms = (time.perf_counter() - start) * 1000
                if chunk.choices:
                    delta = chunk.choices[0].delta
                    if delta and getattr(delta, "content", None):
                        content_parts.append(delta.content)
                if hasattr(chunk, "usage") and chunk.usage:
                    input_tokens = getattr(chunk.usage, "prompt_tokens", 0) or 0
                    output_tokens = getattr(chunk.usage, "completion_tokens", 0) or 0
            content = "".join(content_parts)
            duration_ms = (time.perf_counter() - start) * 1000
            if input_tokens == 0 and output_tokens == 0:
                from app.core.tokenizer import TokenCounter
                input_tokens = TokenCounter.count_openai_tokens(prompt, model)
                output_tokens = TokenCounter.count_openai_tokens(content, model)
        except Exception as e:
            duration_ms = (time.perf_counter() - start) * 1000
            return _normalized_result(
                response="",
                provider="openai",
                model=model,
                input_tokens=0,
                output_tokens=0,
                duration_ms=duration_ms,
                error=str(e),
                ttft_ms=None,
            )
        logger.info(f"[MultiSDK] OpenAI done: {output_tokens} tokens, TTFT={ttft_ms}ms")
        return _normalized_result(
            response=content,
            provider="openai",
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            duration_ms=duration_ms,
            ttft_ms=ttft_ms,
        )

    def _run_anthropic(
        self, prompt: str, model: str | None
    ) -> dict[str, Any]:
        model = model or "claude-sonnet-4-6"
        start = time.perf_counter()
        try:
            message = self.anthropic_client.messages.create(
                model=model,
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
            )
            content = (
                getattr(message.content[0], "text", "")
                if message.content
                else ""
            )
            input_tokens = message.usage.input_tokens
            output_tokens = message.usage.output_tokens
        except Exception as e:
            duration_ms = (time.perf_counter() - start) * 1000
            return _normalized_result(
                response="",
                provider="anthropic",
                model=model,
                input_tokens=0,
                output_tokens=0,
                duration_ms=duration_ms,
                error=str(e),
                ttft_ms=None,
            )
        duration_ms = (time.perf_counter() - start) * 1000
        logger.info(f"[MultiSDK] Anthropic done: {output_tokens} tokens")
        return _normalized_result(
            response=content,
            provider="anthropic",
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            duration_ms=duration_ms,
            ttft_ms=None,
        )

    def _run_gemini(
        self, prompt: str, model: str | None
    ) -> dict[str, Any]:
        model = model or "gemini-2.0-flash"
        start = time.perf_counter()
        if not settings.GOOGLE_API_KEY:
            duration_ms = (time.perf_counter() - start) * 1000
            return _normalized_result(
                response="",
                provider="gemini",
                model=model,
                input_tokens=0,
                output_tokens=0,
                duration_ms=duration_ms,
                error="GOOGLE_API_KEY not configured",
                ttft_ms=None,
            )
        try:
            import google.generativeai as genai

            genai.configure(api_key=settings.GOOGLE_API_KEY)
            gemini_model = genai.GenerativeModel(
                model if "/" in model else f"models/{model}"
            )
            result = gemini_model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.7,
                    max_output_tokens=1024,
                ),
            )
            content = result.text if result.text else ""
            um = result.usage_metadata
            input_tokens = (
                getattr(um, "prompt_token_count", 0) or 0
                if um
                else 0
            )
            output_tokens = (
                getattr(um, "candidates_token_count", 0) or 0
                if um
                else 0
            )
        except Exception as e:
            duration_ms = (time.perf_counter() - start) * 1000
            return _normalized_result(
                response="",
                provider="gemini",
                model=model,
                input_tokens=0,
                output_tokens=0,
                duration_ms=duration_ms,
                error=str(e),
                ttft_ms=None,
            )
        duration_ms = (time.perf_counter() - start) * 1000
        logger.info(f"[MultiSDK] Gemini done: {output_tokens} tokens")
        return _normalized_result(
            response=content,
            provider="gemini",
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            duration_ms=duration_ms,
            ttft_ms=None,
        )

    def _run_vllm(
        self, prompt: str, model: str | None
    ) -> dict[str, Any]:
        model = model or "default"
        start = time.perf_counter()
        if not settings.VLLM_BASE_URL:
            duration_ms = (time.perf_counter() - start) * 1000
            return _normalized_result(
                response="",
                provider="vllm",
                model=model,
                input_tokens=0,
                output_tokens=0,
                duration_ms=duration_ms,
                error="VLLM_BASE_URL not configured",
                ttft_ms=None,
            )
        try:
            vllm_client = OpenAI(
                base_url=settings.VLLM_BASE_URL.rstrip("/"),
                api_key="token-placeholder",
            )
            vllm_model = model if model and model != "default" else settings.VLLM_MODEL
            response = vllm_client.chat.completions.create(
                model=vllm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
            )
            content = response.choices[0].message.content or ""
            usage = response.usage
            input_tokens = usage.prompt_tokens if usage else 0
            output_tokens = usage.completion_tokens if usage else 0
        except Exception as e:
            duration_ms = (time.perf_counter() - start) * 1000
            return _normalized_result(
                response="",
                provider="vllm",
                model=model,
                input_tokens=0,
                output_tokens=0,
                duration_ms=duration_ms,
                error=str(e),
                ttft_ms=None,
            )
        duration_ms = (time.perf_counter() - start) * 1000
        logger.info(f"[MultiSDK] vLLM done: {output_tokens} tokens")
        return _normalized_result(
            response=content,
            provider="vllm",
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            duration_ms=duration_ms,
            ttft_ms=None,
        )

    def _run_llamacpp(
        self, prompt: str, model: str | None
    ) -> dict[str, Any]:
        model = model or "default"
        start = time.perf_counter()
        if not settings.LLAMA_CPP_BASE_URL:
            duration_ms = (time.perf_counter() - start) * 1000
            return _normalized_result(
                response="",
                provider="llamacpp",
                model=model,
                input_tokens=0,
                output_tokens=0,
                duration_ms=duration_ms,
                error="LLAMA_CPP_BASE_URL not configured",
                ttft_ms=None,
            )
        try:
            client = OpenAI(
                base_url=settings.LLAMA_CPP_BASE_URL.rstrip("/"),
                api_key="token-placeholder",
            )
            model_name = model if model and model != "default" else "default"
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
            )
            content = response.choices[0].message.content or ""
            usage = response.usage
            input_tokens = usage.prompt_tokens if usage else 0
            output_tokens = usage.completion_tokens if usage else 0
        except Exception as e:
            duration_ms = (time.perf_counter() - start) * 1000
            return _normalized_result(
                response="",
                provider="llamacpp",
                model=model,
                input_tokens=0,
                output_tokens=0,
                duration_ms=duration_ms,
                error=str(e),
                ttft_ms=None,
            )
        duration_ms = (time.perf_counter() - start) * 1000
        logger.info(f"[MultiSDK] Llama.cpp done: {output_tokens} tokens")
        return _normalized_result(
            response=content,
            provider="llamacpp",
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            duration_ms=duration_ms,
            ttft_ms=None,
        )
