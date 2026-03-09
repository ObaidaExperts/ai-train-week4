"""
Single-Prompt Service — Trip planning

Implements a one-shot LLM solution for trip planning.
One prompt, one API call, one response. No planning, no tool use, no agent loop.
Supports both OpenAI and Anthropic (Claude) models.
"""

import logging
from typing import Any

from openai import OpenAI
from anthropic import Anthropic

from app.core.config import settings

logger = logging.getLogger("single_prompt_service")


def _is_anthropic_model(model: str) -> bool:
    """Return True if the model is an Anthropic/Claude model."""
    return model.startswith("claude")


# ---------------------------------------------------------------------------
# Single-prompt task: Trip planning
# ---------------------------------------------------------------------------

TRIP_PLANNING_PROMPT = """You are a travel planner. Create a detailed trip plan based on the user's request.

User request: {user_request}

Provide a complete plan including:
1. Overview and highlights
2. Day-by-day itinerary with activities
3. Estimated costs breakdown (accommodation, food, activities, transport)
4. Practical tips (weather considerations, packing, local customs)
5. Budget summary and any money-saving suggestions

Be specific and actionable. Format clearly with headers and bullet points."""


class SinglePromptService:
    """
    Executes a trip-planning task with a single LLM call.
    No tools, no planning step, no agent loop.
    Supports OpenAI and Anthropic (Claude) models.
    """

    def __init__(
        self,
        openai_client: OpenAI | None = None,
        anthropic_client: Anthropic | None = None,
    ) -> None:
        self.openai_client = openai_client or OpenAI(api_key=settings.OPENAI_API_KEY)
        self.anthropic_client = anthropic_client or Anthropic(
            api_key=settings.ANTHROPIC_API_KEY
        )

    def run(
        self,
        user_request: str,
        model: str = "gpt-4o",
    ) -> dict[str, Any]:
        """
        Execute the trip-planning task with one prompt and one response.

        Returns:
            - response: the model's complete trip plan
            - input_tokens: approximate input token count
            - output_tokens: from API usage
            - model: model used
        """
        prompt = TRIP_PLANNING_PROMPT.format(user_request=user_request)

        logger.info(f"[SinglePrompt] Running trip plan for: {user_request[:60]}...")

        if _is_anthropic_model(model):
            return self._run_anthropic(prompt, model)
        return self._run_openai(prompt, model)

    def _run_openai(self, prompt: str, model: str) -> dict[str, Any]:
        response = self.openai_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
        )
        content = response.choices[0].message.content or ""
        usage = response.usage
        input_tokens = usage.prompt_tokens if usage else 0
        output_tokens = usage.completion_tokens if usage else 0
        logger.info(f"[SinglePrompt] Done (OpenAI). Output tokens: {output_tokens}")
        return {
            "response": content,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "model": model,
            "approach": "single_prompt",
        }

    def _run_anthropic(self, prompt: str, model: str) -> dict[str, Any]:
        message = self.anthropic_client.messages.create(
            model=model,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
        )
        content = (
            getattr(message.content[0], "text", "") if message.content else ""
        )
        input_tokens = message.usage.input_tokens
        output_tokens = message.usage.output_tokens
        logger.info(f"[SinglePrompt] Done (Claude). Output tokens: {output_tokens}")
        return {
            "response": content,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "model": model,
            "approach": "single_prompt",
        }
