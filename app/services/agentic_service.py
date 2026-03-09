"""
Agentic Service — Trip planning

Implements trip planning using an agent loop with:
  1. Planning step: A dedicated Planner role creates a step-by-step plan.
  2. Role separation: Planner (no tools) vs Executor (with tools).
  3. Agent loop: Executor iterates, calls tools, synthesizes the final answer.

Supports both OpenAI and Anthropic (Claude) models.
"""

import json
import logging
from typing import Any

from openai import OpenAI
from anthropic import Anthropic

from app.core.config import settings
from app.services.tool_service import TOOLS, _validate_and_execute_tool

logger = logging.getLogger("agentic_service")


def _is_anthropic_model(model: str) -> bool:
    """Return True if the model is an Anthropic/Claude model."""
    return model.startswith("claude")


def _anthropic_tools() -> list[dict[str, Any]]:
    """Convert OpenAI tool format to Anthropic format."""
    return [
        {
            "name": t["function"]["name"],
            "description": t["function"]["description"],
            "input_schema": t["function"]["parameters"],
        }
        for t in TOOLS
    ]


# ---------------------------------------------------------------------------
# Planner role — creates the plan (no tools)
# ---------------------------------------------------------------------------

PLANNER_SYSTEM = """You are a travel planning coordinator. Create a structured execution plan.

Given the user's trip request, output a JSON object with this structure:
{
  "steps": [
    {"id": 1, "action": "get_weather", "reason": "Need weather for packing"},
    {"id": 2, "action": "calculate", "reason": "Break down budget allocation"},
    {"id": 3, "action": "synthesize", "reason": "Combine into final plan"}
  ]
}

Rules:
- Use "get_weather" when the user specifies a location.
- Use "calculate" when budget or costs are involved (e.g. "500/3" for daily budget).
- Always end with "synthesize". Keep 3-5 steps. Be concise."""

# ---------------------------------------------------------------------------
# Executor role — runs the plan with tools
# ---------------------------------------------------------------------------

EXECUTOR_SYSTEM = """You are a travel plan executor with tools available.

Tools: get_weather(location), calculate(expression), get_stock_price(ticker).

Execute the plan step by step. Call tools when a step requires real data.
After gathering data, synthesize a complete trip plan.

Format: overview, day-by-day itinerary, costs, tips."""


class AgenticService:
    """
    Executes trip planning via an agent loop with planning and role separation.
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
        max_iterations: int = 10,
    ) -> dict[str, Any]:
        """
        Execute the agentic flow: plan -> execute (with tools) -> synthesize.
        """
        if _is_anthropic_model(model):
            return self._run_anthropic(user_request, model, max_iterations)
        return self._run_openai(user_request, model, max_iterations)

    def _run_openai(
        self, user_request: str, model: str, max_iterations: int
    ) -> dict[str, Any]:
        steps_trace: list[dict[str, Any]] = []
        total_input = 0
        total_output = 0

        planner_messages = [
            {"role": "system", "content": PLANNER_SYSTEM},
            {"role": "user", "content": f"User request: {user_request}"},
        ]

        logger.info("[Agentic] Phase 1: Planning (OpenAI)...")
        plan_response = self.openai_client.chat.completions.create(
            model=model,
            messages=planner_messages,
            temperature=0.3,
        )
        plan_text = plan_response.choices[0].message.content or "{}"
        if plan_response.usage:
            total_input += plan_response.usage.prompt_tokens
            total_output += plan_response.usage.completion_tokens

        plan = self._parse_plan(plan_text)
        steps_trace.append({"phase": "planning", "plan": plan, "raw": plan_text})

        executor_messages: list[dict[str, Any]] = [
            {"role": "system", "content": EXECUTOR_SYSTEM},
            {
                "role": "user",
                "content": (
                    f"Plan to execute:\n{json.dumps(plan, indent=2)}\n\n"
                    f"User request: {user_request}"
                ),
            },
        ]

        iteration = 0
        while iteration < max_iterations:
            iteration += 1
            logger.info(f"[Agentic] Phase 2: Executor iteration {iteration} (OpenAI)")

            exec_response = self.openai_client.chat.completions.create(
                model=model,
                messages=executor_messages,
                tools=TOOLS,
                tool_choice="auto",
            )
            choice = exec_response.choices[0]
            if exec_response.usage:
                total_input += exec_response.usage.prompt_tokens
                total_output += exec_response.usage.completion_tokens

            if choice.finish_reason != "tool_calls":
                final_text = choice.message.content or ""
                steps_trace.append({"phase": "complete", "response": final_text})
                logger.info("[Agentic] Done.")
                return {
                    "response": final_text,
                    "plan": plan,
                    "steps": steps_trace,
                    "input_tokens": total_input,
                    "output_tokens": total_output,
                    "model": model,
                    "approach": "agentic",
                    "iterations": iteration,
                }

            for tool_call in choice.message.tool_calls or []:
                name = tool_call.function.name
                args_raw = tool_call.function.arguments or "{}"
                try:
                    result = _validate_and_execute_tool(name, args_raw)
                    steps_trace.append({
                        "phase": "tool_call",
                        "tool": name,
                        "args": json.loads(args_raw),
                        "result": json.loads(result),
                    })
                    executor_messages.append(choice.message)
                    executor_messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result,
                    })
                except Exception as e:
                    steps_trace.append({"phase": "tool_error", "tool": name, "error": str(e)})
                    executor_messages.append(choice.message)
                    executor_messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps({"error": str(e)}),
                    })

        last = executor_messages[-1] if executor_messages else {}
        last_content = last.get("content", "")
        return {
            "response": f"Agent max iterations. Last: {last_content[:200]}...",
            "plan": plan,
            "steps": steps_trace,
            "input_tokens": total_input,
            "output_tokens": total_output,
            "model": model,
            "approach": "agentic",
            "iterations": max_iterations,
        }

    def _run_anthropic(
        self, user_request: str, model: str, max_iterations: int
    ) -> dict[str, Any]:
        steps_trace: list[dict[str, Any]] = []
        total_input = 0
        total_output = 0

        logger.info("[Agentic] Phase 1: Planning (Claude)...")
        plan_message = self.anthropic_client.messages.create(
            model=model,
            max_tokens=1024,
            system=PLANNER_SYSTEM,
            messages=[{"role": "user", "content": f"User request: {user_request}"}],
            temperature=0.3,
        )
        plan_text = ""
        for block in plan_message.content:
            text = getattr(block, "text", None)
            if text:
                plan_text += text
        plan_text = plan_text or "{}"
        total_input += plan_message.usage.input_tokens
        total_output += plan_message.usage.output_tokens

        plan = self._parse_plan(plan_text)
        steps_trace.append({"phase": "planning", "plan": plan, "raw": plan_text})

        executor_content = (
            f"Plan to execute:\n{json.dumps(plan, indent=2)}\n\n"
            f"User request: {user_request}"
        )
        messages: list[dict[str, Any]] = [
            {"role": "user", "content": executor_content}
        ]

        iteration = 0
        while iteration < max_iterations:
            iteration += 1
            logger.info(f"[Agentic] Phase 2: Executor iteration {iteration} (Claude)")

            exec_message = self.anthropic_client.messages.create(
                model=model,
                max_tokens=1024,
                system=EXECUTOR_SYSTEM,
                tools=_anthropic_tools(),
                tool_choice={"type": "auto"},
                messages=messages,
            )
            total_input += exec_message.usage.input_tokens
            total_output += exec_message.usage.output_tokens

            tool_uses = [
                b for b in exec_message.content
                if getattr(b, "type", None) == "tool_use"
            ]
            if not tool_uses:
                final_text = ""
                for block in exec_message.content:
                    text = getattr(block, "text", None)
                    if text:
                        final_text += text
                steps_trace.append({"phase": "complete", "response": final_text})
                logger.info("[Agentic] Done.")
                return {
                    "response": final_text,
                    "plan": plan,
                    "steps": steps_trace,
                    "input_tokens": total_input,
                    "output_tokens": total_output,
                    "model": model,
                    "approach": "agentic",
                    "iterations": iteration,
                }

            tool_results: list[dict[str, Any]] = []
            for block in exec_message.content:
                if getattr(block, "type", None) == "tool_use":
                    tool_id = getattr(block, "id", "")
                    name = getattr(block, "name", "")
                    inp = getattr(block, "input", {})
                    args_raw = json.dumps(inp) if isinstance(inp, dict) else str(inp)
                    try:
                        result = _validate_and_execute_tool(name, args_raw)
                        steps_trace.append({
                            "phase": "tool_call",
                            "tool": name,
                            "args": inp if isinstance(inp, dict) else {},
                            "result": json.loads(result),
                        })
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": tool_id,
                            "content": result,
                        })
                    except Exception as e:
                        steps_trace.append({
                            "phase": "tool_error",
                            "tool": name,
                            "error": str(e),
                        })
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": tool_id,
                            "content": json.dumps({"error": str(e)}),
                        })

            assistant_content = self._content_to_params(exec_message.content)
            messages.append({"role": "assistant", "content": assistant_content})
            messages.append({"role": "user", "content": tool_results})

        return {
            "response": "Agent reached max iterations.",
            "plan": plan,
            "steps": steps_trace,
            "input_tokens": total_input,
            "output_tokens": total_output,
            "model": model,
            "approach": "agentic",
            "iterations": max_iterations,
        }

    def _content_to_params(self, content: Any) -> list[dict[str, Any]]:
        """Convert Anthropic response content blocks to API param format."""
        out: list[dict[str, Any]] = []
        for block in content:
            if hasattr(block, "model_dump"):
                out.append(block.model_dump(exclude_none=True))
            elif isinstance(block, dict):
                out.append(block)
            elif hasattr(block, "type"):
                t = getattr(block, "type", "")
                if t == "text":
                    out.append({"type": "text", "text": getattr(block, "text", "")})
                elif t == "tool_use":
                    out.append({
                        "type": "tool_use",
                        "id": getattr(block, "id", ""),
                        "name": getattr(block, "name", ""),
                        "input": getattr(block, "input", {}),
                    })
        return out

    def _parse_plan(self, text: str) -> dict[str, Any]:
        """Extract JSON plan from planner output."""
        try:
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                out: dict[str, Any] = json.loads(text[start:end])
                return out
        except json.JSONDecodeError:
            pass
        return {"steps": [{"id": 1, "action": "synthesize", "reason": "Fallback"}]}
