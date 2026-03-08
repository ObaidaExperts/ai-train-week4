"""
Tool / Function Calling Service — Week 4, Task 4

Implements a real end-to-end tool-calling loop using the OpenAI API:
  1. Define JSON schemas for three tools.
  2. Send the user prompt + tool schemas to the model.
  3. Detect when the model requests a tool call.
  4. Validate the arguments returned by the model.
  5. Execute the tool (simulated or real).
  6. Return the result back to the model for a final natural-language answer.
  7. Handle malformed / invalid tool arguments gracefully.
"""

import json
import math
import logging
from typing import Any

from openai import OpenAI

from app.core.config import settings

logger = logging.getLogger("tool_service")


# ---------------------------------------------------------------------------
# Tool argument validation error
# ---------------------------------------------------------------------------

class ToolArgumentError(ValueError):
    """Raised when the model returns invalid or missing tool arguments."""
    pass


# ---------------------------------------------------------------------------
# Tool schemas (JSON Schema format used by OpenAI function calling)
# ---------------------------------------------------------------------------

TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": (
                "Get the current weather for a specified location. "
                "Returns temperature, conditions, and humidity."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City name, e.g. 'Paris' or 'New York'."
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "Temperature unit. Defaults to celsius."
                    }
                },
                "required": ["location"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": (
                "Evaluate a mathematical expression and return the numeric result. "
                "Supports basic arithmetic and functions like sqrt, pow, and abs."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "A valid mathematical expression, e.g. '12 * 15 + 7'."
                    }
                },
                "required": ["expression"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_stock_price",
            "description": "Get the latest stock price for a given ticker symbol.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "Stock ticker symbol, e.g. 'AAPL', 'GOOG', 'MSFT'."
                    }
                },
                "required": ["ticker"]
            }
        }
    }
]


# ---------------------------------------------------------------------------
# Tool executor
# ---------------------------------------------------------------------------

def _validate_and_execute_tool(name: str, raw_args: str) -> str:
    """
    Parse, validate, and execute a tool call.
    Raises ToolArgumentError on bad / missing arguments.
    Returns a JSON string with the tool result.
    """
    # 1. Parse JSON arguments
    try:
        args: dict[str, Any] = json.loads(raw_args)
    except json.JSONDecodeError as exc:
        raise ToolArgumentError(
            f"Model returned malformed JSON for tool '{name}': {exc}"
        )

    # 2. Route to the correct executor
    if name == "get_weather":
        location = args.get("location")
        if not location or not isinstance(location, str) or not location.strip():
            raise ToolArgumentError(
                "'get_weather' requires a non-empty string 'location'."
            )
        unit = args.get("unit", "celsius")
        if unit not in ("celsius", "fahrenheit"):
            raise ToolArgumentError(
                f"'get_weather' 'unit' must be 'celsius' or 'fahrenheit', got '{unit}'."
            )
        # Simulated response
        temp = 18 if unit == "celsius" else 64
        return json.dumps({
            "location": location.strip(),
            "temperature": temp,
            "unit": unit,
            "conditions": "Partly cloudy",
            "humidity": "62%",
            "note": "Simulated weather data"
        })

    elif name == "calculate":
        expression = args.get("expression")
        if not expression or not isinstance(expression, str):
            raise ToolArgumentError(
                "'calculate' requires a non-empty string 'expression'."
            )
        # Safe evaluation with whitelist of builtins
        safe_globals = {
            "__builtins__": {},
            "abs": abs, "round": round,
            "sqrt": math.sqrt, "pow": math.pow,
            "pi": math.pi, "e": math.e,
            "floor": math.floor, "ceil": math.ceil,
            "log": math.log, "sin": math.sin,
            "cos": math.cos, "tan": math.tan,
        }
        try:
            result = eval(expression, safe_globals)  # noqa: S307
            if not isinstance(result, (int, float)):
                raise ToolArgumentError(
                    f"'calculate' expression must produce a number, got {type(result).__name__}."
                )
            return json.dumps({"expression": expression, "result": result})
        except ToolArgumentError:
            raise
        except Exception as exc:
            raise ToolArgumentError(
                f"'calculate' could not evaluate '{expression}': {exc}"
            )

    elif name == "get_stock_price":
        ticker = args.get("ticker")
        if not ticker or not isinstance(ticker, str) or not ticker.strip():
            raise ToolArgumentError(
                "'get_stock_price' requires a non-empty string 'ticker'."
            )
        ticker = ticker.strip().upper()
        # Simulated prices
        prices = {
            "AAPL": 178.50, "GOOG": 151.20, "MSFT": 415.30,
            "AMZN": 185.10, "NVDA": 875.40, "META": 510.80,
        }
        price = prices.get(ticker, "N/A")
        return json.dumps({
            "ticker": ticker,
            "price_usd": price,
            "currency": "USD",
            "note": "Simulated market data"
        })

    else:
        raise ToolArgumentError(f"Unknown tool '{name}'.")


# ---------------------------------------------------------------------------
# ToolCallingService — the main service class
# ---------------------------------------------------------------------------

class ToolCallingService:
    """Manages the full OpenAI tool-calling loop."""

    def __init__(self, openai_client: OpenAI | None = None) -> None:
        self.client = openai_client or OpenAI(api_key=settings.OPENAI_API_KEY)

    def run_tool_loop(
        self,
        prompt: str,
        model: str = "gpt-4o",
        enabled_tools: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Execute the full tool-calling loop for the given prompt.

        Returns a dict with:
          - final_response: the model's final natural-language answer
          - tool_called: name of the tool invoked (or None)
          - tool_args: the arguments the model sent to the tool
          - tool_result: the result returned by the tool executor
          - tool_error: any validation / execution error message (or None)
          - steps: ordered list of all trace steps for UI rendering
        """
        # Filter tool schemas if caller specified a subset
        tools = TOOLS
        if enabled_tools is not None:
            tools = [t for t in TOOLS if t["function"]["name"] in enabled_tools]

        steps: list[dict[str, Any]] = []
        messages: list[dict[str, Any]] = [
            {"role": "user", "content": prompt}
        ]

        steps.append({"step": "user_prompt", "content": prompt})
        logger.info(f"[ToolLoop] Starting with prompt: {prompt[:80]!r}")

        # ── Round 1: initial model call ──────────────────────────────────────
        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice="auto",
        )

        choice = response.choices[0]
        finish_reason = choice.finish_reason

        # Model answered directly, no tool needed
        if finish_reason != "tool_calls":
            final_text = choice.message.content or ""
            steps.append({"step": "model_direct_answer", "content": final_text})
            logger.info("[ToolLoop] Model answered directly (no tool call).")
            return {
                "final_response": final_text,
                "tool_called": None,
                "tool_args": None,
                "tool_result": None,
                "tool_error": None,
                "steps": steps
            }

        # ── Tool call requested ──────────────────────────────────────────────
        tool_call = choice.message.tool_calls[0]
        tool_name = tool_call.function.name
        tool_args_raw = tool_call.function.arguments

        steps.append({
            "step": "tool_call_requested",
            "tool": tool_name,
            "args_raw": tool_args_raw
        })
        logger.info(f"[ToolLoop] Tool call: {tool_name}({tool_args_raw})")

        # ── Argument validation & execution ─────────────────────────────────
        tool_error: str | None = None
        tool_result_str: str = ""
        try:
            tool_result_str = _validate_and_execute_tool(tool_name, tool_args_raw)
            tool_result_data = json.loads(tool_result_str)
            steps.append({
                "step": "tool_result",
                "tool": tool_name,
                "result": tool_result_data
            })
            logger.info(f"[ToolLoop] Tool result: {tool_result_str[:120]}")
        except ToolArgumentError as exc:
            tool_error = str(exc)
            # Return a structured error payload back to the model
            tool_result_str = json.dumps({
                "error": tool_error,
                "note": "The tool call failed due to invalid arguments."
            })
            steps.append({
                "step": "tool_error",
                "tool": tool_name,
                "error": tool_error
            })
            logger.warning(f"[ToolLoop] Tool argument error: {tool_error}")

        # ── Round 2: send tool result back to model ──────────────────────────
        messages.append(choice.message)   # the assistant message with tool_calls
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": tool_result_str
        })

        final_response = self.client.chat.completions.create(
            model=model,
            messages=messages,
        )
        final_text = final_response.choices[0].message.content or ""
        steps.append({"step": "final_answer", "content": final_text})
        logger.info(f"[ToolLoop] Final answer: {final_text[:120]}")

        return {
            "final_response": final_text,
            "tool_called": tool_name,
            "tool_args": json.loads(tool_args_raw) if tool_args_raw else None,
            "tool_result": json.loads(tool_result_str),
            "tool_error": tool_error,
            "steps": steps
        }
