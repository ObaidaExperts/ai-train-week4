# Week 4 - Task 4: Tool / Function Calling

## Goal
Build a real tool-calling loop that demonstrates end-to-end function calling with argument validation, error handling, and the full request/response cycle.

## Tool Definitions

Three tools were defined as JSON schemas and registered with the OpenAI API:

| Tool | Description | Required Args |
|---|---|---|
| `get_weather` | Returns temperature, conditions & humidity for a city | `location` (string) |
| `calculate` | Evaluates a mathematical expression safely | `expression` (string) |
| `get_stock_price` | Returns the latest price for a ticker symbol | `ticker` (string) |

**Schema example (`get_weather`):**
```json
{
  "name": "get_weather",
  "parameters": {
    "type": "object",
    "properties": {
      "location": { "type": "string" },
      "unit": { "type": "string", "enum": ["celsius", "fahrenheit"] }
    },
    "required": ["location"]
  }
}
```

## Tool-Calling Loop

The full loop implemented in `ToolCallingService.run_tool_loop()`:

1. **User prompt** sent to model with tool schemas
2. Model returns either a direct answer **or** a `tool_calls` request
3. Tool arguments are **validated** (type checks, required fields, enum constraints)
4. Tool is **executed** (simulated weather/stock; real `eval` for math)
5. Tool result returned to model as a `tool` role message
6. Model produces the **final natural-language answer**
7. Full **trace** of all steps returned for UI display

## Argument Validation & Error Handling

Validation raises a `ToolArgumentError` for:
- Malformed JSON from the model
- Missing required arguments
- Invalid enum values (e.g. `unit: "kelvin"`)
- Math expressions that can't be evaluated

**Failure case:** The "⚠️ Demo Error" button on the UI forces a malformed call (`unit: 42` instead of `"celsius"`) and demonstrates that the error is caught, wrapped in a structured JSON payload, and sent back to the model—which then gracefully handles it in its final response.

## Test Coverage

20 new tests were added in `tests/test_tool_service.py`:
- Tool schema completeness
- Per-tool argument validation (valid + invalid cases)
- `ToolCallingService` happy path (tool triggered + direct answer)
- Tool error handling round-trip
- Enabled-tools filtering
- API endpoint tests (`/tools/schemas`, `/tool-call`, force_error mode)

**Total tests: 44 — all passing ✅**
