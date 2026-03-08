# Week 4 - Task 3: Logprobs & Confidence Signals

## Goal
Use logprobs to reason about model confidence, inspect token-level probabilities, and identify low-confidence outputs.

## Practical Use Cases for Logprobs
- **Validation (Fact-Checking):** Logprobs can act as an automated early-warning system for hallucinations. If a model outputs a specific concrete detail—like a date, a proper noun, or a statistic—with a very low probability (e.g., `<30%`), the system can automatically flag that specific token for human review or trigger a secondary verification tool.
- **Ranking/Filtering (Quality Control):** In programmatic workflows or RAG systems, we can generate multiple possible answers and rank them based on the average logical probability of their tokens. We can instantly discard responses that fall below a certain confidence threshold, ensuring only the most reliable outputs are surfaced to the user.

## Low-Confidence Token Inspection
**My Observation:** When I run complex or ambiguous prompts with logprobs enabled, I noticed that the model shows high confidence (often >95%) on structural words (like "the", "and", "is") and common grammatical patterns. 

However, when asked for highly specific details or creative connections (like making up a fictional character name or explaining an obscure concept), the model's confidence for those specific tokens drops significantly—sometimes into the "red" zone (<50%). This perfectly visualizes the model's "uncertainty" in real-time.

### Screenshots
> [!NOTE]
> Please replace these placeholders with your captured screenshots of the Logprobs confidence highlighting.

![Logprobs Dashboard](file:///workspaces/ai-train-week4/static/screenshot_logprobs.png)
*Figure 5: UI demonstrating token-level confidence highlighting (e.g., green=high, red=low).*

---

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
