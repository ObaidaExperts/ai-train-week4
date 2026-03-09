# Week 4 Submission Report

This report covers **Tool/Function Calling** and **Single-Agent vs Agentic Flow**.

---

# Task 4: Tool / Function Calling

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

## Test Coverage (Task 4)

20 tests in `tests/test_tool_service.py`:
- Tool schema completeness
- Per-tool argument validation (valid + invalid cases)
- `ToolCallingService` happy path (tool triggered + direct answer)
- Tool error handling round-trip
- Enabled-tools filtering
- API endpoint tests (`/tools/schemas`, `/tool-call`, force_error mode)

---

# Single-Agent vs Agentic Flow

## Goal
Experience agents in practice: implement the same task with a single-prompt solution and an agentic flow, add planning and role separation, then compare complexity vs benefit.

## Task Definition

**Chosen task: Trip planning with budget and weather**

The user provides a free-form request such as:
- *"Plan a 3-day trip to Paris with a $500 budget"*
- *"2 days in Tokyo, $300, include weather"*

The system must produce a complete trip plan with:
- Overview and highlights
- Day-by-day itinerary
- Estimated costs breakdown
- Weather considerations and practical tips

This task is well-suited for comparison because:
1. **Single-prompt:** The LLM can answer in one shot using its training knowledge.
2. **Agentic:** Tools (`get_weather`, `calculate`) can provide real data (weather, budget math), and a planning step can structure the workflow.

---

## Implementation 1: Single-Prompt Solution

**Location:** `app/services/single_prompt_service.py`

**Approach:** One prompt, one API call, one response. No tools, no planning, no agent loop.

**Flow:**
1. User request is injected into a fixed prompt template.
2. The template instructs the model to produce a complete plan (overview, itinerary, costs, tips).
3. Single `chat.completions.create()` call.
4. Response returned as-is.

**Code structure:**
```python
TRIP_PLANNING_PROMPT = """You are a travel planner. Create a detailed trip plan...
User request: {user_request}
Provide a complete plan including: 1. Overview, 2. Day-by-day itinerary, ..."""

class SinglePromptService:
    def run(self, user_request: str, model: str = "gpt-4o") -> dict:
        prompt = TRIP_PLANNING_PROMPT.format(user_request=user_request)
        response = self.client.chat.completions.create(model=model, messages=[...])
        return {"response": content, "input_tokens": ..., "output_tokens": ...}
```

**Characteristics:**
- **Complexity:** Low — ~60 lines, one API call.
- **Latency:** Single round-trip.
- **Token usage:** One request + one response.
- **Data:** Relies entirely on model knowledge (no live weather, no real calculations).

---

## Implementation 2: Agentic Flow

**Location:** `app/services/agentic_service.py`

**Approach:** Two-phase flow with planning and role separation.

### Phase 1: Planning Step (Planner Role)

A dedicated **Planner** role creates a structured execution plan. No tools are available.

- **System prompt:** Instructs the model to output a JSON plan with steps.
- **Example plan:**
  ```json
  {
    "steps": [
      {"id": 1, "action": "get_weather", "reason": "Need weather for packing"},
      {"id": 2, "action": "calculate", "reason": "Break down budget allocation"},
      {"id": 3, "action": "synthesize", "reason": "Combine into final plan"}
    ]
  }
  ```
- **Output:** Parsed and passed to the Executor.

### Phase 2: Execution (Executor Role)

The **Executor** role receives the plan and the user request. It has access to tools:
- `get_weather(location)` — simulated weather for a city
- `calculate(expression)` — safe math (e.g. `500/3` for daily budget)
- `get_stock_price(ticker)` — rarely used for trips

**Agent loop:**
1. Send plan + user request to the model with tool schemas.
2. Model either:
   - Returns a **tool call** → validate args, execute tool, append result to conversation, repeat.
   - Returns a **final answer** → done.
3. Loop until the model produces a final answer or max iterations reached.

**Code structure:**
```python
# Phase 1: Planner (no tools)
plan_response = self.client.chat.completions.create(
    model=model, messages=planner_messages, temperature=0.3
)
plan = self._parse_plan(plan_response.choices[0].message.content)

# Phase 2: Executor (with tools) — agent loop
while iteration < max_iterations:
    exec_response = self.client.chat.completions.create(
        model=model, messages=executor_messages, tools=TOOLS, tool_choice="auto"
    )
    if finish_reason != "tool_calls":
        return final_answer
    for tool_call in tool_calls:
        result = _validate_and_execute_tool(name, args)
        executor_messages.append(tool_result)
```

**Characteristics:**
- **Complexity:** High — ~200 lines, two roles, agent loop, tool integration.
- **Latency:** Multiple round-trips (planning + 1–N execution steps).
- **Token usage:** Higher (planning prompt + execution context + tool results).
- **Data:** Can use real tool outputs (weather, calculations).

---

## Complexity vs Benefit Comparison

| Aspect | Single-Prompt | Agentic |
|--------|---------------|---------|
| **Lines of code** | ~60 | ~200 |
| **API calls** | 1 | 2+ (planning + execution loop) |
| **Latency** | Low | Higher |
| **Token cost** | Lower | Higher |
| **Tool use** | None | Yes (weather, calculate) |
| **Traceability** | Minimal | Full trace (plan, tool calls, steps) |
| **Flexibility** | Fixed flow | Adapts via plan + tool calls |
| **Data accuracy** | Model knowledge only | Can use live/simulated data |

---

## When Agents Are Worth It

### Agents are worth it when:

1. **External data is required** — Weather, stock prices, database lookups, API calls. The single-prompt approach cannot fetch live data.

2. **Multi-step reasoning benefits from structure** — A planning step reduces hallucination and ensures all sub-tasks are addressed. The plan acts as a checklist.

3. **Traceability and debugging matter** — The agentic flow produces a full trace (plan, tool calls, results). This is valuable for auditing, debugging, and user trust.

4. **The task is complex and dynamic** — When the user’s request can lead to different tool sequences (e.g. weather only vs. weather + budget + stocks), the agent can adapt.

5. **Accuracy over cost** — When correctness of numbers (e.g. budget math, weather) matters more than latency or token cost.

### Single-prompt is better when:

1. **The task is simple and self-contained** — A single, well-crafted prompt is enough. No tools needed.

2. **Latency and cost are critical** — One API call is cheaper and faster than planning + multiple execution steps.

3. **Model knowledge is sufficient** — For many trip-planning questions, the model’s training data is adequate. Real-time weather adds value but is not always essential.

4. **Implementation simplicity matters** — Single-prompt is easier to build, test, and maintain.

### Conclusion

For **trip planning**, the single-prompt solution is often sufficient: it is simpler, cheaper, and faster. The agentic approach adds value when:
- The user explicitly wants live weather or precise budget calculations.
- Traceability (e.g. “how was this plan built?”) is important.
- The task is extended (e.g. booking, availability checks) and will need more tools.

**Recommendation:** Start with single-prompt for MVP. Add an agentic flow when external data or structured multi-step execution becomes a requirement.

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/agentic-flow/single` | POST | Run trip planning with single-prompt |
| `/agentic-flow/agentic` | POST | Run trip planning with agentic flow |

**Request body:**
```json
{
  "user_request": "Plan a 3-day trip to Paris with $500 budget",
  "model": "gpt-4o"
}
```

**Response (both):**
```json
{
  "response": "...",
  "input_tokens": 150,
  "output_tokens": 400,
  "model": "gpt-4o",
  "approach": "single_prompt" | "agentic"
}
```

The agentic response additionally includes `plan`, `steps`, and `iterations`.

---

## Test Coverage

9 new tests in `tests/test_agentic_flow.py`:

- **Single-prompt:** Prompt template, return structure, user request passed to LLM.
- **Agentic:** Plan parsing (valid JSON, fallback), planning phase called, tool calls in trace.
- **API:** `POST /agentic-flow/single` and `POST /agentic-flow/agentic` return expected structure.

**Total tests: 53 — all passing ✅**
