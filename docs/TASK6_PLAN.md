# Task 6: Multi-SDK Model Execution — Implementation Plan

> **Status:** ✅ Implemented (see `submission_report.md`)

## Goal
Run the same task across different providers (OpenAI, Anthropic, Gemini, local vLLM, Llama.cpp) with normalized prompts and consistent output capture.

---

## 1. Task Definition

**Chosen task: Trip planning** (reuse from Task 5 single-prompt)

- **Input:** User request (e.g. "Plan a 3-day trip to Paris with $500 budget")
- **Output:** Structured trip plan (overview, itinerary, costs, tips)
- **Why:** Simple, single-prompt, no tools — ideal for cross-provider comparison

---

## 2. Providers & SDKs

| Provider | SDK | Model(s) | Notes |
|----------|-----|----------|-------|
| **OpenAI** | `openai` | gpt-4o-mini (default) | Native |
| **Anthropic** | `anthropic` | claude-sonnet-4-6 (default) | Native |
| **Gemini** | `google-generativeai` | gemini-2.0-flash (default) | Real API |
| **Local (vLLM)** | `openai` with `base_url` | Any model on vLLM server | OpenAI-compatible API |
| **Local (Llama.cpp)** | `openai` with `base_url` | GGUF models via llama-cpp-python server | OpenAI-compatible API |

**Local model strategy:** vLLM and Llama.cpp both expose an **OpenAI-compatible** HTTP API. We use the **same OpenAI SDK** with `base_url` pointing to the local server. No new SDK needed. Llama.cpp uses `llama-cpp-python[server]`.

---

## 3. Normalized Prompt

Single prompt template used by all providers:

```python
TRIP_PLANNING_PROMPT = """You are a travel planner. Create a detailed trip plan based on the user's request.

User request: {user_request}

Provide a complete plan including:
1. Overview and highlights
2. Day-by-day itinerary with activities
3. Estimated costs breakdown (accommodation, food, activities, transport)
4. Practical tips (weather considerations, packing, local customs)
5. Budget summary and any money-saving suggestions

Be specific and actionable. Format clearly with headers and bullet points."""
```

All providers receive the **exact same** formatted prompt. No provider-specific tweaks.

---

## 4. Normalized Output Schema

Every provider returns the same structure:

```python
{
    "response": str,           # The model's text output
    "provider": str,           # "openai" | "anthropic" | "gemini" | "vllm" | "llamacpp"
    "model": str,              # Model identifier used
    "input_tokens": int,
    "output_tokens": int,
    "duration_ms": float,      # Wall-clock time
    "error": str | None,       # If failed
}
```

---

## 5. Architecture

**Implemented:** Single `MultiSDKService` class in `app/services/multi_sdk_service.py` with provider-specific methods (`_run_openai`, `_run_anthropic`, `_run_gemini`, `_run_vllm`, `_run_llamacpp`), all returning the same schema.

---

## 6. Implementation Steps

### Phase 1: Core service ✅
1. Create `MultiSDKService` in `app/services/multi_sdk_service.py`
2. Define `run(user_request, provider, model)` and `run_all(user_request, providers)`
3. Implement OpenAI provider (reuse existing client)
4. Implement Anthropic provider (reuse existing client)

### Phase 2: Gemini ✅
5. Add real Gemini API via `google-generativeai` SDK
6. Map Gemini `usage_metadata` to normalized schema (gemini-2.0-flash default)

### Phase 3: Local (vLLM) ✅
7. Add config: `VLLM_BASE_URL`, `VLLM_MODEL`
8. Add `vllm` provider using OpenAI client with `base_url`

### Phase 4: Local (Llama.cpp) ✅
9. Add `llama-cpp-python[server]` dependency
10. Add config: `LLAMA_CPP_BASE_URL` (e.g. `http://localhost:8080/v1`)
11. Add `llamacpp` provider using OpenAI client with `base_url`

### Phase 5: API & UI ✅
12. Add `POST /multi-sdk/run` and `POST /multi-sdk/run-all`
13. Add Multi-SDK tab with provider selector, results grid

### Phase 6: Tests & docs ✅
14. 12 unit tests in `tests/test_multi_sdk_service.py`
15. Updated submission_report.md

---

## 7. Config (.env)

```env
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...
GOOGLE_API_KEY=...                    # For Gemini
VLLM_BASE_URL=http://localhost:8001/v1   # Optional; local vLLM server
VLLM_MODEL=default                      # Model name on vLLM server
LLAMA_CPP_BASE_URL=http://localhost:8080/v1   # Optional; llama-cpp-python server
```

---

## 8. Local vLLM Setup (for dev/test)

```bash
# Install vLLM (optional, for local testing)
pip install vllm

# Start vLLM server (e.g. small model)
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-3.2-3B-Instruct \
  --host 0.0.0.0 \
  --port 8001
```

Then set `VLLM_BASE_URL=http://localhost:8001/v1`. If not set, vLLM provider returns "VLLM_BASE_URL not configured".

---

## 8b. Local Llama.cpp Setup (for dev/test)

```bash
# llama-cpp-python[server] is in project dependencies
# Download a GGUF model (e.g. from Hugging Face)
huggingface-cli download TheBloke/Llama-2-7B-Chat-GGUF llama-2-7b-chat.Q4_K_M.gguf --local-dir ./models

# Start server (port 8080 to avoid conflict with app on 8000)
python3 -m llama_cpp.server --model ./models/llama-2-7b-chat.Q4_K_M.gguf --port 8080 --host 0.0.0.0
# Or: ./scripts/run_llama_server.sh ./models/llama-2-7b-chat.Q4_K_M.gguf
```

Then set `LLAMA_CPP_BASE_URL=http://localhost:8080/v1` in `.env`.

---

## 9. Acceptance Criteria Checklist

- [x] Same task runs on OpenAI SDK
- [x] Same task runs on Anthropic SDK
- [x] Same task runs on Gemini SDK (real API)
- [x] Same task runs on local model (vLLM via OpenAI-compatible API)
- [x] Same task runs on local model (Llama.cpp via llama-cpp-python server)
- [x] Prompts are normalized (one template for all)
- [x] Outputs are captured consistently (same schema)

---

## 10. Optional Enhancements

- **Parallel execution:** Run all providers at once, compare results side-by-side
- **Fallback:** If one provider fails, others still return
- **Graceful degradation:** vLLM/Gemini/Llama.cpp optional; app works without them
