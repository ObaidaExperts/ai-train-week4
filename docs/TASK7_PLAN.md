# Task 7: Measure Quality, Cost, Latency — Plan

## Goal
Make trade-offs visible across providers.

---

## 1. Metrics to Capture

| Metric | Description | Source |
|--------|-------------|--------|
| **TTFT** (Time To First Token) | Latency until first token received | Streaming API |
| **Total latency** | End-to-end response time | Already have `duration_ms` |
| **Input tokens** | Prompt tokens | Already captured |
| **Output tokens** | Completion tokens | Already captured |
| **Cost (USD)** | Estimated from tokens × model pricing | Pricing lookup |
| **Quality score** | Manual 1–5 rating by user | UI input after viewing |

---

## 2. Implementation Approach

### 2.1 Extend Multi-SDK Result Schema

```python
{
    "response": str,
    "provider": str,
    "model": str,
    "input_tokens": int,
    "output_tokens": int,
    "ttft_ms": float | None,      # NEW: null if streaming not used
    "duration_ms": float,         # Total latency (existing)
    "cost_usd": float,            # NEW: 0 for local models
    "error": str | None,
}
```

### 2.2 TTFT (Time To First Token)

- **OpenAI, Anthropic, vLLM, Llama.cpp:** Use streaming; record time of first chunk.
- **Gemini:** Use `generate_content` with stream if supported; else `ttft_ms = null`.
- **Fallback:** If streaming fails, `ttft_ms = null`; table shows "—".

### 2.3 Cost Calculation

- Add pricing lookup for multi-SDK models (extend or mirror `AIModel.pricing`).
- Models: gpt-4o-mini, claude-sonnet-4-6, gemini-2.0-flash, etc.
- Local (vLLM, Llama.cpp): `cost_usd = 0`.

### 2.4 Quality Score (Manual)

- User rates each result 1–5 after viewing.
- Stored in frontend state; included in comparison table.
- API does not persist quality; it's a per-session UI metric.

---

## 3. Comparison Table

Add a **Metrics Comparison** table in the Multi-SDK tab:

| Provider | Model | TTFT (ms) | Total (ms) | In | Out | Cost ($) | Quality |
|----------|-------|-----------|------------|-----|-----|----------|---------|
| openai   | gpt-4o-mini | 120 | 2500 | 50 | 300 | 0.0002 | ★★★★☆ |
| anthropic| claude-sonnet-4-6 | 80 | 3200 | 55 | 280 | 0.0045 | ★★★★★ |
| ...      | ...   | ... | ... | ... | ... | ... | (user rates) |

- Table appears after "Run All Providers" completes.
- Quality column: star selector (1–5) per row, filled by user.

---

## 4. Architecture

- **Backend:** Extend `multi_sdk_service.py`:
  - Add `ttft_ms` and `cost_usd` to `_normalized_result`.
  - Add streaming where supported for TTFT.
  - Add `_get_model_pricing(provider, model) -> (input_per_m, output_per_m)`.
- **Frontend:** Extend Multi-SDK tab:
  - Add comparison table below results.
  - Add quality star input per result row.
  - Populate table from `results` + user quality scores.

---

## 5. Acceptance Criteria

- [x] TTFT recorded (OpenAI via streaming; others null)
- [x] Total latency recorded
- [x] Token usage recorded
- [x] Cost calculated and displayed
- [x] Manual quality score (1–5) per result
- [x] Comparison table clearly shows trade-offs
