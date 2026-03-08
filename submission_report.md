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

