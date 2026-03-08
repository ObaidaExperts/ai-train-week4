# Week 4 - Task 1: Tokenization and Context Limits

## Project Description
I built this project to be an advanced AI experimentation platform that gives me real-time token analysis, cost estimation, and context limit monitoring. It supports three major providers—**OpenAI**, **Anthropic (Claude)**, and **Google (Gemini)**—and features a modern dashboard I designed for interactive testing and automated "stress test" scenarios.

## Core Functionalities

### 1. Robust Token Counting
One of the first things I tackled was ensuring my app could accurately count tokens for different providers. I used the `tiktoken` library for OpenAI models, which is the industry standard. For models where a direct library wasn't available, like Gemini, I implemented a high-accuracy estimation method. This ensures that I always have a clear view of how much data I'm sending to the AI.

```python
# app/core/tokenizer.py
@staticmethod
def count_openai_tokens(text: str, model: str) -> int:
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        # Fallback for newer models not yet in tiktoken
        encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))
```

### 2. Real-Time Cost Calculation
I wanted to know exactly how much each request costs as I'm making it. I built a dynamic pricing system that pulls the latest per-million-token rates for every model. My `calculate_cost` function takes the actual input and output tokens and gives me a precise USD value, which helps me stay within budget during my experiments.

```python
# app/services/experiment_service.py
def calculate_cost(self, input_tokens: int, output_tokens: int, model: AIModel) -> float:
    rates = model.pricing
    input_cost = (input_tokens / 1_000_000) * rates["input"]
    output_cost = (output_tokens / 1_000_000) * rates["output"]
    return input_cost + output_cost
```

### 3. Multi-Provider Integration (Claude & Gemini)
I didn't want to be limited to just one AI provider. I successfully integrated the Anthropic SDK to support the **Claude 3 family (Sonnet, Opus, Haiku)**. This was a great exercise in handling different API structures and message formats. I also simulated Gemini support to show how I can swap providers flexibly without breaking my dashboard.

```python
# app/services/experiment_service.py
elif model_type.is_anthropic:
    # I use the official Anthropic client to get accurate results
    message = self.anthropic_client.messages.create(
        model=model_str,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}]
    )
    response_text = message.content[0].text
    output_tokens = message.usage.output_tokens
```

### 4. Smart Dynamic Configuration
To keep my frontend clean and scalable, I made the dashboard fully data-driven. Instead of hardcoding model lists in HTML, I created a metadata API. My dashboard "asks" the backend what models and experiment types are available, so whenever I add a new model to my Python code, it instantly appears in my UI's dropdown menu.

### 5. Categorized Experiment Logging
For my analysis, I needed more than just a dump of data. I implemented categorized logging where every run is tagged as a "Baseline", "Stress Test", or "Pricing Comparison". This makes my `experiment_results.csv` incredibly easy to filter and read, giving me a clear history of my project's progress.

## Experiments & Results

### Experiment 1: Prompt Length vs. Tokens
**My Observation**: I noticed that as my prompt length grows, the token count increases linearly. This is a direct reminder to me of how quickly costs can escalate if I don't keep my inputs concise.

### Experiment 2: Context Window Stress Test
**My Observation**: Using the "Stress Test" feature I built, I pushed the context window close to the limit. I can see clear visual flags in my dashboard (the color-coded progress bar) when I hit the 80% mark, which is a lifesaver for preventing truncation or completion failures.

### Screenshots
> [!NOTE]
> Please replace these placeholders with your captured screenshots.

![Main Dashboard](file:///workspaces/ai-train-week4/static/screenshot_dashboard.png)
*Figure 1: Experiment Dashboard showing real-time token analysis and cost estimation.*

![Context Warning](file:///workspaces/ai-train-week4/static/screenshot_warning.png)
*Figure 2: Stress test showing the context window progress bar in the warning zone (80%+).*

---

## Why Context Limits Matter to Me
In my research, I've realized that context limits are essentially the "short-term memory" of an AI model. Mastering these limits is critical for my projects because:
1. **Preventing Information Loss**: If I exceed the limit, the model starts "forgetting" the beginning of our conversation, which ruins the logic of complex tasks.
2. **Controlling My Expenses**: I've seen how larger contexts can quickly spike the cost of a single interaction. Monitoring this helps me keep my API usage efficient.
3. **Optimizing Speed**: I noticed that very large contexts can make the AI slower to respond. By keeping an eye on my usage, I can ensure my apps stay snappy and responsive.
4. **Ensuring Stability**: There's nothing worse than a "400 Bad Request" error in the middle of a run. My monitoring system gives me the reliability I need to avoid context overflow errors.

---

## Artifacts
- **Dashboard**: [index.html](file:///workspaces/ai-train-week4/static/index.html)
- **Log Data**: [experiment_results.csv](file:///workspaces/ai-train-week4/experiment_results.csv)
- **Unit Tests**: [tests/](file:///workspaces/ai-train-week4/tests/)
