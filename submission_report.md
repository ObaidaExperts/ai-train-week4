# Week 4 - Task 2: Decoding Strategies Playground

## Project Description
I built a "Decoding Strategies Playground" to explore and document the impact of different decoding parameters (temperature, top-k, top-p) on AI model outputs. This allows for comparing deterministic vs. creative outputs and identifying use cases for each strategy.

## Core Decoding Parameters


### 1. Temperature Control
I've learned that **Temperature** is a key parameter for controlling the randomness of AI responses.
- **Low Temperature (0.0 - 0.2)**: Essential for tasks that require high precision and determinism, such as code generation or data extraction.
- **High Temperature (0.7 - 1.0+)**: Perfect for creative writing, brainstorming, or roleplaying, where variety and "uniqueness" are valued.

### 2. Top-P (Nucleus Sampling)
**Top-P** acts as a dynamic filter for the model's vocabulary. It tells the AI to only consider the most likely tokens whose cumulative probability exceeds the value $P$. I use this to balance quality and variety without the sometimes "chaotic" results of high temperature alone.

### 3. Top-K Sampling
Mainly used in models like Claude and Gemini, **Top-K** limits the AI's choice to the $K$ most probable next tokens. This is another layer of control I've integrated to ensure my outputs stay relevant and grounded.

### 🧪 Experiment 3: Temperature Comparison
**My Observation**: I ran a prompt with `temperature=0` (Deterministic) and then with `temperature=1` (Creative).
- **At 0.0**: The response was consistent, factual, and direct every single time I ran it.
- **At 1.0**: The model became much more expressive, using varied vocabulary and more descriptive phrasing.

### 🧪 Experiment 4: Top-P vs Top-K
**My Observation**:

| Feature     | Top-K                | Top-P                     |
| ----------- | -------------------- | ------------------------- |
| Control     | number of tokens     | cumulative probability    |
| Flexibility | less flexible        | more flexible             |
| Usage       | experiments/research | widely used in production |

### Screenshots
> [!NOTE]
> Please replace these placeholders with your captured screenshots of the Decoding Strategies Playground.

![Decoding Controls](/workspaces/ai-train-week4/static/screenshot_decoding_controls.png)
*Figure 3: New decoding parameters (Temperature, Top-P, Top-K) in my dashboard.*

![Side-By-Side Comparison](/workspaces/ai-train-week4/static/screenshot_comparison.png)
*Figure 4: Side-by-side comparison of different decoding strategies for the same prompt.*

---

## Why Decoding Parameters Matter to Me
Understanding these parameters allows me to:
1. **Tailor Outputs to the Task**: I can switch from a "creative assistant" to a "starkly factual extractor" just by adjusting a few sliders.
2. **Improve Consistency**: For production-ready apps, I need to know how to set parameters to get reliable results every time.
3. **Explore Creativity**: It's fascinating to see how the same model can produce wildly different "personalities" based on my configuration.

---

## Artifacts
- **Dashboard**: [index.html](file:///workspaces/ai-train-week4/static/index.html)
- **Log Data**: [experiment_results.csv](file:///workspaces/ai-train-week4/experiment_results.csv)
- **Unit Tests**: [tests/](file:///workspaces/ai-train-week4/tests/)
