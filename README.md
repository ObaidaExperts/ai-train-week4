# Tokenization and Context Limits Experiment Dashboard

A premium AI experimentation platform built with FastAPI and Vanilla CSS. Analyze token counts, costs, and context limits across **OpenAI**, **Gemini**, and **Claude** models in real-time.

![Dashboard Preview](file:///workspaces/ai-train-week4/static/screenshot_placeholder.png)

## 🌟 Key Features
- **Multi-Model Support**: Compare results across OpenAI (GPT-4o, o1), Gemini (1.5 Pro/Flash), and Claude (3.5 Sonnet, Opus, Haiku).
- **Interactive Dashboard**: Modern, responsive UI to run experiments and visualize data.
- **Dynamic Configuration**: UI automatically updates selections based on backend model definitions.
- **Context Limit Monitoring**: Visual warnings when a prompt approaches 80%+ of a model's context window.
- **Experiment Categorization**: Label experiments as "Stress Test", "Baseline", etc., for organized logging.
- **Persistent Storage**: All results are saved to a version-controlled CSV repository.
- **Graceful Error Handling**: Concise, human-readable API error messages for out-of-quota or rate-limit scenarios.

## 📁 Project Structure
- `app/api/`: Endpoints, Pydantic schemas, and centralized exception middleware.
- `app/core/`: Model definitions (pricing/limits), configuration, and tokenizers.
- `app/services/`: `ExperimentService` (business logic) and `ResultsRepository` (CSV storage).
- `static/`: Modern dashboard frontend (HTML/CSS/JS).
- `tests/`: Comprehensive unit test suite covering API, logic, and persistence.

## 🚀 Getting Started

1. **Setup Environment**:
   ```bash
   cp .env.example .env
   # Add your API keys (OPENAI_API_KEY, ANTHROPIC_API_KEY) to .env
   ```

2. **Install Dependencies**:
   ```bash
   poetry install
   ```

3. **Start the server**:
   ```bash
   poetry run python -m app.main
   ```

4. **Open the Dashboard**:
   Navigate to [http://localhost:8000](http://localhost:8000)

## 🧪 Experiments
Use the **"Stress Test"** button to automatically push a model's context window to its limits and observe the system's behavior and warnings.

## 📊 Results
Logged experiments can be retrieved via the UI or by visiting `GET /results`.

## 🦙 Llama.cpp (Local Models)

The Multi-SDK tab supports local models via [llama-cpp-python](https://github.com/abetlen/llama-cpp-python)'s OpenAI-compatible server.

1. **Install** (included in `poetry install`): `llama-cpp-python[server]`

2. **Download a GGUF model** (e.g. from [Hugging Face](https://huggingface.co/models?search=gguf)):
   ```bash
   huggingface-cli download TheBloke/Llama-2-7B-Chat-GGUF llama-2-7b-chat.Q4_K_M.gguf --local-dir ./models
   ```

3. **Start the server** (port 8080 to avoid conflict with the app on 8000):
   ```bash
   python3 -m llama_cpp.server --model ./models/llama-2-7b-chat.Q4_K_M.gguf --port 8080 --host 0.0.0.0
   ```
   Or use the helper script: `./scripts/run_llama_server.sh ./models/your-model.gguf`

4. **Configure** in `.env`:
   ```
   LLAMA_CPP_BASE_URL=http://localhost:8080/v1
   ```

5. **Use** the "Llama.cpp (Local)" provider in the Multi-SDK tab.

