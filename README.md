# Tokenization and Context Limits Experiment

A professional Python project to experiment with tokenization and context limits using FastAPI, OpenAI, and Gemini.

## Features
- **Token Analysis**: Count tokens for OpenAI and Gemini models.
- **Cost Estimation**: Real-time cost calculation based on current pricing.
- **Context Monitoring**: Track usage against model-specific context limits.
- **Results Logging**: Persistent logging of all experiments to CSV.
- **Modular Architecture**: Clean separation of concerns (API, Services, Core).

## Project Structure
- `app/api/`: API routers, schemas, and middleware.
- `app/core/`: Centralized configuration and domain models.
- `app/services/`: Business logic and data persistence (repository).
- `tests/`: Automated test suite.

## How to Run

1. **Setup Environment**:
   ```bash
   cp .env.example .env
   # Add your OPENAI_API_KEY to .env
   ```

2. **Start the server**:
   ```bash
   poetry run python -m app.main
   ```

## API Usage

- **Chat & Analyze**: `POST /chat`
  ```bash
  curl -X POST "http://localhost:8000/chat" \
       -H "Content-Type: application/json" \
       -d '{"prompt": "Hello world", "model": "gpt-4o"}'
  ```

- **Get Results**: `GET /results`
  ```bash
  curl http://localhost:8000/results
  ```

- **Health Check**: `GET /health`

