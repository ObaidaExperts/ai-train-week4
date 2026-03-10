#!/usr/bin/env bash
# Run llama-cpp-python OpenAI-compatible server.
# Usage: ./scripts/run_llama_server.sh [path/to/model.gguf]
# Default port: 8080 (to avoid conflict with app on 8000)
# Then set LLAMA_CPP_BASE_URL=http://localhost:8080/v1 in .env

MODEL_PATH="${1:-}"
if [ -z "$MODEL_PATH" ]; then
  echo "Usage: $0 <path/to/model.gguf>"
  echo ""
  echo "Download a GGUF model first, e.g.:"
  echo "  huggingface-cli download TheBloke/Llama-2-7B-Chat-GGUF --local-dir ./models"
  echo "  ./scripts/run_llama_server.sh ./models/llama-2-7b-chat.Q4_K_M.gguf"
  exit 1
fi

python3 -m llama_cpp.server --model "$MODEL_PATH" --port 8080 --host 0.0.0.0
