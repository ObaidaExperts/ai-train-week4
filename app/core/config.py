import os
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    """Application settings using Pydantic Settings."""
    
    # API Keys
    OPENAI_API_KEY: str | None = None
    ANTHROPIC_API_KEY: str | None = None
    GOOGLE_API_KEY: str | None = None

    # Local model (vLLM / OpenAI-compatible server)
    VLLM_BASE_URL: str | None = None  # e.g. http://localhost:8001/v1
    VLLM_MODEL: str = "default"  # Model name on vLLM server

    # Llama.cpp (llama-cpp-python server, OpenAI-compatible)
    LLAMA_CPP_BASE_URL: str | None = None  # e.g. http://localhost:8080/v1 (port 8080 to avoid conflict with app on 8000)

    # Application Config
    APP_NAME: str = "Tokenization Chat Analysis API"
    RESULTS_FILE: str = "experiment_results.csv"
    
    # Environment variables configuration
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

# Global settings instance
settings = Settings()
