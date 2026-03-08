import os
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    """Application settings using Pydantic Settings."""
    
    # API Keys
    OPENAI_API_KEY: str | None = None
    ANTHROPIC_API_KEY: str | None = None
    
    # Application Config
    APP_NAME: str = "Tokenization Chat Analysis API"
    RESULTS_FILE: str = "experiment_results.csv"
    
    # Environment variables configuration
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

# Global settings instance
settings = Settings()
