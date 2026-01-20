"""Configuration settings loaded from environment variables."""

from pydantic_settings import BaseSettings
from typing import Optional
import os


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # OpenAI Configuration
    OPENAI_API_KEY: str
    OPENAI_MODEL: str = "gpt-4o"

    # Application Settings
    DATABASE_PATH: str = "./data/analysis.db"
    LOG_LEVEL: str = "INFO"
    MAX_PARALLEL_TASKS: int = 5

    # MCP Settings
    MCP_YFINANCE_ENABLED: bool = True

    class Config:
        """Pydantic configuration."""
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


# Global settings instance
settings = Settings()
