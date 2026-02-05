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
    MCP_URL: Optional[str] = None
    MCP_YFINANCE_ENABLED: bool = True

    # OAuth Settings (for MCP authentication)
    OAUTH_REDIRECT_URI: str = "https://cbg-obot.com/"
    OAUTH_SCOPE: str = "user repo"
    OAUTH_CLIENT_NAME: str = "Stock Analysis MCP Agent"

    class Config:
        """Pydantic configuration."""
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


# Global settings instance
settings = Settings()
