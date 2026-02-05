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
    # Generic OAuth settings
    OAUTH_REDIRECT_URI: str = "https://cbg-obot.com/"
    OAUTH_SCOPE: str = "user repo"
    OAUTH_CLIENT_NAME: str = "Stock Analysis MCP Agent"

    # GitHub OAuth specific settings (alternative naming)
    MCP_OAUTH_GITHUB_URL: Optional[str] = None
    MCP_OAUTH_GITHUB_REDIRECT_URI: Optional[str] = None
    MCP_OAUTH_GITHUB_SCOPE: Optional[str] = None

    def get_oauth_url(self) -> Optional[str]:
        """Get OAuth MCP URL - prefer GitHub-specific, fallback to generic."""
        return self.MCP_OAUTH_GITHUB_URL or self.MCP_URL

    def get_oauth_redirect_uri(self) -> str:
        """Get OAuth redirect URI - prefer GitHub-specific, fallback to generic."""
        return self.MCP_OAUTH_GITHUB_REDIRECT_URI or self.OAUTH_REDIRECT_URI

    def get_oauth_scope(self) -> str:
        """Get OAuth scope - prefer GitHub-specific, fallback to generic."""
        return self.MCP_OAUTH_GITHUB_SCOPE or self.OAUTH_SCOPE

    class Config:
        """Pydantic configuration."""
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


# Global settings instance
settings = Settings()
