"""Logging utilities for the stock analysis system."""

import logging
import sys
from pathlib import Path
from datetime import datetime
from config.settings import settings


def setup_logger(name: str, log_file: str = None) -> logging.Logger:
    """
    Set up a logger with console and file handlers.

    Args:
        name: Logger name (typically the module or agent name)
        log_file: Optional specific log file path

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, settings.LOG_LEVEL))

    # Avoid duplicate handlers
    if logger.handlers:
        return logger

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, settings.LOG_LEVEL))
    console_format = logging.Formatter(
        '[%(levelname)s] [%(name)s] %(message)s'
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)

    # File handler
    if log_file is None:
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        log_file = log_dir / f"{datetime.now().strftime('%Y%m%d')}_stock_analysis.log"

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter(
        '%(asctime)s - [%(levelname)s] - %(name)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_format)
    logger.addHandler(file_handler)

    return logger


class AgentLogger:
    """
    Specialized logger for agent operations with structured logging.
    """

    def __init__(self, agent_name: str):
        """Initialize agent logger."""
        self.agent_name = agent_name
        self.logger = setup_logger(f"agent.{agent_name}")

    def log_execution_start(self, ticker: str, inputs: dict):
        """Log the start of an agent execution."""
        self.logger.info(f"[{self.agent_name}] Starting analysis for {ticker}")
        self.logger.debug(f"Inputs: {inputs}")

    def log_execution_complete(self, ticker: str, execution_time: float, confidence: float):
        """Log successful completion of an agent execution."""
        self.logger.info(
            f"[{self.agent_name}] Completed analysis for {ticker} "
            f"in {execution_time:.2f}s (confidence: {confidence:.2%})"
        )

    def log_execution_error(self, ticker: str, error: Exception):
        """Log an error during agent execution."""
        self.logger.error(
            f"[{self.agent_name}] Error analyzing {ticker}: {str(error)}",
            exc_info=True
        )

    def log_llm_call(self, prompt_length: int, response_length: int):
        """Log LLM API call details."""
        self.logger.debug(
            f"LLM call - Prompt: {prompt_length} chars, "
            f"Response: {response_length} chars"
        )

    def info(self, message: str):
        """Log info message."""
        self.logger.info(f"[{self.agent_name}] {message}")

    def debug(self, message: str):
        """Log debug message."""
        self.logger.debug(f"[{self.agent_name}] {message}")

    def warning(self, message: str):
        """Log warning message."""
        self.logger.warning(f"[{self.agent_name}] {message}")

    def error(self, message: str, exc_info=False):
        """Log error message."""
        self.logger.error(f"[{self.agent_name}] {message}", exc_info=exc_info)
