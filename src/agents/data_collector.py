"""Data Collector Agent - Validates ticker and provides basic info.

In the multi-agent architecture, specialist agents fetch their own data
via MCP tools. This agent only validates the ticker symbol exists.
"""

from typing import Dict, Any
from src.agents.base_agent import BaseAgent
from src.mcp.yfinance_client import get_yfinance_client
from src.models.prompts import DATA_COLLECTOR_PROMPT


class DataCollectorAgent(BaseAgent):
    """
    Agent responsible for validating ticker symbols.

    Each specialist agent independently fetches data it needs via MCP tools.
    This agent only verifies the ticker is valid before analysis begins.
    """

    def __init__(self, openai_client, db_client):
        """Initialize Data Collector Agent."""
        super().__init__("data_collector", openai_client, db_client)
        self.yfinance_client = get_yfinance_client()

    async def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate ticker exists. Specialist agents fetch their own data via MCP.

        Args:
            inputs: Dictionary with 'ticker' key

        Returns:
            Validation result with basic company info
        """
        ticker = inputs.get("ticker", "").upper()
        if not ticker:
            raise ValueError("Ticker is required")

        self.logger.info(f"Validating ticker {ticker}")

        try:
            # Quick validation using existing YahooFinanceMCPClient
            info = await self.yfinance_client.get_ticker_info(ticker)

            if info and info.get("name"):
                return {
                    "ticker": ticker,
                    "company_name": info.get("name", ticker),
                    "sector": info.get("sector", "Unknown"),
                    "valid": True,
                    "confidence": 0.9,
                    "reasoning": f"Validated ticker {ticker}: {info.get('name', ticker)}"
                    # No more "data" key -- specialists fetch their own
                }
            else:
                return {
                    "ticker": ticker,
                    "valid": False,
                    "error": "Ticker not found or invalid",
                    "confidence": 0.0,
                    "reasoning": f"Ticker {ticker} could not be validated"
                }

        except Exception as e:
            self.logger.error(f"Failed to validate ticker {ticker}: {e}")
            return {
                "ticker": ticker,
                "valid": False,
                "error": str(e),
                "confidence": 0.0,
                "reasoning": f"Ticker validation failed: {str(e)}"
            }
