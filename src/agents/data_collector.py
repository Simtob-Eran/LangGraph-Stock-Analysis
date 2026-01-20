"""Data Collector Agent - Fetches comprehensive stock data."""

from typing import Dict, Any
from datetime import datetime
from src.agents.base_agent import BaseAgent
from src.mcp.yfinance_client import get_yfinance_client
from src.models.schemas import CollectedData, BasicInfo, PriceData, Financials, NewsArticle
from src.models.prompts import DATA_COLLECTOR_PROMPT


class DataCollectorAgent(BaseAgent):
    """
    Agent responsible for collecting comprehensive stock data.

    This agent interfaces with Yahoo Finance to gather all necessary
    data including company info, prices, financials, and news.
    """

    def __init__(self, openai_client, db_client):
        """Initialize Data Collector Agent."""
        super().__init__("data_collector", openai_client, db_client)
        self.yfinance_client = get_yfinance_client()

    async def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute data collection for a ticker.

        Args:
            inputs: Dictionary with 'ticker' key

        Returns:
            Collected data dictionary
        """
        ticker = inputs.get("ticker", "").upper()
        if not ticker:
            raise ValueError("Ticker is required")

        self.logger.info(f"Collecting data for {ticker}")

        # Check cache first
        cached_data = self.db.get_cached_data(ticker)
        if cached_data:
            self.logger.info(f"Using cached data for {ticker}")
            return {
                "ticker": ticker,
                "data": cached_data,
                "from_cache": True,
                "confidence": 1.0,
                "reasoning": "Data retrieved from cache"
            }

        # Fetch comprehensive data
        try:
            raw_data = await self.yfinance_client.get_comprehensive_data(ticker)

            # Structure the data
            structured_data = self._structure_data(raw_data)

            # Cache the data for 24 hours
            self.db.cache_data(ticker, structured_data, hours=24)

            return {
                "ticker": ticker,
                "data": structured_data,
                "from_cache": False,
                "confidence": 0.9,
                "reasoning": "Successfully collected comprehensive data from Yahoo Finance"
            }

        except Exception as e:
            self.logger.error(f"Failed to collect data for {ticker}: {e}")
            return {
                "ticker": ticker,
                "data": None,
                "error": str(e),
                "confidence": 0.0,
                "reasoning": f"Data collection failed: {str(e)}"
            }

    def _structure_data(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Structure raw Yahoo Finance data into organized format.

        Args:
            raw_data: Raw data from Yahoo Finance

        Returns:
            Structured data dictionary
        """
        info = raw_data.get("info", {})
        historical = raw_data.get("historical", {})
        news = raw_data.get("news", [])
        balance_sheet = raw_data.get("balance_sheet", {}).get("data", {})
        income_stmt = raw_data.get("income_statement", {}).get("data", {})
        cash_flow = raw_data.get("cash_flow", {}).get("data", {})

        # Structure basic info
        basic_info = {
            "name": info.get("name", raw_data.get("symbol")),
            "sector": info.get("sector", "Unknown"),
            "industry": info.get("industry", "Unknown"),
            "market_cap": info.get("market_cap", 0),
            "employees": info.get("employees"),
            "website": info.get("website"),
            "description": info.get("description")
        }

        # Structure price data
        price_data = {
            "current_price": historical.get("current_price", 0),
            "52_week_high": historical.get("52_week_high"),
            "52_week_low": historical.get("52_week_low"),
            "historical_data": historical.get("historical_data", {})
        }

        # Structure financials
        financials = {
            "income_statement": income_stmt,
            "balance_sheet": balance_sheet,
            "cash_flow": cash_flow
        }

        # Structure news
        news_articles = []
        for article in news:
            news_articles.append({
                "headline": article.get("headline", ""),
                "source": article.get("source"),
                "published": article.get("published"),
                "url": article.get("url"),
                "summary": article.get("summary")
            })

        return {
            "ticker": raw_data.get("symbol"),
            "timestamp": raw_data.get("timestamp", datetime.now().isoformat()),
            "basic_info": basic_info,
            "price_data": price_data,
            "financials": financials,
            "news": news_articles
        }
