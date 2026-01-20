"""MCP client for Yahoo Finance data integration."""

import json
import asyncio
from typing import Dict, Any, List, Optional
import yfinance as yf
from datetime import datetime
from src.utils.logger import setup_logger

logger = setup_logger("mcp.yfinance")


class YahooFinanceMCPClient:
    """
    Client for Yahoo Finance data using yfinance library.

    Note: This is a direct implementation using yfinance rather than MCP protocol,
    as it's more reliable and doesn't require external MCP server setup.
    """

    def __init__(self):
        """Initialize Yahoo Finance client."""
        self.logger = logger

    async def get_ticker_info(self, symbol: str) -> Dict[str, Any]:
        """
        Get basic ticker information.

        Args:
            symbol: Stock ticker symbol

        Returns:
            Dictionary containing ticker info
        """
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info

            return {
                "symbol": symbol,
                "name": info.get("longName", symbol),
                "sector": info.get("sector", "Unknown"),
                "industry": info.get("industry", "Unknown"),
                "market_cap": info.get("marketCap", 0),
                "employees": info.get("fullTimeEmployees"),
                "website": info.get("website"),
                "description": info.get("longBusinessSummary"),
                "currency": info.get("currency", "USD"),
                "exchange": info.get("exchange", "Unknown")
            }
        except Exception as e:
            self.logger.error(f"Error fetching ticker info for {symbol}: {e}")
            return {
                "symbol": symbol,
                "name": symbol,
                "sector": "Unknown",
                "industry": "Unknown",
                "market_cap": 0,
                "error": str(e)
            }

    async def get_ticker_historical(
        self,
        symbol: str,
        period: str = "2y",
        interval: str = "1d"
    ) -> Dict[str, Any]:
        """
        Get historical price data.

        Args:
            symbol: Stock ticker symbol
            period: Data period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)

        Returns:
            Dictionary containing historical data
        """
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=period, interval=interval)

            if hist.empty:
                return {
                    "symbol": symbol,
                    "data": [],
                    "error": "No historical data available"
                }

            # Get current price info
            info = ticker.info
            current_price = info.get("currentPrice") or info.get("regularMarketPrice", 0)

            return {
                "symbol": symbol,
                "current_price": current_price,
                "52_week_high": info.get("fiftyTwoWeekHigh"),
                "52_week_low": info.get("fiftyTwoWeekLow"),
                "historical_data": hist.to_dict(orient="index"),
                "period": period,
                "interval": interval
            }
        except Exception as e:
            self.logger.error(f"Error fetching historical data for {symbol}: {e}")
            return {
                "symbol": symbol,
                "data": [],
                "error": str(e)
            }

    async def get_ticker_news(self, symbol: str, count: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent news for a ticker.

        Args:
            symbol: Stock ticker symbol
            count: Number of news items to retrieve

        Returns:
            List of news articles
        """
        try:
            ticker = yf.Ticker(symbol)
            news = ticker.news

            articles = []
            for item in news[:count]:
                articles.append({
                    "headline": item.get("title", ""),
                    "source": item.get("publisher", "Unknown"),
                    "published": item.get("providerPublishTime"),
                    "url": item.get("link", ""),
                    "summary": item.get("summary", "")
                })

            return articles
        except Exception as e:
            self.logger.error(f"Error fetching news for {symbol}: {e}")
            return []

    async def get_balance_sheet(self, symbol: str) -> Dict[str, Any]:
        """
        Get balance sheet data.

        Args:
            symbol: Stock ticker symbol

        Returns:
            Balance sheet data
        """
        try:
            ticker = yf.Ticker(symbol)
            balance_sheet = ticker.balance_sheet

            if balance_sheet.empty:
                return {"symbol": symbol, "data": {}, "error": "No balance sheet data"}

            return {
                "symbol": symbol,
                "data": balance_sheet.to_dict(),
                "quarters": list(balance_sheet.columns.astype(str))
            }
        except Exception as e:
            self.logger.error(f"Error fetching balance sheet for {symbol}: {e}")
            return {"symbol": symbol, "data": {}, "error": str(e)}

    async def get_income_statement(self, symbol: str) -> Dict[str, Any]:
        """
        Get income statement data.

        Args:
            symbol: Stock ticker symbol

        Returns:
            Income statement data
        """
        try:
            ticker = yf.Ticker(symbol)
            income_stmt = ticker.income_stmt

            if income_stmt.empty:
                return {"symbol": symbol, "data": {}, "error": "No income statement data"}

            return {
                "symbol": symbol,
                "data": income_stmt.to_dict(),
                "quarters": list(income_stmt.columns.astype(str))
            }
        except Exception as e:
            self.logger.error(f"Error fetching income statement for {symbol}: {e}")
            return {"symbol": symbol, "data": {}, "error": str(e)}

    async def get_cash_flow(self, symbol: str) -> Dict[str, Any]:
        """
        Get cash flow statement data.

        Args:
            symbol: Stock ticker symbol

        Returns:
            Cash flow data
        """
        try:
            ticker = yf.Ticker(symbol)
            cash_flow = ticker.cash_flow

            if cash_flow.empty:
                return {"symbol": symbol, "data": {}, "error": "No cash flow data"}

            return {
                "symbol": symbol,
                "data": cash_flow.to_dict(),
                "quarters": list(cash_flow.columns.astype(str))
            }
        except Exception as e:
            self.logger.error(f"Error fetching cash flow for {symbol}: {e}")
            return {"symbol": symbol, "data": {}, "error": str(e)}

    async def get_comprehensive_data(self, symbol: str) -> Dict[str, Any]:
        """
        Get all available data for a ticker in one call.

        Args:
            symbol: Stock ticker symbol

        Returns:
            Comprehensive data dictionary
        """
        self.logger.info(f"Fetching comprehensive data for {symbol}")

        # Fetch all data concurrently
        results = await asyncio.gather(
            self.get_ticker_info(symbol),
            self.get_ticker_historical(symbol),
            self.get_ticker_news(symbol),
            self.get_balance_sheet(symbol),
            self.get_income_statement(symbol),
            self.get_cash_flow(symbol),
            return_exceptions=True
        )

        info, historical, news, balance_sheet, income_stmt, cash_flow = results

        # Handle any exceptions
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                self.logger.error(f"Error in data fetch {i}: {result}")

        return {
            "symbol": symbol,
            "timestamp": datetime.now().isoformat(),
            "info": info if not isinstance(info, Exception) else {},
            "historical": historical if not isinstance(historical, Exception) else {},
            "news": news if not isinstance(news, Exception) else [],
            "balance_sheet": balance_sheet if not isinstance(balance_sheet, Exception) else {},
            "income_statement": income_stmt if not isinstance(income_stmt, Exception) else {},
            "cash_flow": cash_flow if not isinstance(cash_flow, Exception) else {}
        }


# Singleton instance
_client = None


def get_yfinance_client() -> YahooFinanceMCPClient:
    """Get or create Yahoo Finance MCP client singleton."""
    global _client
    if _client is None:
        _client = YahooFinanceMCPClient()
    return _client
