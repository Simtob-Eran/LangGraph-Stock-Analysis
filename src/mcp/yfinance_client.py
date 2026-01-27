"""
Enhanced MCP client for Yahoo Finance data integration.

This client uses the official MCP library with proper session handling.
It prioritizes MCP servers with automatic fallback to direct yfinance.

MCP Priority Strategy:
1. Try MCP server with proper session (streamablehttp_client)
2. Fallback to direct yfinance library if MCP fails
"""

import json
import asyncio
from typing import Dict, Any, List, Optional
from pathlib import Path
from contextlib import asynccontextmanager
import yfinance as yf
from datetime import datetime
from src.utils.logger import setup_logger

# MCP imports
try:
    from mcp import ClientSession
    from mcp.client.streamable_http import streamablehttp_client
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    print("⚠️ MCP library not installed. Using direct yfinance only.")

logger = setup_logger("mcp.yfinance")


class YahooFinanceMCPClient:
    """
    Yahoo Finance client with MCP priority and fallback.

    Uses proper MCP session handling with streamablehttp_client.
    Falls back to direct yfinance if MCP is unavailable.
    """

    def __init__(self):
        """Initialize Yahoo Finance MCP client."""
        self.logger = logger
        self.mcp_config = self._load_mcp_config()
        self.fallback_enabled = self.mcp_config.get("fallbackStrategy", {}).get("enabled", True)
        self._session = None
        self._tools = None
        self._tools_map = {}

        # Get MCP server config
        servers = self.mcp_config.get("mcpServers", {})
        self.mcp_url = None
        self.mcp_headers = {}

        for name, config in servers.items():
            if config.get("enabled", True):
                self.mcp_url = config.get("url")
                self.mcp_headers = config.get("headers", {})
                self.mcp_name = name
                break

        print(f"\n{'='*60}")
        print(f"🚀 MCP CLIENT INITIALIZED")
        print(f"{'='*60}")
        print(f"MCP Available: {MCP_AVAILABLE}")
        print(f"MCP URL: {self.mcp_url}")
        print(f"Headers: {self.mcp_headers}")
        print(f"Fallback enabled: {self.fallback_enabled}")
        print(f"{'='*60}\n")

        self.logger.info(f"Initialized MCP client - URL: {self.mcp_url}")

    def _load_mcp_config(self) -> Dict[str, Any]:
        """Load MCP configuration from config file."""
        try:
            config_path = Path(__file__).parent.parent.parent / "config" / "mcp_config.json"
            with open(config_path, "r") as f:
                return json.load(f)
        except Exception as e:
            self.logger.warning(f"Failed to load MCP config: {e}, using defaults")
            return {"mcpServers": {}, "fallbackStrategy": {"enabled": True}}

    @asynccontextmanager
    async def _get_mcp_session(self):
        """Create and manage MCP session with proper streaming."""
        if not MCP_AVAILABLE or not self.mcp_url:
            yield None
            return

        try:
            print(f"\n{'='*60}")
            print(f"🔌 CONNECTING TO MCP SERVER")
            print(f"{'='*60}")
            print(f"URL: {self.mcp_url}")
            print(f"Headers: {self.mcp_headers}")

            async with streamablehttp_client(
                self.mcp_url,
                headers=self.mcp_headers
            ) as (read, write, _):
                async with ClientSession(read, write) as session:
                    print("🔄 Initializing session...")
                    await session.initialize()
                    print("✅ Session initialized!")

                    # List available tools
                    tools_result = await session.list_tools()
                    tools = tools_result.tools if hasattr(tools_result, 'tools') else []

                    print(f"📦 Available tools: {len(tools)}")
                    for tool in tools:
                        tool_name = tool.name if hasattr(tool, 'name') else str(tool)
                        print(f"  • {tool_name}")
                        self._tools_map[tool_name] = tool

                    print(f"{'='*60}\n")

                    yield session

        except Exception as e:
            print(f"❌ MCP Connection Error: {e}")
            print(f"{'='*60}\n")
            self.logger.error(f"MCP session error: {e}")
            yield None

    async def _call_mcp_tool(self, session: ClientSession, tool_name: str, arguments: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Call an MCP tool and return the result."""
        try:
            print(f"🔧 Calling tool: {tool_name}")
            print(f"   Arguments: {arguments}")

            result = await session.call_tool(tool_name, arguments)

            # Parse the result
            if hasattr(result, 'content') and result.content:
                content = result.content[0]
                if hasattr(content, 'text'):
                    data = json.loads(content.text)
                    print(f"✅ Tool returned data")
                    return data
                elif hasattr(content, 'data'):
                    return content.data

            return result

        except Exception as e:
            self.logger.error(f"MCP tool call failed: {e}")
            print(f"❌ Tool call failed: {e}")
            return None

    async def _fetch_with_fallback(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        fallback_func
    ) -> Dict[str, Any]:
        """Fetch data via MCP with fallback to direct yfinance."""

        # Try MCP first
        if MCP_AVAILABLE and self.mcp_url:
            try:
                async with self._get_mcp_session() as session:
                    if session:
                        result = await self._call_mcp_tool(session, tool_name, arguments)
                        if result:
                            if isinstance(result, dict):
                                result["source"] = "mcp"
                            return result
            except Exception as e:
                self.logger.error(f"MCP failed: {e}")

        # Fallback to direct yfinance
        if self.fallback_enabled:
            print(f"\n{'='*60}")
            print(f"⚠️ MCP UNAVAILABLE - USING DIRECT YFINANCE")
            print(f"{'='*60}\n")
            self.logger.info(f"Using direct yfinance fallback for {tool_name}")
            return await fallback_func()
        else:
            raise Exception("MCP failed and fallback is disabled")

    async def get_ticker_info(self, symbol: str) -> Dict[str, Any]:
        """Get basic ticker information."""

        async def fallback():
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
                    "exchange": info.get("exchange", "Unknown"),
                    "source": "direct_yfinance"
                }
            except Exception as e:
                self.logger.error(f"Direct yfinance fallback failed: {e}")
                return {
                    "symbol": symbol,
                    "name": symbol,
                    "sector": "Unknown",
                    "industry": "Unknown",
                    "market_cap": 0,
                    "error": str(e),
                    "source": "error"
                }

        return await self._fetch_with_fallback(
            "get_ticker_info",
            {"symbol": symbol},
            fallback
        )

    async def get_ticker_historical(
        self,
        symbol: str,
        period: str = "2y",
        interval: str = "1d"
    ) -> Dict[str, Any]:
        """Get historical price data."""

        async def fallback():
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period=period, interval=interval)

                if hist.empty:
                    return {
                        "symbol": symbol,
                        "data": [],
                        "error": "No historical data available",
                        "source": "direct_yfinance"
                    }

                info = ticker.info
                current_price = info.get("currentPrice") or info.get("regularMarketPrice", 0)

                return {
                    "symbol": symbol,
                    "current_price": current_price,
                    "52_week_high": info.get("fiftyTwoWeekHigh"),
                    "52_week_low": info.get("fiftyTwoWeekLow"),
                    "historical_data": hist.to_dict(orient="index"),
                    "period": period,
                    "interval": interval,
                    "source": "direct_yfinance"
                }
            except Exception as e:
                self.logger.error(f"Direct yfinance historical fallback failed: {e}")
                return {
                    "symbol": symbol,
                    "data": [],
                    "error": str(e),
                    "source": "error"
                }

        return await self._fetch_with_fallback(
            "get_ticker_historical",
            {"symbol": symbol, "period": period, "interval": interval},
            fallback
        )

    async def get_ticker_news(self, symbol: str, count: int = 10) -> List[Dict[str, Any]]:
        """Get recent news for a ticker."""

        async def fallback():
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
                        "summary": item.get("summary", ""),
                        "source_type": "direct_yfinance"
                    })

                return articles
            except Exception as e:
                self.logger.error(f"Direct yfinance news fallback failed: {e}")
                return []

        result = await self._fetch_with_fallback(
            "get_ticker_news",
            {"symbol": symbol, "count": count},
            fallback
        )

        if isinstance(result, dict):
            return result.get("articles", [])
        return result if isinstance(result, list) else []

    async def get_balance_sheet(self, symbol: str) -> Dict[str, Any]:
        """Get balance sheet data."""

        async def fallback():
            try:
                ticker = yf.Ticker(symbol)
                balance_sheet = ticker.balance_sheet

                if balance_sheet.empty:
                    return {
                        "symbol": symbol,
                        "data": {},
                        "error": "No balance sheet data",
                        "source": "direct_yfinance"
                    }

                return {
                    "symbol": symbol,
                    "data": balance_sheet.to_dict(),
                    "quarters": list(balance_sheet.columns.astype(str)),
                    "source": "direct_yfinance"
                }
            except Exception as e:
                self.logger.error(f"Direct yfinance balance sheet fallback failed: {e}")
                return {"symbol": symbol, "data": {}, "error": str(e), "source": "error"}

        return await self._fetch_with_fallback(
            "get_balance_sheet",
            {"symbol": symbol},
            fallback
        )

    async def get_income_statement(self, symbol: str) -> Dict[str, Any]:
        """Get income statement data."""

        async def fallback():
            try:
                ticker = yf.Ticker(symbol)
                income_stmt = ticker.income_stmt

                if income_stmt.empty:
                    return {
                        "symbol": symbol,
                        "data": {},
                        "error": "No income statement data",
                        "source": "direct_yfinance"
                    }

                return {
                    "symbol": symbol,
                    "data": income_stmt.to_dict(),
                    "quarters": list(income_stmt.columns.astype(str)),
                    "source": "direct_yfinance"
                }
            except Exception as e:
                self.logger.error(f"Direct yfinance income statement fallback failed: {e}")
                return {"symbol": symbol, "data": {}, "error": str(e), "source": "error"}

        return await self._fetch_with_fallback(
            "get_income_statement",
            {"symbol": symbol},
            fallback
        )

    async def get_cash_flow(self, symbol: str) -> Dict[str, Any]:
        """Get cash flow statement data."""

        async def fallback():
            try:
                ticker = yf.Ticker(symbol)
                cash_flow = ticker.cash_flow

                if cash_flow.empty:
                    return {
                        "symbol": symbol,
                        "data": {},
                        "error": "No cash flow data",
                        "source": "direct_yfinance"
                    }

                return {
                    "symbol": symbol,
                    "data": cash_flow.to_dict(),
                    "quarters": list(cash_flow.columns.astype(str)),
                    "source": "direct_yfinance"
                }
            except Exception as e:
                self.logger.error(f"Direct yfinance cash flow fallback failed: {e}")
                return {"symbol": symbol, "data": {}, "error": str(e), "source": "error"}

        return await self._fetch_with_fallback(
            "get_cash_flow",
            {"symbol": symbol},
            fallback
        )

    async def get_comprehensive_data(self, symbol: str) -> Dict[str, Any]:
        """Get all available data for a ticker in one call."""
        self.logger.info(f"Fetching comprehensive data for {symbol}")
        self.logger.info(f"Priority: MCP server → Direct yfinance")

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

        # Log data sources
        sources = []
        if isinstance(info, dict):
            sources.append(info.get("source", "unknown"))
        if isinstance(historical, dict):
            sources.append(historical.get("source", "unknown"))
        self.logger.info(f"Data sources: {', '.join(set(sources))}")

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
