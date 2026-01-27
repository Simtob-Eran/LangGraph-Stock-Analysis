"""
Enhanced MCP client for Yahoo Finance data integration.

This client uses the official MCP library with proper session handling.
It discovers available tools dynamically and maps them to our internal methods.

MCP Priority Strategy:
1. Try MCP server with proper session (streamablehttp_client)
2. Fallback to direct yfinance library if MCP fails
"""

import json
import asyncio
import os
from typing import Dict, Any, List, Optional, Tuple
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


# Tool mapping: our internal names -> possible MCP tool names
TOOL_MAPPING = {
    "get_ticker_info": ["get_ticker_info", "get_stock_info"],
    "get_ticker_historical": ["get_price_history", "get_historical_stock_prices"],
    "get_ticker_news": ["get_ticker_news", "get_yahoo_finance_news"],
    "get_balance_sheet": ["get_financial_statement"],
    "get_income_statement": ["get_financial_statement"],
    "get_cash_flow": ["get_financial_statement"],
}

# Argument mapping: our argument names -> possible MCP argument names
ARG_MAPPING = {
    "symbol": ["ticker", "symbol", "symbols"],
    "period": ["period", "range"],
    "interval": ["interval"],
    "count": ["max_items", "count", "limit", "num_articles"],
}


class YahooFinanceMCPClient:
    """
    Yahoo Finance client with MCP priority and fallback.

    Uses proper MCP session handling with streamablehttp_client.
    Discovers available tools dynamically and maps arguments correctly.
    """

    def __init__(self):
        """Initialize Yahoo Finance MCP client."""
        self.logger = logger
        self.mcp_config = self._load_mcp_config()
        self.fallback_enabled = self.mcp_config.get("fallbackStrategy", {}).get("enabled", True)

        # Tool discovery cache
        self._available_tools: Dict[str, Any] = {}
        self._tools_discovered = False

        # Priority: Environment variable > Config file
        self.mcp_url = os.getenv("MCP_URL")
        self.mcp_headers = {}
        self.mcp_name = "env"

        # If no env var, try config file
        if not self.mcp_url:
            servers = self.mcp_config.get("mcpServers", {})
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
        print(f"Source: {'Environment Variable' if os.getenv('MCP_URL') else 'Config File'}")
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

    def _find_mcp_tool(self, our_method: str) -> Optional[str]:
        """Find the correct MCP tool name for our method."""
        possible_names = TOOL_MAPPING.get(our_method, [our_method])
        for name in possible_names:
            if name in self._available_tools:
                return name
        return None

    def _map_arguments(self, tool_name: str, our_args: Dict[str, Any]) -> Dict[str, Any]:
        """Map our argument names to the MCP tool's expected argument names."""
        if tool_name not in self._available_tools:
            return our_args

        tool_info = self._available_tools[tool_name]
        input_schema = tool_info.get("inputSchema", {})
        expected_props = input_schema.get("properties", {})
        expected_keys = set(expected_props.keys())

        mapped_args = {}

        for our_key, our_value in our_args.items():
            # Try to find the correct argument name
            possible_names = ARG_MAPPING.get(our_key, [our_key])
            matched = False

            for possible_name in possible_names:
                if possible_name in expected_keys:
                    mapped_args[possible_name] = our_value
                    matched = True
                    break

            if not matched:
                # Keep original if no mapping found
                mapped_args[our_key] = our_value

        return mapped_args

    @asynccontextmanager
    async def _get_mcp_session(self):
        """Create and manage MCP session with proper streaming."""
        if not MCP_AVAILABLE or not self.mcp_url:
            yield None
            return

        try:
            async with streamablehttp_client(
                self.mcp_url,
                headers=self.mcp_headers
            ) as (read, write, _):
                async with ClientSession(read, write) as session:
                    await session.initialize()

                    # Discover tools if not already done
                    if not self._tools_discovered:
                        tools_result = await session.list_tools()
                        tools = tools_result.tools if hasattr(tools_result, 'tools') else []

                        print(f"\n{'='*60}")
                        print(f"📦 DISCOVERED {len(tools)} MCP TOOLS:")
                        print(f"{'='*60}")

                        for tool in tools:
                            tool_name = tool.name if hasattr(tool, 'name') else str(tool)
                            tool_desc = tool.description if hasattr(tool, 'description') else ""
                            input_schema = tool.inputSchema if hasattr(tool, 'inputSchema') else {}

                            self._available_tools[tool_name] = {
                                "name": tool_name,
                                "description": tool_desc,
                                "inputSchema": input_schema
                            }

                            # Show tool info
                            props = input_schema.get("properties", {})
                            args_list = list(props.keys())
                            print(f"  • {tool_name}")
                            if args_list:
                                print(f"    Args: {', '.join(args_list)}")

                        print(f"{'='*60}\n")
                        self._tools_discovered = True

                    yield session

        except Exception as e:
            print(f"❌ MCP Connection Error: {e}")
            self.logger.error(f"MCP session error: {e}")
            yield None

    async def _call_mcp_tool(
        self,
        session: ClientSession,
        our_method: str,
        our_args: Dict[str, Any],
        extra_args: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """
        Call an MCP tool with automatic tool and argument mapping.

        Returns:
            Tuple of (success, data)
        """
        # Find the correct MCP tool
        mcp_tool = self._find_mcp_tool(our_method)

        if not mcp_tool:
            print(f"⚠️ No MCP tool found for: {our_method}")
            self.logger.warning(f"No MCP tool mapping for {our_method}")
            return False, None

        # Map arguments
        mapped_args = self._map_arguments(mcp_tool, our_args)

        # Add extra arguments if provided (e.g., statement_type for financial_statement)
        if extra_args:
            mapped_args.update(extra_args)

        try:
            print(f"🔧 Calling: {mcp_tool}")
            print(f"   Args: {mapped_args}")

            result = await session.call_tool(mcp_tool, mapped_args)

            # Parse the result
            if hasattr(result, 'content') and result.content:
                content = result.content[0]
                if hasattr(content, 'text'):
                    try:
                        data = json.loads(content.text)
                        print(f"✅ Success!")
                        return True, data
                    except json.JSONDecodeError:
                        return True, {"raw_text": content.text}
                elif hasattr(content, 'data'):
                    return True, content.data

            return True, result

        except Exception as e:
            error_msg = str(e)
            print(f"❌ Tool call failed: {error_msg}")
            self.logger.error(f"MCP tool call failed: {e}")
            return False, None

    async def _fetch_with_fallback(
        self,
        our_method: str,
        our_args: Dict[str, Any],
        fallback_func,
        extra_mcp_args: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Fetch data via MCP with fallback to direct yfinance."""

        # Try MCP first
        if MCP_AVAILABLE and self.mcp_url:
            try:
                async with self._get_mcp_session() as session:
                    if session:
                        success, result = await self._call_mcp_tool(
                            session, our_method, our_args, extra_mcp_args
                        )
                        if success and result:
                            if isinstance(result, dict):
                                result["source"] = "mcp"
                            return result
            except Exception as e:
                self.logger.error(f"MCP failed: {e}")

        # Fallback to direct yfinance
        if self.fallback_enabled:
            print(f"⚠️ Using fallback for: {our_method}")
            self.logger.info(f"Using direct yfinance fallback for {our_method}")
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
            return result.get("articles", result.get("news", []))
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
            fallback,
            extra_mcp_args={"statement_type": "balance_sheet"}
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
            fallback,
            extra_mcp_args={"statement_type": "income_statement"}
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
            fallback,
            extra_mcp_args={"statement_type": "cash_flow"}
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
