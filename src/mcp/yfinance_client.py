"""
Enhanced MCP client for Yahoo Finance data integration.

This client prioritizes using free MCP servers with automatic fallback
to direct yfinance library if MCP servers fail.

MCP Priority Strategy:
1. Try primary MCP server (@modelcontextprotocol/server-yahoo-finance)
2. Try alternative MCP servers (AgentX-ai, Alex2Yang97, leoncuhk, Zentickr)
3. Fallback to direct yfinance library

This ensures maximum reliability while preferring MCP when available.
"""

import json
import asyncio
import subprocess
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import yfinance as yf
from datetime import datetime
import aiohttp
from src.utils.logger import setup_logger

logger = setup_logger("mcp.yfinance")


class MCPServerConfig:
    """Configuration for a single MCP server."""

    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.type = config.get("type", "command")  # 'command' or 'http'
        self.command = config.get("command")
        self.args = config.get("args", [])
        self.url = config.get("url")  # For HTTP-based servers
        self.description = config.get("description", "")
        self.priority = config.get("priority", 99)
        self.enabled = config.get("enabled", True)


class YahooFinanceMCPClient:
    """
    Enhanced Yahoo Finance client with MCP priority and fallback.

    Strategy:
    1. Try MCP servers in priority order
    2. If all MCP servers fail, use direct yfinance library
    3. Cache results to reduce API calls
    """

    def __init__(self):
        """Initialize Yahoo Finance MCP client with multiple server support."""
        self.logger = logger
        self.mcp_config = self._load_mcp_config()
        self.mcp_servers = self._initialize_mcp_servers()
        self.fallback_enabled = self.mcp_config.get("fallbackStrategy", {}).get("enabled", True)
        self.max_retries = self.mcp_config.get("fallbackStrategy", {}).get("maxRetries", 3)

        print(f"\n{'='*60}")
        print(f"🚀 MCP CLIENT INITIALIZED")
        print(f"{'='*60}")
        print(f"Total MCP servers: {len(self.mcp_servers)}")
        print(f"Fallback enabled: {self.fallback_enabled}")
        print(f"\nConfigured MCP Servers:")
        for server in self.mcp_servers:
            print(f"  Priority {server.priority}: {server.name}")
            print(f"    Type: {server.type}")
            if server.type == "http":
                print(f"    URL: {server.url}")
            else:
                print(f"    Command: {server.command} {' '.join(server.args)}")
            print(f"    Description: {server.description}")
            print()
        print(f"{'='*60}\n")

        self.logger.info(f"Initialized MCP client with {len(self.mcp_servers)} servers")
        for server in self.mcp_servers:
            self.logger.info(f"  [{server.priority}] {server.name}: {server.description}")

    def _load_mcp_config(self) -> Dict[str, Any]:
        """Load MCP configuration from config file."""
        try:
            config_path = Path(__file__).parent.parent.parent / "config" / "mcp_config.json"
            with open(config_path, "r") as f:
                return json.load(f)
        except Exception as e:
            self.logger.warning(f"Failed to load MCP config: {e}, using defaults")
            return {"mcpServers": {}, "fallbackStrategy": {"enabled": True}}

    def _initialize_mcp_servers(self) -> List[MCPServerConfig]:
        """Initialize and sort MCP servers by priority."""
        servers = []
        for name, config in self.mcp_config.get("mcpServers", {}).items():
            server = MCPServerConfig(name, config)
            if server.enabled:
                servers.append(server)

        # Sort by priority (lower number = higher priority)
        servers.sort(key=lambda x: x.priority)
        return servers

    async def _try_mcp_server(
        self,
        server: MCPServerConfig,
        method: str,
        params: Dict[str, Any]
    ) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """
        Try to fetch data from a specific MCP server.

        Args:
            server: MCP server configuration
            method: MCP method to call (e.g., "get_ticker_info")
            params: Parameters for the method

        Returns:
            Tuple of (success, data)
        """
        try:
            print(f"\n{'='*60}")
            print(f"🔌 ATTEMPTING MCP CONNECTION")
            print(f"{'='*60}")
            print(f"Server: {server.name}")
            print(f"Type: {server.type}")
            print(f"Method: {method}")
            print(f"Params: {params}")

            if server.type == "http" and server.url:
                print(f"URL: {server.url}")
                print(f"Making HTTP request to MCP server...")

                # Prepare headers required by MCP server
                headers = {
                    "Accept": "application/json, text/event-stream",
                    "Content-Type": "application/json"
                }

                # Make HTTP request to MCP server
                async with aiohttp.ClientSession() as session:
                    # Prepare MCP request payload
                    mcp_payload = {
                        "jsonrpc": "2.0",
                        "id": 1,
                        "method": method,
                        "params": params
                    }

                    print(f"Payload: {json.dumps(mcp_payload, indent=2)}")
                    print(f"Headers: {headers}")

                    async with session.post(
                        server.url,
                        json=mcp_payload,
                        headers=headers,
                        timeout=aiohttp.ClientTimeout(total=10)
                    ) as response:
                        print(f"Response Status: {response.status}")

                        if response.status == 200:
                            data = await response.json()
                            print(f"✅ MCP SUCCESS!")
                            print(f"Response: {json.dumps(data, indent=2)[:500]}...")
                            print(f"{'='*60}\n")

                            # Extract result from JSON-RPC response
                            if "result" in data:
                                return True, data["result"]
                            return True, data
                        else:
                            error_text = await response.text()
                            print(f"❌ MCP FAILED: HTTP {response.status}")
                            print(f"Error: {error_text[:200]}")
                            print(f"{'='*60}\n")
                            return False, None
            else:
                # Command-based MCP server (legacy support)
                print(f"Command: {server.command} {' '.join(server.args)}")
                print(f"⚠️ Command-based MCP not implemented yet")
                print(f"{'='*60}\n")
                return False, None

        except aiohttp.ClientError as e:
            print(f"❌ MCP CONNECTION ERROR: {e}")
            print(f"{'='*60}\n")
            self.logger.debug(f"MCP server {server.name} connection failed: {e}")
            return False, None
        except Exception as e:
            print(f"❌ MCP ERROR: {e}")
            print(f"{'='*60}\n")
            self.logger.debug(f"MCP server {server.name} failed: {e}")
            return False, None

    async def _fetch_with_mcp_priority(
        self,
        method: str,
        params: Dict[str, Any],
        fallback_func
    ) -> Dict[str, Any]:
        """
        Fetch data with MCP priority and automatic fallback.

        Args:
            method: MCP method name
            params: Method parameters
            fallback_func: Fallback function to use if MCP fails

        Returns:
            Data dictionary
        """
        # Try each MCP server in priority order
        for server in self.mcp_servers:
            self.logger.debug(f"Attempting {method} via MCP: {server.name}")
            success, data = await self._try_mcp_server(server, method, params)

            if success and data:
                self.logger.info(f"✓ MCP SUCCESS: {server.name} returned data for {method}")
                print(f"\n✅ Using MCP data from: {server.name}\n")
                return data

            self.logger.debug(f"✗ MCP server {server.name} unavailable, trying next...")

        # All MCP servers failed, use fallback
        if self.fallback_enabled:
            print(f"\n{'='*60}")
            print(f"⚠️ ALL MCP SERVERS FAILED - USING FALLBACK")
            print(f"{'='*60}")
            print(f"Method: {method}")
            print(f"Falling back to: Direct yfinance library")
            print(f"{'='*60}\n")
            self.logger.info(f"All MCP servers failed, using direct yfinance fallback for {method}")
            return await fallback_func(**params)
        else:
            raise Exception("All MCP servers failed and fallback is disabled")

    async def get_ticker_info(self, symbol: str) -> Dict[str, Any]:
        """
        Get basic ticker information.

        Priority: MCP servers → Direct yfinance

        Args:
            symbol: Stock ticker symbol

        Returns:
            Dictionary containing ticker info
        """

        async def fallback(**kwargs):
            """Direct yfinance fallback."""
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

        return await self._fetch_with_mcp_priority(
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
        """
        Get historical price data.

        Priority: MCP servers → Direct yfinance

        Args:
            symbol: Stock ticker symbol
            period: Data period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)

        Returns:
            Dictionary containing historical data
        """

        async def fallback(**kwargs):
            """Direct yfinance fallback."""
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

        return await self._fetch_with_mcp_priority(
            "get_ticker_historical",
            {"symbol": symbol, "period": period, "interval": interval},
            fallback
        )

    async def get_ticker_news(self, symbol: str, count: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent news for a ticker.

        Priority: MCP servers → Direct yfinance

        Args:
            symbol: Stock ticker symbol
            count: Number of news items to retrieve

        Returns:
            List of news articles
        """

        async def fallback(**kwargs):
            """Direct yfinance fallback."""
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

        result = await self._fetch_with_mcp_priority(
            "get_ticker_news",
            {"symbol": symbol, "count": count},
            fallback
        )

        # Handle case where MCP returns dict instead of list
        if isinstance(result, dict):
            return result.get("articles", [])
        return result

    async def get_balance_sheet(self, symbol: str) -> Dict[str, Any]:
        """
        Get balance sheet data.

        Priority: MCP servers → Direct yfinance

        Args:
            symbol: Stock ticker symbol

        Returns:
            Balance sheet data
        """

        async def fallback(**kwargs):
            """Direct yfinance fallback."""
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

        return await self._fetch_with_mcp_priority(
            "get_balance_sheet",
            {"symbol": symbol},
            fallback
        )

    async def get_income_statement(self, symbol: str) -> Dict[str, Any]:
        """
        Get income statement data.

        Priority: MCP servers → Direct yfinance

        Args:
            symbol: Stock ticker symbol

        Returns:
            Income statement data
        """

        async def fallback(**kwargs):
            """Direct yfinance fallback."""
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

        return await self._fetch_with_mcp_priority(
            "get_income_statement",
            {"symbol": symbol},
            fallback
        )

    async def get_cash_flow(self, symbol: str) -> Dict[str, Any]:
        """
        Get cash flow statement data.

        Priority: MCP servers → Direct yfinance

        Args:
            symbol: Stock ticker symbol

        Returns:
            Cash flow data
        """

        async def fallback(**kwargs):
            """Direct yfinance fallback."""
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

        return await self._fetch_with_mcp_priority(
            "get_cash_flow",
            {"symbol": symbol},
            fallback
        )

    async def get_comprehensive_data(self, symbol: str) -> Dict[str, Any]:
        """
        Get all available data for a ticker in one call.

        Priority: MCP servers → Direct yfinance

        Args:
            symbol: Stock ticker symbol

        Returns:
            Comprehensive data dictionary
        """
        self.logger.info(f"Fetching comprehensive data for {symbol}")
        self.logger.info(f"Priority: MCP servers ({len(self.mcp_servers)} available) → Direct yfinance")

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
        sources = [
            info.get("source", "unknown") if isinstance(info, dict) else "error",
            historical.get("source", "unknown") if isinstance(historical, dict) else "error"
        ]
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
