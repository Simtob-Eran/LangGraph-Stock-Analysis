"""
MCP Client Factory -- creates MultiServerMCPClient from project config.
Reuses the MCP URL/headers logic already in YahooFinanceMCPClient.
"""
import os
import json
from pathlib import Path
from typing import Optional, Tuple, Dict
from langchain_mcp_adapters.client import MultiServerMCPClient
from src.utils.logger import setup_logger

logger = setup_logger("mcp.client_factory")


def _load_mcp_url_and_headers() -> Tuple[Optional[str], Dict[str, str]]:
    """
    Load MCP URL and headers using same priority as YahooFinanceMCPClient:
    1. MCP_URL env var (URL only, no headers)
    2. config/mcp_config.json first enabled server
    Returns (url, headers) tuple.
    """
    # Try env var first
    url = os.getenv("MCP_URL")
    if url:
        logger.info(f"Using MCP URL from environment: {url}")
        return url, {}

    # Try config file
    try:
        config_path = Path(__file__).parent.parent.parent / "config" / "mcp_config.json"
        with open(config_path) as f:
            config = json.load(f)

        for name, server in config.get("mcpServers", {}).items():
            if server.get("enabled", True):
                url = server.get("url")
                headers = server.get("headers", {})
                logger.info(f"Using MCP server '{name}': {url}")
                return url, headers
    except Exception as e:
        logger.warning(f"Could not load mcp_config.json: {e}")

    return None, {}


def create_mcp_client() -> Optional[MultiServerMCPClient]:
    """
    Create a MultiServerMCPClient connected to the yfinance MCP server.

    Returns None if no MCP URL is configured (agents will run without live tools).

    Usage:
        client = create_mcp_client()
        if client:
            tools = await client.get_tools()
        else:
            tools = []

    Note: MultiServerMCPClient is stateless -- each tool call creates a fresh
    MCP session. No context manager needed. Just call client.get_tools() and
    pass tools to create_agent.
    """
    url, headers = _load_mcp_url_and_headers()

    if not url:
        logger.warning("No MCP URL configured -- MultiServerMCPClient not created")
        return None

    server_config = {
        "transport": "http",
        "url": url,
    }
    if headers:
        server_config["headers"] = headers

    logger.info(f"Creating MultiServerMCPClient -> {url}")

    return MultiServerMCPClient({
        "yfinance": server_config
    })
