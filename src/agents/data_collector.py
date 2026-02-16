"""Data Collector Agent -- now a real create_agent autonomous agent.

The data_collector agent is created via create_agent in src/agents/agent_factory.py
using the DATA_COLLECTOR_PROMPT from src/models/prompts.py.

It autonomously uses MCP tools to:
1. Validate ticker symbols via get_ticker_info
2. Check data availability (prices, financials, news)
3. Return a structured validation report as JSON

This file is kept for backward compatibility only.
The actual agent logic lives in the prompt + create_agent ReAct loop.
"""
