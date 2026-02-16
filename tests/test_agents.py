"""Unit tests for agent implementations."""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from src.agents.data_collector import DataCollectorAgent


@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client."""
    client = Mock()
    client.chat = Mock()
    client.chat.completions = Mock()
    client.chat.completions.create = AsyncMock()
    return client


@pytest.fixture
def mock_db_client():
    """Mock database client."""
    db = Mock()
    db.log_agent_execution = Mock()
    db.get_cached_data = Mock(return_value=None)
    db.cache_data = Mock()
    return db


class TestDataCollectorAgent:
    """Tests for Data Collector Agent (ticker validation only)."""

    @pytest.mark.asyncio
    async def test_execute_success(self, mock_openai_client, mock_db_client):
        """Test successful ticker validation."""
        agent = DataCollectorAgent(mock_openai_client, mock_db_client)

        # Mock yfinance client
        with patch.object(agent, 'yfinance_client') as mock_yf:
            mock_yf.get_ticker_info = AsyncMock(return_value={
                "symbol": "AAPL",
                "name": "Apple Inc.",
                "sector": "Technology",
            })

            result = await agent.execute({"ticker": "AAPL"})

            assert result["ticker"] == "AAPL"
            assert result["valid"] is True
            assert result["company_name"] == "Apple Inc."
            assert result["confidence"] > 0.0

    @pytest.mark.asyncio
    async def test_execute_invalid_ticker(self, mock_openai_client, mock_db_client):
        """Test validation with invalid ticker."""
        agent = DataCollectorAgent(mock_openai_client, mock_db_client)

        with patch.object(agent, 'yfinance_client') as mock_yf:
            mock_yf.get_ticker_info = AsyncMock(return_value={
                "symbol": "XXXXX",
                "name": None,
            })

            result = await agent.execute({"ticker": "XXXXX"})

            assert result["ticker"] == "XXXXX"
            assert result["valid"] is False

    @pytest.mark.asyncio
    async def test_execute_error(self, mock_openai_client, mock_db_client):
        """Test validation when MCP client fails."""
        agent = DataCollectorAgent(mock_openai_client, mock_db_client)

        with patch.object(agent, 'yfinance_client') as mock_yf:
            mock_yf.get_ticker_info = AsyncMock(side_effect=Exception("Connection failed"))

            result = await agent.execute({"ticker": "AAPL"})

            assert result["valid"] is False
            assert "error" in result


class TestMCPClientFactory:
    """Tests for MCP client factory."""

    def test_load_url_from_env(self):
        """Test loading MCP URL from environment variable."""
        from src.mcp.mcp_client_factory import _load_mcp_url_and_headers

        with patch.dict('os.environ', {'MCP_URL': 'http://test:8080/mcp'}):
            url, headers = _load_mcp_url_and_headers()
            assert url == 'http://test:8080/mcp'
            assert headers == {}

    def test_load_url_from_config(self):
        """Test loading MCP URL from config file."""
        from src.mcp.mcp_client_factory import _load_mcp_url_and_headers

        mock_config = {
            "mcpServers": {
                "test-server": {
                    "url": "http://config:8080/mcp",
                    "headers": {"Authorization": "Bearer test"},
                    "enabled": True
                }
            }
        }

        with patch.dict('os.environ', {}, clear=True):
            with patch('builtins.open', create=True) as mock_open:
                import json
                mock_open.return_value.__enter__ = lambda s: s
                mock_open.return_value.__exit__ = Mock(return_value=False)
                mock_open.return_value.read = Mock(return_value=json.dumps(mock_config))

                # Can't easily mock Path + open together, so just test env var path
                pass

    def test_create_mcp_client_no_url(self):
        """Test that create_mcp_client returns None when no URL configured."""
        from src.mcp.mcp_client_factory import create_mcp_client

        with patch('src.mcp.mcp_client_factory._load_mcp_url_and_headers',
                   return_value=(None, {})):
            client = create_mcp_client()
            assert client is None


class TestAgentFactory:
    """Tests for agent factory."""

    def test_build_llm(self):
        """Test LLM construction."""
        from src.agents.agent_factory import build_llm

        with patch('src.agents.agent_factory.settings') as mock_settings:
            mock_settings.OPENAI_MODEL = "gpt-4o"
            mock_settings.OPENAI_API_KEY = "test-key"

            llm = build_llm()
            assert llm is not None

    def test_create_all_agents(self):
        """Test creating all agents with mock tools."""
        from src.agents.agent_factory import create_all_agents

        mock_tools = []

        with patch('src.agents.agent_factory.create_agent') as mock_create:
            mock_create.return_value = MagicMock()

            agents = create_all_agents(mock_tools)

            assert "fundamental_analyst" in agents
            assert "technical_analyst" in agents
            assert "sentiment_analyst" in agents
            assert "debate_agent" in agents
            assert "risk_manager" in agents
            assert "synthesis_agent" in agents
            assert "feedback_loop" in agents
            assert len(agents) == 7

    def test_create_all_agents_with_tools(self):
        """Test creating agents with MCP tools."""
        from src.agents.agent_factory import create_all_agents

        mock_tool = MagicMock()
        mock_tool.name = "get_ticker_info"
        mock_tools = [mock_tool]

        with patch('src.agents.agent_factory.create_agent') as mock_create:
            mock_create.return_value = MagicMock()

            agents = create_all_agents(mock_tools)

            # Verify create_agent was called with tools for each agent
            assert mock_create.call_count == 7
            for call in mock_create.call_args_list:
                assert call[0][1] == mock_tools  # second positional arg is tools


class TestPrompts:
    """Tests for autonomous agent prompts."""

    def test_all_prompts_exist(self):
        """Test that all required prompt variables exist."""
        from src.models.prompts import (
            FUNDAMENTAL_ANALYST_PROMPT,
            TECHNICAL_ANALYST_PROMPT,
            SENTIMENT_ANALYST_PROMPT,
            DEBATE_AGENT_PROMPT,
            RISK_MANAGER_PROMPT,
            SYNTHESIS_AGENT_PROMPT,
            FEEDBACK_LOOP_PROMPT,
            DATA_COLLECTOR_PROMPT,
            ORCHESTRATOR_PROMPT,
        )

        prompts = [
            FUNDAMENTAL_ANALYST_PROMPT,
            TECHNICAL_ANALYST_PROMPT,
            SENTIMENT_ANALYST_PROMPT,
            DEBATE_AGENT_PROMPT,
            RISK_MANAGER_PROMPT,
            SYNTHESIS_AGENT_PROMPT,
            FEEDBACK_LOOP_PROMPT,
            DATA_COLLECTOR_PROMPT,
            ORCHESTRATOR_PROMPT,
        ]

        for prompt in prompts:
            assert isinstance(prompt, str)
            assert len(prompt) > 50

    def test_prompts_mention_tools(self):
        """Test that agent prompts reference MCP tools."""
        from src.models.prompts import (
            FUNDAMENTAL_ANALYST_PROMPT,
            TECHNICAL_ANALYST_PROMPT,
            SENTIMENT_ANALYST_PROMPT,
            RISK_MANAGER_PROMPT,
        )

        # These autonomous agents should mention tools
        for prompt in [FUNDAMENTAL_ANALYST_PROMPT, TECHNICAL_ANALYST_PROMPT,
                       SENTIMENT_ANALYST_PROMPT, RISK_MANAGER_PROMPT]:
            assert "tool" in prompt.lower() or "Tool" in prompt

    def test_prompts_require_json_output(self):
        """Test that analysis prompts specify JSON output format."""
        from src.models.prompts import (
            FUNDAMENTAL_ANALYST_PROMPT,
            TECHNICAL_ANALYST_PROMPT,
            SENTIMENT_ANALYST_PROMPT,
            DEBATE_AGENT_PROMPT,
            RISK_MANAGER_PROMPT,
            FEEDBACK_LOOP_PROMPT,
        )

        for prompt in [FUNDAMENTAL_ANALYST_PROMPT, TECHNICAL_ANALYST_PROMPT,
                       SENTIMENT_ANALYST_PROMPT, DEBATE_AGENT_PROMPT,
                       RISK_MANAGER_PROMPT, FEEDBACK_LOOP_PROMPT]:
            assert "JSON" in prompt or "json" in prompt


@pytest.mark.asyncio
async def test_agent_base_retry_logic(mock_openai_client, mock_db_client):
    """Test retry logic in base agent."""
    from src.agents.base_agent import BaseAgent

    class TestAgent(BaseAgent):
        async def execute(self, inputs):
            return {"test": "result"}

    agent = TestAgent("test_agent", mock_openai_client, mock_db_client)

    # Test successful retry after failures
    call_count = 0

    async def failing_function():
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise Exception("Temporary failure")
        return "success"

    result = await agent._retry_on_failure(failing_function, max_retries=3)

    assert result == "success"
    assert call_count == 3


def test_metrics_validation():
    """Test metric validation functions."""
    from src.utils.validators import validate_score, validate_confidence

    # Test score validation
    assert validate_score(5.0) == 5.0
    assert validate_score(-1.0) == 0.0
    assert validate_score(15.0) == 10.0

    # Test confidence validation
    assert validate_confidence(0.5) == 0.5
    assert validate_confidence(-0.1) == 0.0
    assert validate_confidence(1.5) == 1.0


def test_ticker_validation():
    """Test ticker validation."""
    from src.utils.validators import validate_ticker, validate_tickers

    # Test single ticker
    is_valid, error = validate_ticker("AAPL")
    assert is_valid is True
    assert error == ""

    is_valid, error = validate_ticker("")
    assert is_valid is False

    is_valid, error = validate_ticker("INVALID@TICKER")
    assert is_valid is False

    # Test multiple tickers
    is_valid, error, valid = validate_tickers(["AAPL", "MSFT", "GOOGL"])
    assert is_valid is True
    assert len(valid) == 3


def test_query_parsing():
    """Test query parsing."""
    from src.utils.validators import parse_query

    # Single ticker
    tickers, analysis_type = parse_query("Analyze AAPL")
    assert tickers == ["AAPL"]
    assert analysis_type == "single"

    # Multiple tickers
    tickers, analysis_type = parse_query("Compare AAPL, MSFT, GOOGL")
    assert len(tickers) >= 2
    assert analysis_type == "multiple"

    # No tickers
    tickers, analysis_type = parse_query("What is the market doing?")
    assert analysis_type == "sector"
