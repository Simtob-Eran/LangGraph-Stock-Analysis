"""Tests for orchestrator."""

import json
import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from src.orchestrator import Orchestrator


@pytest.fixture
def mock_orchestrator():
    """Create orchestrator with mocked components."""
    with patch('src.orchestrator.AsyncOpenAI'), \
         patch('src.orchestrator.Database'):
        orchestrator = Orchestrator()
        return orchestrator


@pytest.mark.asyncio
async def test_orchestrator_initialization(mock_orchestrator):
    """Test orchestrator initializes with pending state."""
    assert mock_orchestrator._initialized is False
    assert mock_orchestrator.agents == {}
    assert mock_orchestrator.workflow is None


@pytest.mark.asyncio
async def test_orchestrator_initialize(mock_orchestrator):
    """Test that initialize() creates agents and workflow."""
    mock_tools = [MagicMock(name="get_ticker_info")]

    with patch('src.orchestrator.create_mcp_client') as mock_create_mcp, \
         patch('src.orchestrator.create_all_agents') as mock_create_agents:

        mock_client = MagicMock()
        mock_client.get_tools = AsyncMock(return_value=mock_tools)
        mock_create_mcp.return_value = mock_client

        mock_agents = {
            "data_collector": MagicMock(),
            "fundamental_analyst": MagicMock(),
            "technical_analyst": MagicMock(),
            "sentiment_analyst": MagicMock(),
            "debate_agent": MagicMock(),
            "risk_manager": MagicMock(),
            "synthesis_agent": MagicMock(),
            "feedback_loop": MagicMock(),
        }
        mock_create_agents.return_value = mock_agents

        await mock_orchestrator.initialize()

        assert mock_orchestrator._initialized is True
        assert mock_orchestrator.agents == mock_agents
        assert mock_orchestrator.workflow is not None


@pytest.mark.asyncio
async def test_orchestrator_initialize_no_mcp(mock_orchestrator):
    """Test initialization when MCP is not available."""
    with patch('src.orchestrator.create_mcp_client') as mock_create_mcp, \
         patch('src.orchestrator.create_all_agents') as mock_create_agents:

        mock_create_mcp.return_value = None
        mock_create_agents.return_value = {"agent1": MagicMock()}

        await mock_orchestrator.initialize()

        assert mock_orchestrator._initialized is True
        # create_all_agents should be called with empty tools
        mock_create_agents.assert_called_once_with([])


@pytest.mark.asyncio
async def test_orchestrator_initialize_idempotent(mock_orchestrator):
    """Test that calling initialize() twice doesn't re-initialize."""
    with patch('src.orchestrator.create_mcp_client') as mock_create_mcp, \
         patch('src.orchestrator.create_all_agents') as mock_create_agents:

        mock_create_mcp.return_value = None
        mock_create_agents.return_value = {}

        await mock_orchestrator.initialize()
        await mock_orchestrator.initialize()

        # Should only be called once
        assert mock_create_mcp.call_count == 1


@pytest.mark.asyncio
async def test_invoke_agent(mock_orchestrator):
    """Test _invoke_agent parses JSON from agent response."""
    mock_agent = MagicMock()
    mock_message = MagicMock()
    mock_message.content = '{"score": 7.5, "confidence": 0.8, "recommendation": "buy"}'
    mock_agent.ainvoke = AsyncMock(return_value={
        "messages": [MagicMock(), mock_message]
    })

    mock_orchestrator.agents = {"test_agent": mock_agent}

    result = await mock_orchestrator._invoke_agent("test_agent", "Analyze AAPL")

    assert result["score"] == 7.5
    assert result["confidence"] == 0.8
    assert result["recommendation"] == "buy"


@pytest.mark.asyncio
async def test_invoke_agent_json_in_markdown(mock_orchestrator):
    """Test _invoke_agent extracts JSON from markdown code blocks."""
    mock_agent = MagicMock()
    mock_message = MagicMock()
    mock_message.content = 'Here is my analysis:\n```json\n{"score": 8.0}\n```\nDone.'
    mock_agent.ainvoke = AsyncMock(return_value={
        "messages": [MagicMock(), mock_message]
    })

    mock_orchestrator.agents = {"test_agent": mock_agent}

    result = await mock_orchestrator._invoke_agent("test_agent", "Analyze AAPL")

    assert result["score"] == 8.0


@pytest.mark.asyncio
async def test_invoke_agent_failure(mock_orchestrator):
    """Test _invoke_agent handles agent failure gracefully."""
    mock_agent = MagicMock()
    mock_agent.ainvoke = AsyncMock(side_effect=Exception("LLM timeout"))

    mock_orchestrator.agents = {"test_agent": mock_agent}

    result = await mock_orchestrator._invoke_agent("test_agent", "Analyze AAPL")

    assert "error" in result
    assert result["confidence"] == 0.0
    assert result["recommendation"] == "hold"


def test_extract_json_direct(mock_orchestrator):
    """Test JSON extraction from raw JSON string."""
    result = mock_orchestrator._extract_json('{"score": 5.0}')
    assert result["score"] == 5.0


def test_extract_json_markdown(mock_orchestrator):
    """Test JSON extraction from markdown code block."""
    content = 'Analysis:\n```json\n{"score": 7.0}\n```'
    result = mock_orchestrator._extract_json(content)
    assert result["score"] == 7.0


def test_extract_json_embedded(mock_orchestrator):
    """Test JSON extraction from text with embedded JSON."""
    content = 'My analysis is {"score": 6.0, "recommendation": "hold"} based on data.'
    result = mock_orchestrator._extract_json(content)
    assert result["score"] == 6.0


def test_extract_json_fallback(mock_orchestrator):
    """Test JSON extraction fallback for non-JSON content."""
    result = mock_orchestrator._extract_json("This is plain text analysis.")
    assert result["score"] == 5.0
    assert result["confidence"] == 0.3
    assert "reasoning" in result


def test_calculate_overall_score(mock_orchestrator):
    """Test overall score calculation."""
    fundamental = {"score": 8.0}
    technical = {"score": 7.0}
    sentiment = {"sentiment_score": 0.5}  # Maps to 7.5
    risk = {"risk_score": 3.0}  # Inverted to 7.0

    score = mock_orchestrator._calculate_overall_score(
        fundamental, technical, sentiment, risk
    )

    # Expected: 8.0*0.35 + 7.0*0.25 + 7.5*0.20 + 7.0*0.20
    # = 2.8 + 1.75 + 1.5 + 1.4 = 7.45 -> 7.4 (rounded)
    assert 7.0 <= score <= 8.0


@pytest.mark.asyncio
async def test_analyze_auto_initializes(mock_orchestrator):
    """Test that analyze() calls initialize() if not yet initialized."""
    with patch.object(mock_orchestrator, 'initialize', new_callable=AsyncMock) as mock_init, \
         patch.object(mock_orchestrator, '_extract_tickers_with_llm',
                      new_callable=AsyncMock, return_value=([], "none")):

        await mock_orchestrator.analyze("AAPL")

        mock_init.assert_called_once()


@pytest.mark.asyncio
async def test_parallel_analysis(mock_orchestrator):
    """Test parallel analysis of multiple stocks."""
    mock_orchestrator._initialized = True
    mock_orchestrator.workflow = MagicMock()

    async def mock_analyze_single(ticker, query):
        return {
            "status": "success",
            "analyses": [{"ticker": ticker}],
            "error_message": None
        }

    mock_orchestrator._analyze_single = mock_analyze_single

    result = await mock_orchestrator._analyze_multiple(
        ["AAPL", "MSFT"],
        "Compare stocks"
    )

    assert result["status"] in ["success", "partial"]
    assert "analyses" in result


@pytest.mark.asyncio
async def test_collect_data_node_uses_agent(mock_orchestrator):
    """Test that _collect_data_node invokes data_collector agent."""
    mock_agent = MagicMock()
    mock_message = MagicMock()
    mock_message.content = json.dumps({
        "valid": True,
        "ticker": "AAPL",
        "company_name": "Apple Inc.",
        "confidence": 0.95
    })
    mock_agent.ainvoke = AsyncMock(return_value={
        "messages": [MagicMock(), mock_message]
    })

    mock_orchestrator.agents = {"data_collector": mock_agent}

    state = {
        "ticker": "AAPL",
        "query": "Analyze AAPL",
        "run_id": "test-123"
    }

    result = await mock_orchestrator._collect_data_node(state)

    assert "collected_data" in result
    assert result["collected_data"]["valid"] is True
    assert result["collected_data"]["company_name"] == "Apple Inc."
