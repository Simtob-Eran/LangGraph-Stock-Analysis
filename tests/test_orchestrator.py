"""Tests for orchestrator."""

import pytest
from unittest.mock import Mock, AsyncMock, patch
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
    """Test orchestrator initializes all agents."""
    assert mock_orchestrator.data_collector is not None
    assert mock_orchestrator.fundamental_analyst is not None
    assert mock_orchestrator.technical_analyst is not None
    assert mock_orchestrator.sentiment_analyst is not None
    assert mock_orchestrator.debate_agent is not None
    assert mock_orchestrator.risk_manager is not None
    assert mock_orchestrator.synthesis_agent is not None
    assert mock_orchestrator.feedback_loop is not None


@pytest.mark.asyncio
async def test_analyze_invalid_query(mock_orchestrator):
    """Test analysis with invalid query."""
    result = await mock_orchestrator.analyze("No tickers here")

    # Should handle gracefully
    assert result["status"] in ["error", "success"]


@pytest.mark.asyncio
async def test_analyze_single_ticker(mock_orchestrator):
    """Test single ticker analysis."""
    # Mock the workflow
    mock_orchestrator.workflow = Mock()
    mock_orchestrator.workflow.ainvoke = AsyncMock(return_value={
        "ticker": "AAPL",
        "status": "completed",
        "synthesis": {"report_id": "test-123"}
    })

    # Mock database
    mock_orchestrator.db.create_analysis_run = Mock(return_value="run-123")
    mock_orchestrator.db.update_analysis_run = Mock()

    result = await mock_orchestrator.analyze("AAPL")

    assert "status" in result
    assert "execution_time" in result


def test_workflow_graph_structure(mock_orchestrator):
    """Test that workflow graph has correct structure."""
    workflow = mock_orchestrator.workflow

    # Workflow should be compiled
    assert workflow is not None


@pytest.mark.asyncio
async def test_parallel_analysis(mock_orchestrator):
    """Test parallel analysis of multiple stocks."""
    # Mock single analysis
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
