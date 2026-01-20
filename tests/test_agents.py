"""Unit tests for agent implementations."""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from src.agents.data_collector import DataCollectorAgent
from src.agents.fundamental_analyst import FundamentalAnalystAgent
from src.agents.technical_analyst import TechnicalAnalystAgent
from src.agents.sentiment_analyst import SentimentAnalystAgent


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
    """Tests for Data Collector Agent."""

    @pytest.mark.asyncio
    async def test_execute_success(self, mock_openai_client, mock_db_client):
        """Test successful data collection."""
        agent = DataCollectorAgent(mock_openai_client, mock_db_client)

        # Mock yfinance client
        with patch.object(agent, 'yfinance_client') as mock_yf:
            mock_yf.get_comprehensive_data = AsyncMock(return_value={
                "symbol": "AAPL",
                "info": {"name": "Apple Inc.", "sector": "Technology"},
                "historical": {"current_price": 150.0},
                "news": []
            })

            result = await agent.execute({"ticker": "AAPL"})

            assert result["ticker"] == "AAPL"
            assert "data" in result
            assert result["confidence"] > 0.0

    @pytest.mark.asyncio
    async def test_execute_with_cache(self, mock_openai_client, mock_db_client):
        """Test data collection with cache hit."""
        mock_db_client.get_cached_data = Mock(return_value={"ticker": "AAPL"})

        agent = DataCollectorAgent(mock_openai_client, mock_db_client)
        result = await agent.execute({"ticker": "AAPL"})

        assert result["from_cache"] is True
        assert result["confidence"] == 1.0


class TestFundamentalAnalystAgent:
    """Tests for Fundamental Analyst Agent."""

    @pytest.mark.asyncio
    async def test_calculate_metrics(self, mock_openai_client, mock_db_client):
        """Test metrics calculation."""
        agent = FundamentalAnalystAgent(mock_openai_client, mock_db_client)

        collected_data = {
            "basic_info": {"market_cap": 1000000000},
            "financials": {
                "income_statement": {
                    "2023": {
                        "Total Revenue": 100000000,
                        "Net Income": 20000000
                    }
                },
                "balance_sheet": {
                    "2023": {
                        "Total Assets": 200000000,
                        "Current Assets": 50000000,
                        "Current Liabilities": 30000000
                    }
                }
            }
        }

        metrics = agent._calculate_metrics(collected_data)

        assert "profitability" in metrics
        assert "health" in metrics
        assert isinstance(metrics["profitability"], dict)

    @pytest.mark.asyncio
    async def test_execute_with_llm(self, mock_openai_client, mock_db_client):
        """Test execution with LLM analysis."""
        # Mock LLM response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = '''{
            "score": 7.5,
            "strengths": ["Strong financials"],
            "weaknesses": ["High competition"],
            "recommendation": "buy",
            "confidence": 0.8,
            "reasoning": "Good fundamentals"
        }'''

        mock_openai_client.chat.completions.create.return_value = mock_response

        agent = FundamentalAnalystAgent(mock_openai_client, mock_db_client)

        collected_data = {
            "basic_info": {"name": "Apple", "sector": "Tech", "market_cap": 1e12},
            "financials": {"income_statement": {}, "balance_sheet": {}},
            "price_data": {"current_price": 150}
        }

        result = await agent.execute({
            "ticker": "AAPL",
            "collected_data": collected_data
        })

        assert result["ticker"] == "AAPL"
        assert "score" in result
        assert "recommendation" in result


class TestTechnicalAnalystAgent:
    """Tests for Technical Analyst Agent."""

    def test_calculate_rsi(self, mock_openai_client, mock_db_client):
        """Test RSI calculation."""
        import pandas as pd
        import numpy as np

        agent = TechnicalAnalystAgent(mock_openai_client, mock_db_client)

        # Create sample price data
        prices = pd.Series([100, 102, 101, 103, 105, 104, 106, 108, 107, 109,
                           110, 108, 107, 109, 111, 110, 112, 114, 113, 115])

        rsi = agent._calculate_rsi(prices, period=14)

        assert rsi is not None
        assert 0 <= rsi <= 100

    @pytest.mark.asyncio
    async def test_determine_trend(self, mock_openai_client, mock_db_client):
        """Test trend determination."""
        import pandas as pd

        agent = TechnicalAnalystAgent(mock_openai_client, mock_db_client)

        # Create sample dataframe
        df = pd.DataFrame({
            'Close': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]
        })

        indicators = {
            "SMA_50": 105,
            "SMA_200": 100,
            "RSI": 65,
            "MACD": {"signal": "Bullish"}
        }

        trend = agent._determine_trend(indicators, df)

        assert trend in ["bullish", "bearish", "neutral"]


class TestSentimentAnalystAgent:
    """Tests for Sentiment Analyst Agent."""

    def test_fallback_sentiment(self, mock_openai_client, mock_db_client):
        """Test fallback sentiment analysis."""
        agent = SentimentAnalystAgent(mock_openai_client, mock_db_client)

        news = [
            {
                "headline": "Company beats earnings expectations",
                "summary": "Strong profit growth reported",
                "source": "Reuters"
            },
            {
                "headline": "Stock faces lawsuit over practices",
                "summary": "Legal concerns arise",
                "source": "Bloomberg"
            }
        ]

        result = agent._fallback_sentiment_analysis(news)

        assert "sentiment_score" in result
        assert -1 <= result["sentiment_score"] <= 1
        assert "news_summary" in result
        assert len(result["news_summary"]) == 2

    @pytest.mark.asyncio
    async def test_execute_no_news(self, mock_openai_client, mock_db_client):
        """Test execution with no news."""
        agent = SentimentAnalystAgent(mock_openai_client, mock_db_client)

        result = await agent.execute({
            "ticker": "AAPL",
            "collected_data": {"news": []}
        })

        assert result["sentiment_score"] == 0.0
        assert result["overall_mood"] == "neutral"
        assert result["confidence"] < 0.5


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
