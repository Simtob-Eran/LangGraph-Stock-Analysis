# 📊 Multi-Agent Stock Analysis System

A production-ready AI-powered system that uses 9 specialized agents to perform comprehensive stock market analysis. Built with **LangGraph** for multi-agent orchestration and **OpenAI GPT** for intelligent analysis.

**⚠️ FOR RESEARCH AND EDUCATIONAL PURPOSES ONLY - NOT INVESTMENT ADVICE**

## 🎯 Overview

This system demonstrates advanced multi-agent AI architecture where specialized agents collaborate to analyze stocks from multiple perspectives:

- **Data Collector**: Gathers comprehensive financial data
- **Fundamental Analyst**: Analyzes financial health and intrinsic value
- **Technical Analyst**: Studies price patterns and indicators
- **Sentiment Analyst**: Processes news and market psychology
- **Debate Agent**: Creates bull/bear investment cases
- **Risk Manager**: Assesses investment risks
- **Synthesis Agent**: Generates comprehensive reports
- **Feedback Loop**: Ensures analysis quality
- **Orchestrator**: Coordinates all agents using LangGraph

## ✨ Features

- 🤖 **9 Specialized AI Agents** working in concert
- 📈 **Comprehensive Analysis**: Fundamental, Technical, Sentiment, Risk
- 🔄 **LangGraph Orchestration**: Efficient workflow management
- 💾 **SQLite Logging**: Complete analysis history and caching
- 🆓 **Free Data Source**: Yahoo Finance (no API key needed)
- 📝 **Professional Reports**: Markdown and JSON output
- ⚡ **Parallel Processing**: Analyze multiple stocks concurrently
- 🔒 **Robust Error Handling**: Graceful degradation and retry logic

## 📋 Requirements

- Python 3.11 or higher
- OpenAI API key
- Internet connection for Yahoo Finance data

## 🚀 Quick Start

### 1. Clone and Setup

```bash
git clone <repository-url>
cd LangGraph-Stock-Analysis

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment

Create a `.env` file in the project root:

```bash
cp .env.example .env
```

Edit `.env` and add your OpenAI API key:

```env
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4o

DATABASE_PATH=./data/analysis.db
LOG_LEVEL=INFO
MAX_PARALLEL_TASKS=5

MCP_YFINANCE_ENABLED=true
```

### 3. Run Analysis

```bash
# Analyze a single stock
python -m src.main analyze "AAPL"

# Analyze with custom query
python -m src.main analyze "Analyze Apple stock"

# Compare multiple stocks
python -m src.main analyze "AAPL,MSFT,GOOGL"

# Save report to file
python -m src.main analyze "AAPL" -o reports/aapl_analysis.md

# Get JSON output
python -m src.main analyze "AAPL" --json
```

## 📁 Project Structure

```
LangGraph-Stock-Analysis/
├── config/                      # Configuration files
│   ├── __init__.py
│   ├── settings.py             # Environment settings
│   └── mcp_config.json         # Yahoo Finance MCP config
├── src/
│   ├── __init__.py
│   ├── main.py                 # CLI entry point
│   ├── orchestrator.py         # LangGraph orchestrator
│   ├── agents/                 # All agent implementations
│   │   ├── base_agent.py
│   │   ├── data_collector.py
│   │   ├── fundamental_analyst.py
│   │   ├── technical_analyst.py
│   │   ├── sentiment_analyst.py
│   │   ├── debate_agent.py
│   │   ├── risk_manager.py
│   │   ├── synthesis_agent.py
│   │   └── feedback_loop.py
│   ├── mcp/                    # Yahoo Finance integration
│   │   └── yfinance_client.py
│   ├── utils/                  # Utilities
│   │   ├── database.py
│   │   ├── logger.py
│   │   └── validators.py
│   └── models/                 # Data models
│       ├── schemas.py
│       └── prompts.py
├── data/                       # Database and cache
│   └── analysis.db
├── logs/                       # Log files
├── tests/                      # Unit tests
├── .env                        # Environment variables (gitignored)
├── .env.example                # Environment template
├── .gitignore
├── requirements.txt
├── setup.py
└── README.md
```

## 🔧 Architecture

### Multi-Agent Workflow

```
User Query
    ↓
Orchestrator (LangGraph)
    ↓
Data Collector Agent
    ↓
┌─────────────┬──────────────┬─────────────┐
│             │              │             │
Fundamental   Technical    Sentiment
Analyst       Analyst       Analyst
│             │              │
└─────────────┴──────────────┴─────────────┘
    ↓
Debate Agent (Bull vs Bear)
    ↓
Risk Manager Agent
    ↓
Synthesis Agent (Report Generation)
    ↓
Feedback Loop Agent (Quality Check)
    ↓
Final Report
```

### Agent Responsibilities

#### 1. Data Collector Agent
- Fetches stock data from Yahoo Finance
- Collects: prices, financials, news
- Implements 24-hour caching
- Handles data normalization

#### 2. Fundamental Analyst Agent
- Calculates financial ratios
- Analyzes profitability, growth, health
- Provides investment recommendation
- Scores: 0-10 scale

#### 3. Technical Analyst Agent
- Calculates indicators (RSI, MACD, SMA, EMA)
- Identifies trends and patterns
- Provides trading signals
- Finds support/resistance levels

#### 4. Sentiment Analyst Agent
- Analyzes news headlines
- Extracts market sentiment
- Identifies trending themes
- Scores: -1 (negative) to +1 (positive)

#### 5. Debate Agent
- Creates bull case arguments
- Creates bear case arguments
- Identifies conflicts between analyses
- Provides balanced recommendation

#### 6. Risk Manager Agent
- Calculates volatility metrics
- Identifies risk factors
- Provides position size recommendations
- Risk level: low/medium/high/very_high

#### 7. Synthesis Agent
- Generates comprehensive markdown report
- Creates JSON summary
- Calculates overall score
- Adds mandatory disclaimers

#### 8. Feedback Loop Agent
- Identifies missing data
- Flags low confidence areas
- Suggests additional analysis
- Quality assurance check

#### 9. Orchestrator
- Coordinates all agents
- Manages workflow with LangGraph
- Handles parallel execution
- Error handling and retry logic

## 📊 Example Output

```markdown
# Stock Analysis Report: Apple Inc. (AAPL)

**Analysis Date:** 2025-01-20 | **Purpose:** Research Only

## Executive Summary

| Metric | Value |
|--------|-------|
| **Overall Score** | **8.2/10** |
| **Recommendation** | **BUY** |
| **Current Price** | $175.50 |
| **Market Cap** | $2.75T |
| **Risk Level** | Medium |

### Quick Takeaways
- Strong fundamental metrics with consistent growth
- Bullish technical trend with positive momentum
- Positive market sentiment around new products
- Moderate risk level with good liquidity

[... detailed analysis sections ...]

## ⚠️ IMPORTANT DISCLAIMER

**FOR RESEARCH AND EDUCATIONAL PURPOSES ONLY**

This report was generated by an AI system. This is NOT investment advice.
Always consult a licensed financial advisor before investing.
```

## 🧪 Testing

Run the test suite:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src tests/

# Run specific test file
pytest tests/test_agents.py
```

## 🔍 Data Sources

All data is sourced from **Yahoo Finance** via the `yfinance` library:
- Price data: Real-time and historical
- Financial statements: Annual and quarterly
- News: Aggregated from multiple sources
- Company information: Sector, industry, fundamentals

**No API key required** for Yahoo Finance data.

## ⚙️ Configuration

### Environment Variables

- `OPENAI_API_KEY`: Your OpenAI API key (required)
- `OPENAI_MODEL`: Model to use (default: gpt-4o)
- `DATABASE_PATH`: SQLite database location
- `LOG_LEVEL`: Logging level (DEBUG, INFO, WARNING, ERROR)
- `MAX_PARALLEL_TASKS`: Max concurrent stock analyses

### Model Selection

Supported models:
- `gpt-4o` (recommended): Best balance of cost and quality
- `gpt-4-turbo`: Higher quality, higher cost
- `gpt-3.5-turbo`: Lower cost, faster, less accurate

## 📈 Performance

- **Single Stock Analysis**: 30-60 seconds
- **Multiple Stocks** (parallel): 40-90 seconds for 5 stocks
- **Data Caching**: 24-hour cache reduces repeated API calls
- **Database Logging**: All analyses logged for history

## 🛡️ Error Handling

The system includes comprehensive error handling:
- Retry logic with exponential backoff
- Graceful degradation when data is missing
- Fallback analysis methods
- Detailed error logging
- Database transaction management

## 🔐 Security Best Practices

- Never commit `.env` file
- API keys stored in environment variables
- Input validation and sanitization
- SQL injection prevention (parameterized queries)
- Rate limiting for external APIs

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new features
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## ⚠️ Disclaimer

**IMPORTANT**: This system is provided for **research and educational purposes only**.

- This is **NOT** investment advice
- The system is **NOT** a licensed financial advisor
- All information may be inaccurate, incomplete, or outdated
- Past performance does not guarantee future results
- Investing involves risk, including loss of principal
- **Always consult a licensed financial advisor** before making investment decisions
- Do your own research and due diligence

## 🙏 Acknowledgments

- Built with [LangGraph](https://github.com/langchain-ai/langgraph) for agent orchestration
- Powered by [OpenAI](https://openai.com/) GPT models
- Data from [Yahoo Finance](https://finance.yahoo.com/) via yfinance

## 📞 Support

For issues, questions, or contributions:
- Open an issue on GitHub
- Check existing documentation
- Review the examples in this README

---

**Remember**: This is a research tool. Never make investment decisions based solely on AI analysis.
