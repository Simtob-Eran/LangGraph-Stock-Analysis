"""
Example: Modern LangChain 0.3.x Chain Implementation

This file demonstrates how to implement a stock analysis agent using
modern LangChain patterns instead of custom agent implementations.

Compare this to the old src/agents/fundamental_analyst.py to see the difference.
"""

from typing import List, Dict, Any, Literal
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import (
    RunnablePassthrough,
    RunnableLambda,
    RunnableParallel
)


# ============================================================================
# STEP 1: Define Output Schema with Pydantic
# ============================================================================

class FundamentalMetrics(BaseModel):
    """Financial metrics calculated from raw data."""
    profitability: Dict[str, float] = Field(description="Profitability ratios")
    growth: Dict[str, float] = Field(description="Growth metrics")
    health: Dict[str, float] = Field(description="Financial health indicators")
    valuation: Dict[str, float] = Field(description="Valuation metrics")


class FundamentalAnalysisOutput(BaseModel):
    """
    Structured output for fundamental analysis.

    This replaces manual JSON parsing with automatic Pydantic validation.
    """
    ticker: str = Field(description="Stock ticker symbol")
    score: float = Field(
        ge=0,
        le=10,
        description="Overall fundamental strength score"
    )
    metrics: FundamentalMetrics
    strengths: List[str] = Field(
        min_items=3,
        max_items=5,
        description="Key competitive advantages and strengths"
    )
    weaknesses: List[str] = Field(
        min_items=3,
        max_items=5,
        description="Key concerns and weaknesses"
    )
    recommendation: Literal["strong_buy", "buy", "hold", "sell", "strong_sell"]
    confidence: float = Field(ge=0, le=1, description="Analysis confidence level")
    reasoning: str = Field(description="Detailed explanation of the analysis")


# ============================================================================
# STEP 2: Pure Python Calculator (No LLM)
# ============================================================================

class FinancialMetricsCalculator:
    """
    Pure Python calculator for financial metrics.

    No LLM calls here - just mathematical computations.
    Separating calculation from analysis makes code cleaner and testable.
    """

    @staticmethod
    def calculate_all_metrics(financial_data: Dict[str, Any]) -> FundamentalMetrics:
        """
        Calculate all financial metrics from raw data.

        Args:
            financial_data: Raw financial data from yfinance

        Returns:
            FundamentalMetrics with all calculated ratios
        """
        return FundamentalMetrics(
            profitability=FinancialMetricsCalculator._calc_profitability(financial_data),
            growth=FinancialMetricsCalculator._calc_growth(financial_data),
            health=FinancialMetricsCalculator._calc_health(financial_data),
            valuation=FinancialMetricsCalculator._calc_valuation(financial_data)
        )

    @staticmethod
    def _calc_profitability(data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate profitability metrics."""
        income_stmt = data.get("income_statement", {})
        balance_sheet = data.get("balance_sheet", {})

        # Example calculations (simplified)
        revenue = income_stmt.get("revenue", 1)
        net_income = income_stmt.get("net_income", 0)
        total_assets = balance_sheet.get("total_assets", 1)

        return {
            "net_profit_margin": (net_income / revenue * 100) if revenue else 0,
            "roa": (net_income / total_assets * 100) if total_assets else 0,
        }

    @staticmethod
    def _calc_growth(data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate growth metrics."""
        # Simplified example
        return {
            "revenue_growth_yoy": 6.5,
            "earnings_growth_yoy": 19.5
        }

    @staticmethod
    def _calc_health(data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate financial health metrics."""
        # Simplified example
        return {
            "current_ratio": 0.89,
            "debt_to_equity": 1.34
        }

    @staticmethod
    def _calc_valuation(data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate valuation metrics."""
        # Simplified example
        return {
            "pe_ratio": 33.1
        }


# ============================================================================
# STEP 3: Create Modern LangChain Chain
# ============================================================================

def create_fundamental_analyst_chain():
    """
    Create a modern LangChain chain for fundamental analysis.

    This replaces the old custom BaseAgent implementation.
    Uses LCEL (LangChain Expression Language) for composition.
    """
    from config.settings import settings

    # Initialize LLM with structured output support
    # Model comes from environment configuration
    llm = ChatOpenAI(
        model=settings.OPENAI_MODEL,
        temperature=0.7
    )

    # Initialize calculator
    calculator = FinancialMetricsCalculator()

    # Define prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a fundamental analysis expert specializing in evaluating companies' financial health and intrinsic value.

Analyze the provided financial metrics and data to produce a comprehensive fundamental analysis including:
1. Overall score (0-10) based on fundamental strength
2. Identify 3-5 key strengths
3. Identify 3-5 key weaknesses
4. Provide a clear recommendation (strong_buy, buy, hold, sell, strong_sell)
5. Assess your confidence level (0-1)
6. Explain your reasoning in detail

Be objective and data-driven. Consider profitability, growth, financial health, and valuation."""),
        ("human", """Analyze the following data for {ticker}:

Company: {company_name}
Sector: {sector}
Market Cap: ${market_cap:,.0f}

Calculated Metrics:
{metrics_json}

Provide your analysis in the requested format.""")
    ])

    # Helper function to format data
    def format_metrics_for_prompt(state: Dict[str, Any]) -> str:
        """Format metrics into readable JSON string."""
        import json
        metrics = state["calculated_metrics"]
        return json.dumps(metrics.dict(), indent=2)

    # Build the chain using LCEL
    # This is the modern way to compose LangChain components
    chain = (
        # Step 1: Calculate metrics (pure Python)
        RunnablePassthrough.assign(
            calculated_metrics=RunnableLambda(
                lambda x: calculator.calculate_all_metrics(x["financial_data"])
            )
        )
        # Step 2: Format metrics for prompt
        | RunnablePassthrough.assign(
            metrics_json=RunnableLambda(format_metrics_for_prompt)
        )
        # Step 3: Generate prompt
        | prompt
        # Step 4: Call LLM with structured output
        | llm.with_structured_output(FundamentalAnalysisOutput)
    )

    return chain


# ============================================================================
# STEP 4: Add Error Handling and Retry Logic
# ============================================================================

def create_robust_fundamental_chain():
    """
    Create chain with built-in error handling and retry logic.

    This replaces custom retry implementations with LangChain's built-in features.
    """

    # Create base chain
    base_chain = create_fundamental_analyst_chain()

    # Add retry logic (automatic retries on failure)
    chain_with_retry = base_chain.with_retry(
        retry_if_exception_type=(Exception,),
        stop_after_attempt=3,
        wait_exponential_jitter=True
    )

    # Add fallback (simpler analysis if main chain fails)
    def create_fallback_analysis(state: Dict[str, Any]) -> FundamentalAnalysisOutput:
        """Create basic analysis when LLM fails."""
        return FundamentalAnalysisOutput(
            ticker=state["ticker"],
            score=5.0,
            metrics=FundamentalMetrics(
                profitability={},
                growth={},
                health={},
                valuation={}
            ),
            strengths=["Unable to analyze"],
            weaknesses=["Analysis failed"],
            recommendation="hold",
            confidence=0.3,
            reasoning="Fallback analysis due to error"
        )

    fallback = RunnableLambda(create_fallback_analysis)

    # Combine with fallback
    robust_chain = chain_with_retry.with_fallbacks([fallback])

    return robust_chain


# ============================================================================
# STEP 5: Usage Example
# ============================================================================

async def main():
    """
    Example usage of the modern chain.

    Compare this to the old agent usage:

    OLD:
        agent = FundamentalAnalystAgent(openai_client, db_client)
        result = await agent.run({"ticker": "AAPL", "collected_data": data})

    NEW (cleaner, more composable):
        chain = create_robust_fundamental_chain()
        result = await chain.ainvoke({
            "ticker": "AAPL",
            "company_name": "Apple Inc.",
            "sector": "Technology",
            "market_cap": 3700000000000,
            "financial_data": collected_data["financials"]
        })
    """

    # Create the chain
    chain = create_robust_fundamental_chain()

    # Prepare input data
    input_data = {
        "ticker": "AAPL",
        "company_name": "Apple Inc.",
        "sector": "Technology",
        "market_cap": 3700000000000,
        "financial_data": {
            "income_statement": {...},  # Actual data here
            "balance_sheet": {...},
            "cash_flow": {...}
        }
    }

    # Invoke the chain (automatically handles retry, fallback, structured output)
    result = await chain.ainvoke(input_data)

    # Result is automatically validated Pydantic model
    print(f"Ticker: {result.ticker}")
    print(f"Score: {result.score}/10")
    print(f"Recommendation: {result.recommendation}")
    print(f"Confidence: {result.confidence:.0%}")
    print(f"\nStrengths:")
    for strength in result.strengths:
        print(f"  - {strength}")
    print(f"\nWeaknesses:")
    for weakness in result.weaknesses:
        print(f"  - {weakness}")


# ============================================================================
# STEP 6: Parallel Execution Example
# ============================================================================

def create_parallel_analysis_chain():
    """
    Example of running multiple analyses in parallel.

    This would replace sequential execution in the orchestrator.
    """

    fundamental_chain = create_fundamental_analyst_chain()
    # technical_chain = create_technical_analyst_chain()  # Similar implementation
    # sentiment_chain = create_sentiment_analyst_chain()  # Similar implementation

    # Run all three analyses in parallel
    parallel_chain = RunnableParallel(
        fundamental=fundamental_chain,
        # technical=technical_chain,
        # sentiment=sentiment_chain
    )

    return parallel_chain


# ============================================================================
# COMPARISON SUMMARY
# ============================================================================

"""
OLD APPROACH (src/agents/fundamental_analyst.py):
    - Custom BaseAgent class (~200 lines)
    - Manual LLM calls with openai_client
    - Manual JSON parsing
    - Custom retry logic
    - Custom error handling
    - Manual validation

NEW APPROACH (this file):
    - LangChain Runnable chain (~150 lines total, more reusable)
    - Automatic LLM calls via chain
    - Automatic Pydantic validation
    - Built-in retry (with_retry())
    - Built-in fallback (with_fallbacks())
    - Automatic structured output

BENEFITS:
    ✓ 30% less code
    ✓ Better type safety
    ✓ Easier testing
    ✓ More composable
    ✓ Framework best practices
    ✓ Automatic error handling
    ✓ Better maintainability

CODE QUALITY:
    - Old: Good, but custom implementation
    - New: Excellent, follows LangChain patterns

MIGRATION EFFORT:
    - Medium (2-3 days for all agents)
    - High value for long-term maintainability

RECOMMENDATION:
    Current code works fine. Modernize when:
    1. Adding new features
    2. Refactoring for maintainability
    3. Team wants to follow latest patterns
"""


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
