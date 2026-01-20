"""Pydantic schemas for data validation throughout the system."""

from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any, Literal
from datetime import datetime
import pandas as pd


class BasicInfo(BaseModel):
    """Basic company information."""
    name: str
    sector: str
    industry: str
    market_cap: float
    employees: Optional[int] = None


class PriceData(BaseModel):
    """Price and historical data."""
    current_price: float
    historical: Optional[Any] = None  # DataFrame - can't validate directly
    high_52_week: Optional[float] = None
    low_52_week: Optional[float] = None


class Financials(BaseModel):
    """Financial statements data."""
    income_statement: Optional[Dict[str, Any]] = None
    balance_sheet: Optional[Dict[str, Any]] = None
    cash_flow: Optional[Dict[str, Any]] = None


class NewsArticle(BaseModel):
    """News article information."""
    headline: str
    source: Optional[str] = None
    published: Optional[str] = None
    url: Optional[str] = None
    summary: Optional[str] = None


class CollectedData(BaseModel):
    """Complete data collected for a stock."""
    ticker: str
    timestamp: datetime
    basic_info: BasicInfo
    price_data: PriceData
    financials: Financials
    news: List[NewsArticle] = []

    class Config:
        arbitrary_types_allowed = True


class FundamentalMetrics(BaseModel):
    """Fundamental analysis metrics."""
    profitability: Dict[str, float]
    growth: Dict[str, float]
    efficiency: Dict[str, float]
    health: Dict[str, float]
    valuation: Dict[str, float]


class FundamentalAnalysis(BaseModel):
    """Fundamental analysis results."""
    ticker: str
    score: float = Field(ge=0, le=10)
    metrics: FundamentalMetrics
    strengths: List[str]
    weaknesses: List[str]
    recommendation: Literal["strong_buy", "buy", "hold", "sell", "strong_sell"]
    confidence: float = Field(ge=0, le=1)
    reasoning: str = ""


class TechnicalIndicators(BaseModel):
    """Technical analysis indicators."""
    RSI: Optional[float] = None
    MACD: Optional[Dict[str, Any]] = None
    SMA_50: Optional[float] = None
    SMA_200: Optional[float] = None
    EMA_12: Optional[float] = None
    EMA_26: Optional[float] = None
    golden_death_cross: Optional[str] = None


class TechnicalSignals(BaseModel):
    """Trading signals from technical analysis."""
    short_term: Literal["buy", "sell", "hold"]
    medium_term: Literal["buy", "sell", "hold"]
    long_term: Literal["buy", "sell", "hold"]


class SupportResistance(BaseModel):
    """Support and resistance levels."""
    support_levels: List[float] = []
    resistance_levels: List[float] = []


class TechnicalAnalysis(BaseModel):
    """Technical analysis results."""
    ticker: str
    trend: Literal["bullish", "bearish", "neutral"]
    indicators: TechnicalIndicators
    signals: TechnicalSignals
    support_resistance: SupportResistance
    confidence: float = Field(ge=0, le=1)
    reasoning: str = ""


class SentimentNewsItem(BaseModel):
    """Analyzed news item with sentiment."""
    headline: str
    sentiment: Literal["positive", "negative", "neutral"]
    impact: Literal["high", "medium", "low"]
    source: Optional[str] = None
    date: Optional[str] = None
    key_themes: List[str] = []


class SentimentAnalysis(BaseModel):
    """Sentiment analysis results."""
    ticker: str
    sentiment_score: float = Field(ge=-1, le=1)
    news_summary: List[SentimentNewsItem]
    overall_mood: Literal["positive", "negative", "neutral"]
    trending_themes: List[str] = []
    confidence: float = Field(ge=0, le=1)
    reasoning: str = ""


class ArgumentCase(BaseModel):
    """Bull or Bear argument case."""
    summary: str
    key_points: List[str]
    supporting_evidence: List[str]
    strength: float = Field(ge=0, le=1)


class Conflict(BaseModel):
    """Identified conflict between agent analyses."""
    issue: str
    agents_involved: List[str]
    resolution: str


class DebateAnalysis(BaseModel):
    """Debate agent results."""
    ticker: str
    bull_case: ArgumentCase
    bear_case: ArgumentCase
    conflicts: List[Conflict] = []
    final_recommendation: Literal["strong_buy", "buy", "hold", "sell", "strong_sell"]
    confidence_level: float = Field(ge=0, le=1)
    reasoning: str = ""


class RiskFactor(BaseModel):
    """Individual risk factor."""
    type: str
    level: Literal["low", "medium", "high"]
    description: str
    mitigation: str


class VolatilityMetrics(BaseModel):
    """Volatility and risk metrics."""
    beta: Optional[float] = None
    std_dev: Optional[float] = None
    var_95: Optional[float] = None


class RiskAnalysis(BaseModel):
    """Risk assessment results."""
    ticker: str
    risk_score: float = Field(ge=0, le=10)
    risk_level: Literal["low", "medium", "high", "very_high"]
    risk_factors: List[RiskFactor]
    volatility_metrics: VolatilityMetrics
    warnings: List[str] = []
    max_position_size: str
    reasoning: str = ""


class SynthesisReport(BaseModel):
    """Final synthesis report."""
    report_id: str
    ticker: str
    generated_at: datetime
    overall_score: float = Field(ge=0, le=10)
    recommendation: Literal["strong_buy", "buy", "hold", "sell", "strong_sell"]
    markdown_report: str
    json_summary: Dict[str, Any]
    saved_to_db: bool = False


class FeedbackRequest(BaseModel):
    """Request for additional data or analysis."""
    agent: str
    request_type: str
    parameters: Dict[str, Any]


class FeedbackLoop(BaseModel):
    """Feedback loop results."""
    needs_more_data: bool
    missing_info: List[str] = []
    additional_requests: List[FeedbackRequest] = []
    retry_count: int = 0


class ExecutionStrategy(BaseModel):
    """Execution strategy for analysis."""
    mode: Literal["sequential", "parallel"]
    tickers: List[str]
    analysis_type: str


class StockAnalysis(BaseModel):
    """Complete analysis for a single stock."""
    ticker: str
    collected_data: Optional[CollectedData] = None
    fundamental: Optional[FundamentalAnalysis] = None
    technical: Optional[TechnicalAnalysis] = None
    sentiment: Optional[SentimentAnalysis] = None
    debate: Optional[DebateAnalysis] = None
    risk: Optional[RiskAnalysis] = None
    synthesis: Optional[SynthesisReport] = None


class AnalysisResult(BaseModel):
    """Final analysis result returned to user."""
    status: Literal["success", "error", "partial"]
    analyses: List[StockAnalysis]
    execution_time: float
    error_message: Optional[str] = None
