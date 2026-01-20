"""Sentiment Analyst Agent - Analyzes news and market sentiment."""

from typing import Dict, Any, List
from src.agents.base_agent import BaseAgent
from src.models.prompts import SENTIMENT_ANALYST_PROMPT
from src.utils.validators import validate_confidence


class SentimentAnalystAgent(BaseAgent):
    """
    Agent responsible for analyzing news sentiment and market psychology.

    Processes news articles to determine overall market sentiment
    and identifies key themes affecting the stock.
    """

    def __init__(self, openai_client, db_client):
        """Initialize Sentiment Analyst Agent."""
        super().__init__("sentiment_analyst", openai_client, db_client)

    async def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute sentiment analysis.

        Args:
            inputs: Dictionary with 'ticker' and 'collected_data'

        Returns:
            Sentiment analysis results
        """
        ticker = inputs.get("ticker", "").upper()
        collected_data = inputs.get("collected_data", {})

        if not collected_data:
            return {
                "ticker": ticker,
                "error": "No data available for analysis",
                "confidence": 0.0,
                "reasoning": "Missing collected data"
            }

        self.logger.info(f"Performing sentiment analysis for {ticker}")

        # Get news articles
        news = collected_data.get("news", [])

        if not news:
            return {
                "ticker": ticker,
                "sentiment_score": 0.0,
                "news_summary": [],
                "overall_mood": "neutral",
                "trending_themes": [],
                "confidence": 0.3,
                "reasoning": "No news articles available for sentiment analysis"
            }

        # Analyze news with LLM
        analysis = await self._analyze_news_with_llm(ticker, news)

        return {
            "ticker": ticker,
            "sentiment_score": self._validate_sentiment_score(analysis.get("sentiment_score", 0.0)),
            "news_summary": analysis.get("news_summary", []),
            "overall_mood": analysis.get("overall_mood", "neutral"),
            "trending_themes": analysis.get("trending_themes", []),
            "confidence": validate_confidence(analysis.get("confidence", 0.7)),
            "reasoning": analysis.get("reasoning", "")
        }

    def _validate_sentiment_score(self, score: float) -> float:
        """Validate and clamp sentiment score to [-1, 1]."""
        return max(-1.0, min(1.0, score))

    async def _analyze_news_with_llm(
        self,
        ticker: str,
        news: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Analyze news articles using LLM.

        Args:
            ticker: Stock ticker
            news: List of news articles

        Returns:
            Analysis dictionary
        """
        # Prepare news summary for LLM
        news_text = []
        for i, article in enumerate(news[:10], 1):  # Limit to 10 articles
            headline = article.get("headline", "")
            summary = article.get("summary", "")
            source = article.get("source", "Unknown")
            published = article.get("published", "")

            news_text.append(f"""
Article {i}:
Headline: {headline}
Source: {source}
Published: {published}
Summary: {summary}
""")

        news_content = "\n".join(news_text)

        messages = [
            {"role": "system", "content": SENTIMENT_ANALYST_PROMPT},
            {"role": "user", "content": f"""
Analyze the sentiment of the following news articles for {ticker}:

{news_content}

Provide your analysis in JSON format with the following structure:
{{
    "sentiment_score": <float from -1 (very negative) to 1 (very positive)>,
    "overall_mood": "<positive|negative|neutral>",
    "news_summary": [
        {{
            "headline": "<headline>",
            "sentiment": "<positive|negative|neutral>",
            "impact": "<high|medium|low>",
            "source": "<source>",
            "date": "<date>",
            "key_themes": ["<theme1>", "<theme2>"]
        }}
    ],
    "trending_themes": ["<theme1>", "<theme2>", ...],
    "confidence": <float 0-1>,
    "reasoning": "<explanation of sentiment analysis>"
}}

Analyze each article individually first, then provide an overall sentiment score.
"""}
        ]

        try:
            response = await self._call_llm_json(messages, temperature=0.7, max_tokens=3000)
            return response
        except Exception as e:
            self.logger.error(f"LLM sentiment analysis failed: {e}")

            # Fallback: Simple keyword-based analysis
            return self._fallback_sentiment_analysis(news)

    def _fallback_sentiment_analysis(self, news: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Simple keyword-based sentiment analysis as fallback.

        Args:
            news: List of news articles

        Returns:
            Basic sentiment analysis
        """
        positive_keywords = [
            'beats', 'growth', 'profit', 'innovation', 'expansion', 'partnership',
            'breakthrough', 'record', 'strong', 'surpasses', 'exceeds', 'gains',
            'rises', 'upgrade', 'bullish', 'optimistic'
        ]

        negative_keywords = [
            'lawsuit', 'recall', 'investigation', 'loss', 'decline', 'warning',
            'scandal', 'downgrade', 'misses', 'falls', 'drops', 'weak', 'concern',
            'risk', 'threat', 'bearish', 'pessimistic'
        ]

        sentiment_scores = []
        news_summary = []

        for article in news[:10]:
            headline = article.get("headline", "").lower()
            summary = article.get("summary", "").lower()
            text = headline + " " + summary

            # Count keywords
            positive_count = sum(1 for kw in positive_keywords if kw in text)
            negative_count = sum(1 for kw in negative_keywords if kw in text)

            # Calculate article sentiment
            if positive_count > negative_count:
                sentiment = "positive"
                score = 0.5
            elif negative_count > positive_count:
                sentiment = "negative"
                score = -0.5
            else:
                sentiment = "neutral"
                score = 0.0

            sentiment_scores.append(score)

            # Determine impact (simple heuristic)
            impact = "high" if (positive_count + negative_count) > 2 else "medium"

            news_summary.append({
                "headline": article.get("headline", ""),
                "sentiment": sentiment,
                "impact": impact,
                "source": article.get("source"),
                "date": article.get("published"),
                "key_themes": []
            })

        # Calculate overall sentiment
        avg_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0.0

        if avg_sentiment > 0.2:
            mood = "positive"
        elif avg_sentiment < -0.2:
            mood = "negative"
        else:
            mood = "neutral"

        return {
            "sentiment_score": avg_sentiment,
            "overall_mood": mood,
            "news_summary": news_summary,
            "trending_themes": [],
            "confidence": 0.5,
            "reasoning": f"Fallback keyword-based analysis. Analyzed {len(news_summary)} articles."
        }
