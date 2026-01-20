"""Feedback Loop Agent - Identifies gaps and requests additional analysis."""

from typing import Dict, Any, List
from src.agents.base_agent import BaseAgent
from src.models.prompts import FEEDBACK_LOOP_PROMPT


class FeedbackLoopAgent(BaseAgent):
    """
    Agent responsible for quality assurance and identifying analysis gaps.

    Reviews all agent outputs to detect missing data, inconsistencies,
    or areas needing deeper analysis.
    """

    def __init__(self, openai_client, db_client):
        """Initialize Feedback Loop Agent."""
        super().__init__("feedback_loop", openai_client, db_client)

    async def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute feedback loop analysis.

        Args:
            inputs: Dictionary with all agent outputs and synthesis

        Returns:
            Feedback analysis with any additional requests
        """
        ticker = inputs.get("ticker", "").upper()
        synthesis = inputs.get("synthesis", {})
        collected_data = inputs.get("collected_data", {})
        fundamental = inputs.get("fundamental", {})
        technical = inputs.get("technical", {})
        sentiment = inputs.get("sentiment", {})
        debate = inputs.get("debate", {})
        risk = inputs.get("risk", {})

        self.logger.info(f"Running feedback loop analysis for {ticker}")

        # Check for data quality issues
        missing_info = self._identify_missing_data(
            collected_data,
            fundamental,
            technical,
            sentiment
        )

        # Check for low confidence areas
        low_confidence = self._identify_low_confidence_areas(
            fundamental,
            technical,
            sentiment,
            debate,
            risk
        )

        # Check for unresolved conflicts
        unresolved_conflicts = self._check_unresolved_conflicts(debate)

        # Determine if more data/analysis is needed
        needs_more_data = len(missing_info) > 0 or len(low_confidence) > 0

        # Generate additional requests if needed
        additional_requests = []
        if needs_more_data:
            additional_requests = self._generate_additional_requests(
                missing_info,
                low_confidence
            )

        return {
            "needs_more_data": needs_more_data,
            "missing_info": missing_info,
            "low_confidence_areas": low_confidence,
            "unresolved_conflicts": unresolved_conflicts,
            "additional_requests": additional_requests,
            "retry_count": inputs.get("retry_count", 0),
            "confidence": 0.8,
            "reasoning": self._generate_reasoning(
                needs_more_data,
                missing_info,
                low_confidence
            )
        }

    def _identify_missing_data(
        self,
        data: Dict[str, Any],
        fundamental: Dict[str, Any],
        technical: Dict[str, Any],
        sentiment: Dict[str, Any]
    ) -> List[str]:
        """
        Identify missing or insufficient data.

        Args:
            data: Collected data
            fundamental: Fundamental analysis
            technical: Technical analysis
            sentiment: Sentiment analysis

        Returns:
            List of missing data items
        """
        missing = []

        # Check collected data
        if not data or "error" in data:
            missing.append("Basic stock data unavailable")

        price_data = data.get("price_data", {})
        if not price_data.get("historical_data"):
            missing.append("Historical price data missing")

        financials = data.get("financials", {})
        if not financials.get("income_statement"):
            missing.append("Income statement data missing")
        if not financials.get("balance_sheet"):
            missing.append("Balance sheet data missing")

        # Check fundamental analysis
        fund_metrics = fundamental.get("metrics", {})
        if not fund_metrics or all("error" in v for v in fund_metrics.values() if isinstance(v, dict)):
            missing.append("Insufficient financial metrics for fundamental analysis")

        # Check technical analysis
        if "error" in technical:
            missing.append("Technical analysis could not be completed")

        tech_indicators = technical.get("indicators", {})
        if not tech_indicators or len(tech_indicators) < 3:
            missing.append("Insufficient technical indicators calculated")

        # Check sentiment analysis
        news = sentiment.get("news_summary", [])
        if not news or len(news) < 3:
            missing.append("Limited news data for sentiment analysis")

        return missing

    def _identify_low_confidence_areas(
        self,
        fundamental: Dict[str, Any],
        technical: Dict[str, Any],
        sentiment: Dict[str, Any],
        debate: Dict[str, Any],
        risk: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Identify areas with low confidence scores.

        Args:
            All agent analysis results

        Returns:
            List of low confidence areas
        """
        low_confidence = []
        threshold = 0.6

        if fundamental.get("confidence", 1.0) < threshold:
            low_confidence.append({
                "agent": "fundamental_analyst",
                "confidence": fundamental.get("confidence"),
                "reason": "Low confidence in fundamental analysis"
            })

        if technical.get("confidence", 1.0) < threshold:
            low_confidence.append({
                "agent": "technical_analyst",
                "confidence": technical.get("confidence"),
                "reason": "Low confidence in technical analysis"
            })

        if sentiment.get("confidence", 1.0) < threshold:
            low_confidence.append({
                "agent": "sentiment_analyst",
                "confidence": sentiment.get("confidence"),
                "reason": "Low confidence in sentiment analysis"
            })

        if debate.get("confidence_level", 1.0) < threshold:
            low_confidence.append({
                "agent": "debate_agent",
                "confidence": debate.get("confidence_level"),
                "reason": "Low confidence in final recommendation"
            })

        return low_confidence

    def _check_unresolved_conflicts(self, debate: Dict[str, Any]) -> List[str]:
        """
        Check for unresolved conflicts in the debate.

        Args:
            debate: Debate analysis

        Returns:
            List of unresolved conflict descriptions
        """
        conflicts = debate.get("conflicts", [])
        unresolved = []

        for conflict in conflicts:
            resolution = conflict.get("resolution", "")
            # Check if resolution is vague or incomplete
            if not resolution or len(resolution) < 20 or "unclear" in resolution.lower():
                unresolved.append(conflict.get("issue", "Unknown conflict"))

        return unresolved

    def _generate_additional_requests(
        self,
        missing_info: List[str],
        low_confidence: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Generate requests for additional data or analysis.

        Args:
            missing_info: List of missing data
            low_confidence: List of low confidence areas

        Returns:
            List of additional requests
        """
        requests = []

        # Requests based on missing data
        if "Historical price data missing" in missing_info:
            requests.append({
                "agent": "data_collector",
                "request_type": "fetch_historical_data",
                "parameters": {"period": "1y", "interval": "1d"},
                "priority": "high"
            })

        if "Income statement data missing" in missing_info or "Balance sheet data missing" in missing_info:
            requests.append({
                "agent": "data_collector",
                "request_type": "fetch_financials",
                "parameters": {},
                "priority": "high"
            })

        if "Limited news data for sentiment analysis" in missing_info:
            requests.append({
                "agent": "data_collector",
                "request_type": "fetch_more_news",
                "parameters": {"count": 20},
                "priority": "medium"
            })

        # Requests based on low confidence
        for area in low_confidence:
            agent = area.get("agent")
            if agent == "fundamental_analyst":
                requests.append({
                    "agent": "fundamental_analyst",
                    "request_type": "detailed_analysis",
                    "parameters": {"focus": "financial_health"},
                    "priority": "medium"
                })
            elif agent == "technical_analyst":
                requests.append({
                    "agent": "technical_analyst",
                    "request_type": "extended_analysis",
                    "parameters": {"additional_indicators": ["bollinger_bands", "fibonacci"]},
                    "priority": "low"
                })

        return requests

    def _generate_reasoning(
        self,
        needs_more_data: bool,
        missing_info: List[str],
        low_confidence: List[Dict[str, Any]]
    ) -> str:
        """Generate reasoning explanation."""
        if not needs_more_data:
            return "Analysis is complete and comprehensive. No additional data needed."

        reasons = []
        if missing_info:
            reasons.append(f"{len(missing_info)} data gaps identified")
        if low_confidence:
            reasons.append(f"{len(low_confidence)} areas with low confidence")

        return f"Additional analysis recommended: {', '.join(reasons)}. However, current analysis provides sufficient information for initial assessment."
