"""Debate Agent - Synthesizes bull and bear cases from all analyses."""

from typing import Dict, Any, List
from src.agents.base_agent import BaseAgent
from src.models.prompts import DEBATE_AGENT_PROMPT
from src.utils.validators import validate_confidence


class DebateAgent(BaseAgent):
    """
    Agent responsible for creating balanced bull and bear investment cases.

    Synthesizes all previous analyses (fundamental, technical, sentiment)
    to create comprehensive arguments for and against investment.
    """

    def __init__(self, openai_client, db_client):
        """Initialize Debate Agent."""
        super().__init__("debate_agent", openai_client, db_client)

    async def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute debate analysis.

        Args:
            inputs: Dictionary with 'ticker', 'fundamental', 'technical', 'sentiment'

        Returns:
            Debate analysis results with bull and bear cases
        """
        ticker = inputs.get("ticker", "").upper()
        fundamental = inputs.get("fundamental", {})
        technical = inputs.get("technical", {})
        sentiment = inputs.get("sentiment", {})

        self.logger.info(f"Creating investment debate for {ticker}")

        # Create debate analysis using LLM
        analysis = await self._create_debate_with_llm(
            ticker,
            fundamental,
            technical,
            sentiment
        )

        return {
            "ticker": ticker,
            "bull_case": analysis.get("bull_case", {}),
            "bear_case": analysis.get("bear_case", {}),
            "conflicts": analysis.get("conflicts", []),
            "final_recommendation": analysis.get("final_recommendation", "hold"),
            "confidence_level": validate_confidence(analysis.get("confidence_level", 0.5)),
            "reasoning": analysis.get("reasoning", "")
        }

    async def _create_debate_with_llm(
        self,
        ticker: str,
        fundamental: Dict[str, Any],
        technical: Dict[str, Any],
        sentiment: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Create bull/bear debate using LLM.

        Args:
            ticker: Stock ticker
            fundamental: Fundamental analysis results
            technical: Technical analysis results
            sentiment: Sentiment analysis results

        Returns:
            Debate analysis dictionary
        """
        # Prepare summary of all analyses
        analyses_summary = {
            "ticker": ticker,
            "fundamental": {
                "score": fundamental.get("score"),
                "recommendation": fundamental.get("recommendation"),
                "strengths": fundamental.get("strengths", []),
                "weaknesses": fundamental.get("weaknesses", []),
                "confidence": fundamental.get("confidence")
            },
            "technical": {
                "trend": technical.get("trend"),
                "signals": technical.get("signals", {}),
                "indicators": {
                    "RSI": technical.get("indicators", {}).get("RSI"),
                    "MACD": technical.get("indicators", {}).get("MACD"),
                    "trend_signal": technical.get("indicators", {}).get("golden_death_cross")
                },
                "confidence": technical.get("confidence")
            },
            "sentiment": {
                "sentiment_score": sentiment.get("sentiment_score"),
                "overall_mood": sentiment.get("overall_mood"),
                "trending_themes": sentiment.get("trending_themes", []),
                "confidence": sentiment.get("confidence")
            }
        }

        messages = [
            {"role": "system", "content": DEBATE_AGENT_PROMPT},
            {"role": "user", "content": f"""
Create a balanced investment debate for {ticker} based on the following analyses:

{self._format_data_for_prompt(analyses_summary)}

Provide your debate analysis in JSON format with the following structure:
{{
    "bull_case": {{
        "summary": "<concise bull case summary>",
        "key_points": ["<point 1>", "<point 2>", ...],
        "supporting_evidence": ["<evidence 1>", "<evidence 2>", ...],
        "strength": <float 0-1>
    }},
    "bear_case": {{
        "summary": "<concise bear case summary>",
        "key_points": ["<point 1>", "<point 2>", ...],
        "supporting_evidence": ["<evidence 1>", "<evidence 2>", ...],
        "strength": <float 0-1>
    }},
    "conflicts": [
        {{
            "issue": "<description of conflict>",
            "agents_involved": ["<agent1>", "<agent2>"],
            "resolution": "<how to resolve this conflict>"
        }}
    ],
    "final_recommendation": "<strong_buy|buy|hold|sell|strong_sell>",
    "confidence_level": <float 0-1>,
    "reasoning": "<explanation of final recommendation>"
}}

Be objective and consider all perspectives. Identify real conflicts between analyses.
"""}
        ]

        try:
            response = await self._call_llm_json(messages, temperature=0.7, max_tokens=3000)
            return response
        except Exception as e:
            self.logger.error(f"LLM debate analysis failed: {e}")

            # Fallback to simple logic
            return self._fallback_debate_analysis(fundamental, technical, sentiment)

    def _fallback_debate_analysis(
        self,
        fundamental: Dict[str, Any],
        technical: Dict[str, Any],
        sentiment: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Simple fallback debate analysis.

        Args:
            fundamental: Fundamental analysis
            technical: Technical analysis
            sentiment: Sentiment analysis

        Returns:
            Basic debate structure
        """
        # Extract key information
        fund_rec = fundamental.get("recommendation", "hold")
        tech_trend = technical.get("trend", "neutral")
        sent_mood = sentiment.get("overall_mood", "neutral")

        # Build simple bull case
        bull_points = []
        if fund_rec in ["buy", "strong_buy"]:
            bull_points.extend(fundamental.get("strengths", [])[:3])
        if tech_trend == "bullish":
            bull_points.append("Technical indicators show bullish trend")
        if sent_mood == "positive":
            bull_points.append("Positive market sentiment")

        # Build simple bear case
        bear_points = []
        if fund_rec in ["sell", "strong_sell"]:
            bear_points.extend(fundamental.get("weaknesses", [])[:3])
        if tech_trend == "bearish":
            bear_points.append("Technical indicators show bearish trend")
        if sent_mood == "negative":
            bear_points.append("Negative market sentiment")

        # Identify conflicts
        conflicts = []
        if (fund_rec in ["buy", "strong_buy"] and tech_trend == "bearish") or \
           (fund_rec in ["sell", "strong_sell"] and tech_trend == "bullish"):
            conflicts.append({
                "issue": "Fundamental and technical analyses disagree",
                "agents_involved": ["fundamental_analyst", "technical_analyst"],
                "resolution": "Consider longer-term fundamental strength vs short-term technical weakness"
            })

        # Determine final recommendation
        buy_votes = sum([
            1 if fund_rec in ["buy", "strong_buy"] else 0,
            1 if tech_trend == "bullish" else 0,
            1 if sent_mood == "positive" else 0
        ])

        sell_votes = sum([
            1 if fund_rec in ["sell", "strong_sell"] else 0,
            1 if tech_trend == "bearish" else 0,
            1 if sent_mood == "negative" else 0
        ])

        if buy_votes >= 2:
            final_rec = "buy"
        elif sell_votes >= 2:
            final_rec = "sell"
        else:
            final_rec = "hold"

        bull_strength = len(bull_points) / 5.0
        bear_strength = len(bear_points) / 5.0

        return {
            "bull_case": {
                "summary": "Investment case based on positive indicators",
                "key_points": bull_points if bull_points else ["No strong bullish signals"],
                "supporting_evidence": bull_points,
                "strength": bull_strength
            },
            "bear_case": {
                "summary": "Investment concerns based on negative indicators",
                "key_points": bear_points if bear_points else ["No strong bearish signals"],
                "supporting_evidence": bear_points,
                "strength": bear_strength
            },
            "conflicts": conflicts,
            "final_recommendation": final_rec,
            "confidence_level": 0.5,
            "reasoning": f"Fallback analysis: {buy_votes} bullish vs {sell_votes} bearish signals"
        }
