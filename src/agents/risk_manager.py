"""Risk Manager Agent - Assesses investment risks and provides warnings."""

import numpy as np
import pandas as pd
from typing import Dict, Any, List
from src.agents.base_agent import BaseAgent
from src.models.prompts import RISK_MANAGER_PROMPT
from src.utils.validators import validate_score


class RiskManagerAgent(BaseAgent):
    """
    Agent responsible for comprehensive risk assessment.

    Evaluates volatility, financial risks, and specific concerns
    to provide risk scores and position size recommendations.
    """

    def __init__(self, openai_client, db_client):
        """Initialize Risk Manager Agent."""
        super().__init__("risk_manager", openai_client, db_client)

    async def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute risk assessment.

        Args:
            inputs: Dictionary with 'ticker', 'collected_data', and other analyses

        Returns:
            Risk assessment results
        """
        ticker = inputs.get("ticker", "").upper()
        collected_data = inputs.get("collected_data", {})
        fundamental = inputs.get("fundamental", {})
        technical = inputs.get("technical", {})
        sentiment = inputs.get("sentiment", {})

        self.logger.info(f"Performing risk assessment for {ticker}")

        # Calculate volatility metrics
        volatility_metrics = self._calculate_volatility_metrics(collected_data)

        # Identify risk factors
        risk_factors = self._identify_risk_factors(
            collected_data,
            fundamental,
            technical,
            sentiment,
            volatility_metrics
        )

        # Calculate overall risk score
        risk_score = self._calculate_risk_score(risk_factors, volatility_metrics)

        # Determine risk level
        risk_level = self._determine_risk_level(risk_score)

        # Use LLM for additional insights
        analysis = await self._analyze_with_llm(
            ticker,
            risk_factors,
            volatility_metrics,
            risk_score
        )

        # Determine position size recommendation
        max_position = self._recommend_position_size(risk_level)

        return {
            "ticker": ticker,
            "risk_score": validate_score(risk_score),
            "risk_level": risk_level,
            "risk_factors": risk_factors,
            "volatility_metrics": volatility_metrics,
            "warnings": analysis.get("warnings", []),
            "max_position_size": max_position,
            "reasoning": analysis.get("reasoning", "")
        }

    def _calculate_volatility_metrics(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate volatility and risk metrics.

        Args:
            data: Collected stock data

        Returns:
            Dictionary of volatility metrics
        """
        metrics = {}

        try:
            price_data = data.get("price_data", {})
            historical = price_data.get("historical_data", {})

            if not historical:
                return {"error": "No historical data for volatility calculation"}

            # Convert to DataFrame
            df = pd.DataFrame.from_dict(historical, orient='index')

            if 'Close' not in df.columns or len(df) < 30:
                return {"error": "Insufficient data"}

            # Calculate daily returns
            df['Returns'] = df['Close'].pct_change()
            returns = df['Returns'].dropna()

            # Standard deviation (annualized)
            std_dev = float(returns.std() * np.sqrt(252))
            metrics["std_dev"] = std_dev

            # Beta calculation (simplified - using market proxy)
            # For now, use relative volatility as proxy
            # In production, would compare to S&P 500
            metrics["beta"] = min(2.0, std_dev / 0.15)  # Assuming market std_dev ~0.15

            # Value at Risk (95% confidence)
            var_95 = float(returns.quantile(0.05))
            metrics["var_95"] = var_95

            # Maximum drawdown
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            max_drawdown = float(drawdown.min())
            metrics["max_drawdown"] = max_drawdown

        except Exception as e:
            self.logger.error(f"Error calculating volatility metrics: {e}")
            metrics["error"] = str(e)

        return metrics

    def _identify_risk_factors(
        self,
        data: Dict[str, Any],
        fundamental: Dict[str, Any],
        technical: Dict[str, Any],
        sentiment: Dict[str, Any],
        volatility: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Identify specific risk factors.

        Args:
            data: Collected data
            fundamental: Fundamental analysis
            technical: Technical analysis
            sentiment: Sentiment analysis
            volatility: Volatility metrics

        Returns:
            List of risk factors
        """
        risk_factors = []

        # Volatility risk
        std_dev = volatility.get("std_dev", 0)
        if std_dev > 0.4:
            risk_factors.append({
                "type": "High Volatility",
                "level": "high",
                "description": f"Stock exhibits high volatility (σ = {std_dev:.2%})",
                "mitigation": "Use smaller position sizes and consider options for hedging"
            })
        elif std_dev > 0.25:
            risk_factors.append({
                "type": "Moderate Volatility",
                "level": "medium",
                "description": f"Stock shows moderate volatility (σ = {std_dev:.2%})",
                "mitigation": "Monitor position closely and use stop losses"
            })

        # Financial health risks
        fund_metrics = fundamental.get("metrics", {})
        health = fund_metrics.get("health", {})

        debt_to_equity = health.get("debt_to_equity")
        if debt_to_equity and debt_to_equity > 2.0:
            risk_factors.append({
                "type": "High Leverage",
                "level": "high",
                "description": f"High debt-to-equity ratio ({debt_to_equity:.2f})",
                "mitigation": "Monitor credit ratings and debt refinancing plans"
            })

        current_ratio = health.get("current_ratio")
        if current_ratio and current_ratio < 1.0:
            risk_factors.append({
                "type": "Liquidity Risk",
                "level": "high",
                "description": f"Current ratio below 1.0 ({current_ratio:.2f})",
                "mitigation": "Watch for potential liquidity issues"
            })

        # Technical risks
        tech_trend = technical.get("trend", "neutral")
        if tech_trend == "bearish":
            risk_factors.append({
                "type": "Negative Technical Trend",
                "level": "medium",
                "description": "Technical indicators show bearish trend",
                "mitigation": "Wait for trend reversal or use short-term trading strategies"
            })

        # Sentiment risks
        sent_score = sentiment.get("sentiment_score", 0)
        if sent_score < -0.5:
            risk_factors.append({
                "type": "Negative Sentiment",
                "level": "medium",
                "description": "Predominantly negative news sentiment",
                "mitigation": "Understand root causes and monitor for improvement"
            })

        # Check for conflicting signals
        fund_rec = fundamental.get("recommendation", "hold")
        if fund_rec in ["buy", "strong_buy"] and tech_trend == "bearish":
            risk_factors.append({
                "type": "Analysis Conflict",
                "level": "medium",
                "description": "Fundamental and technical analyses disagree",
                "mitigation": "Consider timing and use staged entry approach"
            })

        # Market cap risk
        market_cap = data.get("basic_info", {}).get("market_cap", 0)
        if market_cap < 2_000_000_000:  # < $2B
            risk_factors.append({
                "type": "Small Cap Risk",
                "level": "medium",
                "description": "Small market capitalization increases volatility",
                "mitigation": "Limit position size and expect higher volatility"
            })

        return risk_factors

    def _calculate_risk_score(
        self,
        risk_factors: List[Dict[str, Any]],
        volatility: Dict[str, Any]
    ) -> float:
        """
        Calculate overall risk score (0-10, higher = riskier).

        Args:
            risk_factors: List of identified risks
            volatility: Volatility metrics

        Returns:
            Risk score
        """
        base_score = 5.0  # Start at medium risk

        # Adjust for number and severity of risk factors
        for factor in risk_factors:
            level = factor.get("level", "medium")
            if level == "high":
                base_score += 1.5
            elif level == "medium":
                base_score += 0.75
            else:
                base_score += 0.25

        # Adjust for volatility
        std_dev = volatility.get("std_dev", 0.2)
        if std_dev > 0.4:
            base_score += 1.5
        elif std_dev > 0.25:
            base_score += 0.75

        # Clamp to 0-10
        return validate_score(base_score)

    def _determine_risk_level(self, risk_score: float) -> str:
        """
        Determine categorical risk level.

        Args:
            risk_score: Numerical risk score

        Returns:
            Risk level category
        """
        if risk_score < 3.0:
            return "low"
        elif risk_score < 5.5:
            return "medium"
        elif risk_score < 8.0:
            return "high"
        else:
            return "very_high"

    def _recommend_position_size(self, risk_level: str) -> str:
        """
        Recommend maximum position size based on risk.

        Args:
            risk_level: Risk level category

        Returns:
            Position size recommendation
        """
        recommendations = {
            "low": "Up to 10% of portfolio",
            "medium": "Up to 5% of portfolio",
            "high": "Up to 2-3% of portfolio",
            "very_high": "Up to 1% of portfolio or avoid"
        }

        return recommendations.get(risk_level, "Up to 5% of portfolio")

    async def _analyze_with_llm(
        self,
        ticker: str,
        risk_factors: List[Dict[str, Any]],
        volatility: Dict[str, Any],
        risk_score: float
    ) -> Dict[str, Any]:
        """
        Use LLM for additional risk insights.

        Args:
            ticker: Stock ticker
            risk_factors: Identified risk factors
            volatility: Volatility metrics
            risk_score: Calculated risk score

        Returns:
            Additional analysis
        """
        risk_summary = {
            "ticker": ticker,
            "risk_score": risk_score,
            "risk_factors": risk_factors,
            "volatility": volatility
        }

        messages = [
            {"role": "system", "content": RISK_MANAGER_PROMPT},
            {"role": "user", "content": f"""
Perform risk assessment for {ticker}:

{self._format_data_for_prompt(risk_summary)}

Provide additional insights in JSON format:
{{
    "warnings": ["<warning 1>", "<warning 2>", ...],
    "reasoning": "<detailed risk assessment explanation>"
}}
"""}
        ]

        try:
            response = await self._call_llm_json(messages, temperature=0.7)
            return response
        except Exception as e:
            self.logger.error(f"LLM risk analysis failed: {e}")
            return {
                "warnings": ["See risk factors for detailed warnings"],
                "reasoning": f"Risk score: {risk_score:.1f}/10. {len(risk_factors)} risk factors identified."
            }
