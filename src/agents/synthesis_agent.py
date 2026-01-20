"""Synthesis Agent - Creates comprehensive final reports."""

import uuid
from datetime import datetime
from typing import Dict, Any
from src.agents.base_agent import BaseAgent
from src.models.prompts import SYNTHESIS_AGENT_PROMPT
from src.utils.validators import validate_score
import src


class SynthesisAgent(BaseAgent):
    """
    Agent responsible for synthesizing all analyses into final report.

    Creates comprehensive markdown reports with all agent outputs,
    calculates overall scores, and adds mandatory disclaimers.
    """

    def __init__(self, openai_client, db_client):
        """Initialize Synthesis Agent."""
        super().__init__("synthesis_agent", openai_client, db_client)

    async def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute synthesis and report generation.

        Args:
            inputs: Dictionary with all agent outputs

        Returns:
            Final synthesis report
        """
        ticker = inputs.get("ticker", "").upper()
        collected_data = inputs.get("collected_data")
        fundamental = inputs.get("fundamental", {})
        technical = inputs.get("technical", {})
        sentiment = inputs.get("sentiment", {})
        debate = inputs.get("debate", {})
        risk = inputs.get("risk", {})

        self.logger.info(f"Synthesizing final report for {ticker}")

        # Handle missing collected_data
        if not collected_data or collected_data is None:
            self.logger.warning(f"No collected data available for {ticker}, generating limited report")
            return {
                "report_id": str(uuid.uuid4()),
                "ticker": ticker,
                "generated_at": datetime.now(),
                "overall_score": 0.0,
                "recommendation": "hold",
                "markdown_report": self._create_error_report(ticker, "Insufficient data collected"),
                "json_summary": {"error": "Insufficient data for analysis"},
                "saved_to_db": False,
                "confidence": 0.0,
                "reasoning": "Unable to generate comprehensive report due to data collection failure"
            }

        # Calculate overall score
        overall_score = self._calculate_overall_score(
            fundamental,
            technical,
            sentiment,
            risk
        )

        # Determine final recommendation
        final_recommendation = debate.get("final_recommendation", "hold")

        # Create markdown report
        markdown_report = self._create_markdown_report(
            ticker,
            collected_data,
            overall_score,
            final_recommendation,
            fundamental,
            technical,
            sentiment,
            debate,
            risk
        )

        # Create JSON summary
        json_summary = self._create_json_summary(
            ticker,
            overall_score,
            final_recommendation,
            fundamental,
            technical,
            sentiment,
            debate,
            risk
        )

        # Generate report ID
        report_id = str(uuid.uuid4())

        # Save to database
        try:
            self.db.save_report(
                report_id=report_id,
                run_id=inputs.get("run_id", report_id),
                ticker=ticker,
                overall_score=overall_score,
                recommendation=final_recommendation,
                markdown_report=markdown_report,
                json_data=json_summary
            )
            saved_to_db = True
        except Exception as e:
            self.logger.error(f"Failed to save report to database: {e}")
            saved_to_db = False

        return {
            "report_id": report_id,
            "ticker": ticker,
            "generated_at": datetime.now(),
            "overall_score": overall_score,
            "recommendation": final_recommendation,
            "markdown_report": markdown_report,
            "json_summary": json_summary,
            "saved_to_db": saved_to_db,
            "confidence": 0.9,
            "reasoning": "Comprehensive report generated from all agent analyses"
        }

    def _calculate_overall_score(
        self,
        fundamental: Dict[str, Any],
        technical: Dict[str, Any],
        sentiment: Dict[str, Any],
        risk: Dict[str, Any]
    ) -> float:
        """
        Calculate weighted overall score.

        Args:
            fundamental: Fundamental analysis
            technical: Technical analysis
            sentiment: Sentiment analysis
            risk: Risk analysis

        Returns:
            Overall score (0-10)
        """
        # Weights for each component
        weights = {
            "fundamental": 0.40,
            "technical": 0.25,
            "sentiment": 0.15,
            "risk": 0.20
        }

        # Get scores
        fund_score = fundamental.get("score", 5.0)

        # Convert technical trend to score
        tech_trend = technical.get("trend", "neutral")
        if tech_trend == "bullish":
            tech_score = 7.5
        elif tech_trend == "bearish":
            tech_score = 3.5
        else:
            tech_score = 5.0

        # Convert sentiment to score
        sent_score = sentiment.get("sentiment_score", 0.0)
        # Map from [-1, 1] to [0, 10]
        sentiment_score = (sent_score + 1) * 5

        # Risk score (invert since lower risk is better)
        risk_score_raw = risk.get("risk_score", 5.0)
        risk_score = 10 - risk_score_raw  # Invert

        # Calculate weighted average
        overall = (
            fund_score * weights["fundamental"] +
            tech_score * weights["technical"] +
            sentiment_score * weights["sentiment"] +
            risk_score * weights["risk"]
        )

        return validate_score(overall)

    def _create_markdown_report(
        self,
        ticker: str,
        data: Dict[str, Any],
        overall_score: float,
        recommendation: str,
        fundamental: Dict[str, Any],
        technical: Dict[str, Any],
        sentiment: Dict[str, Any],
        debate: Dict[str, Any],
        risk: Dict[str, Any]
    ) -> str:
        """Create comprehensive markdown report."""

        company_name = data.get("basic_info", {}).get("name", ticker)
        sector = data.get("basic_info", {}).get("sector", "Unknown")
        current_price = data.get("price_data", {}).get("current_price", 0)
        market_cap = data.get("basic_info", {}).get("market_cap", 0)

        # Format market cap
        if market_cap >= 1_000_000_000:
            market_cap_str = f"${market_cap / 1_000_000_000:.2f}B"
        elif market_cap >= 1_000_000:
            market_cap_str = f"${market_cap / 1_000_000:.2f}M"
        else:
            market_cap_str = f"${market_cap:,.0f}"

        report = f"""# Stock Analysis Report: {company_name} ({ticker})

**Analysis Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | **Purpose:** Research Only

---

## Executive Summary

| Metric | Value |
|--------|-------|
| **Overall Score** | **{overall_score:.1f}/10** |
| **Recommendation** | **{recommendation.upper().replace('_', ' ')}** |
| **Current Price** | ${current_price:.2f} |
| **Market Cap** | {market_cap_str} |
| **Sector** | {sector} |
| **Risk Level** | {risk.get('risk_level', 'Unknown').title()} |

### Quick Takeaways
- **Fundamental Score:** {fundamental.get('score', 'N/A'):.1f}/10 ({fundamental.get('recommendation', 'N/A')})
- **Technical Trend:** {technical.get('trend', 'Unknown').title()}
- **Sentiment:** {sentiment.get('overall_mood', 'Unknown').title()} (Score: {sentiment.get('sentiment_score', 0):.2f})
- **Risk Score:** {risk.get('risk_score', 'N/A'):.1f}/10

---

## Fundamental Analysis

**Score:** {fundamental.get('score', 0):.1f}/10 | **Recommendation:** {fundamental.get('recommendation', 'hold').replace('_', ' ').title()} | **Confidence:** {fundamental.get('confidence', 0):.0%}

### Financial Metrics

#### Profitability
{self._format_metrics_table(fundamental.get('metrics', {}).get('profitability', {}))}

#### Growth
{self._format_metrics_table(fundamental.get('metrics', {}).get('growth', {}))}

#### Financial Health
{self._format_metrics_table(fundamental.get('metrics', {}).get('health', {}))}

#### Valuation
{self._format_metrics_table(fundamental.get('metrics', {}).get('valuation', {}))}

### Strengths
{self._format_list(fundamental.get('strengths', []))}

### Weaknesses
{self._format_list(fundamental.get('weaknesses', []))}

---

## Technical Analysis

**Trend:** {technical.get('trend', 'Unknown').title()} | **Confidence:** {technical.get('confidence', 0):.0%}

### Key Indicators

{self._format_technical_indicators(technical.get('indicators', {}))}

### Trading Signals

| Timeframe | Signal |
|-----------|--------|
| **Short-term** | {technical.get('signals', {}).get('short_term', 'hold').upper()} |
| **Medium-term** | {technical.get('signals', {}).get('medium_term', 'hold').upper()} |
| **Long-term** | {technical.get('signals', {}).get('long_term', 'hold').upper()} |

### Support & Resistance Levels

**Support:** {', '.join([f'${x:.2f}' for x in technical.get('support_resistance', {}).get('support_levels', [])][:3]) or 'Not identified'}

**Resistance:** {', '.join([f'${x:.2f}' for x in technical.get('support_resistance', {}).get('resistance_levels', [])][:3]) or 'Not identified'}

---

## Sentiment Analysis

**Sentiment Score:** {sentiment.get('sentiment_score', 0):.2f} | **Mood:** {sentiment.get('overall_mood', 'Unknown').title()} | **Confidence:** {sentiment.get('confidence', 0):.0%}

### Recent News Summary

{self._format_news_summary(sentiment.get('news_summary', []))}

### Trending Themes
{self._format_list(sentiment.get('trending_themes', []))}

---

## Investment Debate

**Final Recommendation:** {debate.get('final_recommendation', 'hold').replace('_', ' ').upper()} | **Confidence:** {debate.get('confidence_level', 0):.0%}

### 🐂 Bull Case
**Strength:** {debate.get('bull_case', {}).get('strength', 0):.0%}

{debate.get('bull_case', {}).get('summary', 'No bull case available')}

**Key Points:**
{self._format_list(debate.get('bull_case', {}).get('key_points', []))}

### 🐻 Bear Case
**Strength:** {debate.get('bear_case', {}).get('strength', 0):.0%}

{debate.get('bear_case', {}).get('summary', 'No bear case available')}

**Key Points:**
{self._format_list(debate.get('bear_case', {}).get('key_points', []))}

### Conflicts & Resolutions

{self._format_conflicts(debate.get('conflicts', []))}

---

## Risk Assessment

**Risk Score:** {risk.get('risk_score', 0):.1f}/10 | **Risk Level:** {risk.get('risk_level', 'Unknown').upper()} | **Max Position Size:** {risk.get('max_position_size', 'Unknown')}

### Risk Factors

{self._format_risk_factors(risk.get('risk_factors', []))}

### Volatility Metrics

{self._format_metrics_table(risk.get('volatility_metrics', {}))}

### Warnings
{self._format_list(risk.get('warnings', []))}

---

## Final Recommendation

**Investment Recommendation:** {recommendation.replace('_', ' ').upper()}

**Overall Score:** {overall_score:.1f}/10

{debate.get('reasoning', 'See individual analyses above for detailed reasoning.')}

---

## Data Sources

- **Price Data:** Yahoo Finance
- **Financial Statements:** Yahoo Finance
- **News:** Yahoo Finance Aggregated News
- **Analysis Date:** {datetime.now().strftime('%Y-%m-%d')}

---

## ⚠️ IMPORTANT DISCLAIMER

**FOR RESEARCH AND EDUCATIONAL PURPOSES ONLY**

This report was generated by an AI-powered stock analysis system and is provided for research and educational purposes only. **THIS IS NOT INVESTMENT ADVICE.**

**Important Points:**
- This system is NOT a licensed financial advisor
- All information may be inaccurate, incomplete, or outdated
- Past performance does not guarantee future results
- Investing involves risk, including possible loss of principal
- Always consult with a licensed financial advisor before making investment decisions
- Do your own due diligence and research

**System Information:**
- Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- System Version: {src.__version__}
- Report ID: {uuid.uuid4()}

---

*End of Report*
"""

        return report

    def _format_metrics_table(self, metrics: Dict[str, Any]) -> str:
        """Format metrics dictionary as markdown table."""
        if not metrics or "error" in metrics:
            return "*No data available*\n"

        lines = ["| Metric | Value |", "|--------|-------|"]
        for key, value in metrics.items():
            if key != "error":
                formatted_key = key.replace('_', ' ').title()
                if isinstance(value, (int, float)):
                    if abs(value) < 10:
                        formatted_value = f"{value:.2f}"
                    else:
                        formatted_value = f"{value:,.2f}"
                else:
                    formatted_value = str(value)
                lines.append(f"| {formatted_key} | {formatted_value} |")

        return "\n".join(lines) + "\n"

    def _format_list(self, items: list) -> str:
        """Format list as markdown bullet points."""
        if not items:
            return "*None identified*\n"
        return "\n".join([f"- {item}" for item in items]) + "\n"

    def _format_technical_indicators(self, indicators: Dict[str, Any]) -> str:
        """Format technical indicators."""
        lines = ["| Indicator | Value |", "|-----------|-------|"]

        for key, value in indicators.items():
            if key not in ['error', 'avg_volume_20d', 'current_volume', 'volume_ratio']:
                formatted_key = key.replace('_', ' ').upper()
                if isinstance(value, dict):
                    formatted_value = value.get('signal', str(value))
                elif isinstance(value, float):
                    formatted_value = f"{value:.2f}"
                else:
                    formatted_value = str(value)
                lines.append(f"| {formatted_key} | {formatted_value} |")

        return "\n".join(lines) + "\n"

    def _format_news_summary(self, news: list) -> str:
        """Format news summary."""
        if not news:
            return "*No recent news available*\n"

        lines = []
        for item in news[:5]:  # Top 5 news items
            headline = item.get('headline', 'No headline')
            sentiment = item.get('sentiment', 'neutral')
            impact = item.get('impact', 'medium')

            emoji = "🟢" if sentiment == "positive" else "🔴" if sentiment == "negative" else "⚪"

            lines.append(f"{emoji} **{headline}**")
            lines.append(f"   *Sentiment: {sentiment.title()} | Impact: {impact.title()}*\n")

        return "\n".join(lines)

    def _format_conflicts(self, conflicts: list) -> str:
        """Format conflicts."""
        if not conflicts:
            return "*No major conflicts identified*\n"

        lines = []
        for conflict in conflicts:
            lines.append(f"**Issue:** {conflict.get('issue', 'Unknown')}")
            lines.append(f"**Involved:** {', '.join(conflict.get('agents_involved', []))}")
            lines.append(f"**Resolution:** {conflict.get('resolution', 'N/A')}\n")

        return "\n".join(lines)

    def _format_risk_factors(self, factors: list) -> str:
        """Format risk factors."""
        if not factors:
            return "*No significant risk factors identified*\n"

        lines = []
        for factor in factors:
            level = factor.get('level', 'medium')
            emoji = "🔴" if level == "high" else "🟡" if level == "medium" else "🟢"

            lines.append(f"{emoji} **{factor.get('type', 'Unknown')}** ({level.upper()})")
            lines.append(f"   {factor.get('description', '')}")
            lines.append(f"   *Mitigation: {factor.get('mitigation', 'N/A')}*\n")

        return "\n".join(lines)

    def _create_json_summary(
        self,
        ticker: str,
        overall_score: float,
        recommendation: str,
        fundamental: Dict[str, Any],
        technical: Dict[str, Any],
        sentiment: Dict[str, Any],
        debate: Dict[str, Any],
        risk: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create JSON summary of analysis."""
        return {
            "ticker": ticker,
            "timestamp": datetime.now().isoformat(),
            "overall_score": overall_score,
            "recommendation": recommendation,
            "components": {
                "fundamental": {
                    "score": fundamental.get("score"),
                    "recommendation": fundamental.get("recommendation"),
                    "confidence": fundamental.get("confidence")
                },
                "technical": {
                    "trend": technical.get("trend"),
                    "signals": technical.get("signals"),
                    "confidence": technical.get("confidence")
                },
                "sentiment": {
                    "score": sentiment.get("sentiment_score"),
                    "mood": sentiment.get("overall_mood"),
                    "confidence": sentiment.get("confidence")
                },
                "risk": {
                    "score": risk.get("risk_score"),
                    "level": risk.get("risk_level"),
                    "max_position": risk.get("max_position_size")
                }
            },
            "debate": {
                "final_recommendation": debate.get("final_recommendation"),
                "confidence": debate.get("confidence_level")
            }
        }

    def _create_error_report(self, ticker: str, error_message: str) -> str:
        """
        Create a minimal error report when data is insufficient.

        Args:
            ticker: Stock ticker
            error_message: Error description

        Returns:
            Markdown error report
        """
        return f"""# Stock Analysis Report: {ticker}

**Analysis Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | **Status:** ERROR

---

## ⚠️ Analysis Failed

**Error:** {error_message}

The stock analysis system was unable to collect sufficient data to perform a comprehensive analysis for **{ticker}**.

### Possible Causes:
- Invalid or unrecognized ticker symbol
- Data source (Yahoo Finance) temporarily unavailable
- Network connectivity issues
- Stock has been delisted or is not publicly traded

### Recommended Actions:
1. Verify the ticker symbol is correct
2. Check if the stock is actively traded on a major exchange
3. Try again in a few minutes if this is a temporary issue
4. Check your internet connection

---

## ⚠️ IMPORTANT DISCLAIMER

**FOR RESEARCH AND EDUCATIONAL PURPOSES ONLY**

This system is NOT a licensed financial advisor. Always consult with a licensed financial advisor before making investment decisions.

---

*Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
*System Version: {src.__version__}*
"""
