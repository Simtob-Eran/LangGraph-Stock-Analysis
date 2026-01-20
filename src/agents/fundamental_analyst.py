"""Fundamental Analyst Agent - Analyzes financial health and intrinsic value."""

import json
from typing import Dict, Any, List
from src.agents.base_agent import BaseAgent
from src.models.prompts import FUNDAMENTAL_ANALYST_PROMPT
from src.utils.validators import validate_score, validate_confidence


class FundamentalAnalystAgent(BaseAgent):
    """
    Agent responsible for fundamental analysis of stocks.

    Analyzes financial statements, calculates key metrics,
    and provides investment recommendations based on fundamentals.
    """

    def __init__(self, openai_client, db_client):
        """Initialize Fundamental Analyst Agent."""
        super().__init__("fundamental_analyst", openai_client, db_client)

    async def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute fundamental analysis.

        Args:
            inputs: Dictionary with 'ticker' and 'collected_data'

        Returns:
            Fundamental analysis results
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

        self.logger.info(f"Performing fundamental analysis for {ticker}")

        # Calculate metrics
        metrics = self._calculate_metrics(collected_data)

        # Use LLM to analyze and provide insights
        analysis = await self._analyze_with_llm(ticker, collected_data, metrics)

        # Structure the response
        return {
            "ticker": ticker,
            "score": validate_score(analysis.get("score", 5.0)),
            "metrics": metrics,
            "strengths": analysis.get("strengths", []),
            "weaknesses": analysis.get("weaknesses", []),
            "recommendation": analysis.get("recommendation", "hold"),
            "confidence": validate_confidence(analysis.get("confidence", 0.7)),
            "reasoning": analysis.get("reasoning", "")
        }

    def _calculate_metrics(self, data: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
        """
        Calculate key financial metrics.

        Args:
            data: Collected stock data

        Returns:
            Dictionary of calculated metrics
        """
        financials = data.get("financials", {})
        income_stmt = financials.get("income_statement", {})
        balance_sheet = financials.get("balance_sheet", {})
        cash_flow = financials.get("cash_flow", {})
        basic_info = data.get("basic_info", {})

        metrics = {
            "profitability": self._calc_profitability(income_stmt, balance_sheet),
            "growth": self._calc_growth(income_stmt),
            "efficiency": self._calc_efficiency(income_stmt, balance_sheet),
            "health": self._calc_health(balance_sheet, cash_flow),
            "valuation": self._calc_valuation(income_stmt, basic_info, data.get("price_data", {}))
        }

        return metrics

    def _calc_profitability(self, income_stmt: Dict, balance_sheet: Dict) -> Dict[str, float]:
        """Calculate profitability metrics."""
        metrics = {}

        try:
            # Get most recent period data
            if not income_stmt:
                return {"error": "No income statement data"}

            # Extract values (yfinance returns dict with date keys)
            # We'll take the first (most recent) available
            periods = list(income_stmt.keys())
            if not periods:
                return {"error": "No periods available"}

            latest = periods[0] if isinstance(periods[0], str) else str(periods[0])
            data = income_stmt.get(latest, {}) if isinstance(income_stmt, dict) else {}

            # Net Profit Margin
            revenue = self._get_value(data, ["Total Revenue", "TotalRevenue", "Revenue"])
            net_income = self._get_value(data, ["Net Income", "NetIncome"])

            if revenue and revenue != 0:
                metrics["net_profit_margin"] = (net_income / revenue) * 100

            # Operating Margin
            operating_income = self._get_value(data, ["Operating Income", "OperatingIncome"])
            if revenue and revenue != 0 and operating_income:
                metrics["operating_margin"] = (operating_income / revenue) * 100

            # ROE and ROA require balance sheet
            if balance_sheet:
                bs_periods = list(balance_sheet.keys())
                if bs_periods:
                    bs_latest = bs_periods[0] if isinstance(bs_periods[0], str) else str(bs_periods[0])
                    bs_data = balance_sheet.get(bs_latest, {})

                    total_assets = self._get_value(bs_data, ["Total Assets", "TotalAssets"])
                    shareholders_equity = self._get_value(bs_data, ["Stockholders Equity", "StockholdersEquity", "Total Equity"])

                    if total_assets and total_assets != 0:
                        metrics["roa"] = (net_income / total_assets) * 100

                    if shareholders_equity and shareholders_equity != 0:
                        metrics["roe"] = (net_income / shareholders_equity) * 100

        except Exception as e:
            self.logger.warning(f"Error calculating profitability: {e}")
            metrics["error"] = str(e)

        return metrics

    def _calc_growth(self, income_stmt: Dict) -> Dict[str, float]:
        """Calculate growth metrics."""
        metrics = {}

        try:
            if not income_stmt or len(income_stmt) < 2:
                return {"error": "Insufficient data for growth calculation"}

            periods = sorted(list(income_stmt.keys()), reverse=True)[:2]

            if len(periods) < 2:
                return {"error": "Need at least 2 periods"}

            current = income_stmt.get(periods[0], {})
            previous = income_stmt.get(periods[1], {})

            # Revenue growth
            current_revenue = self._get_value(current, ["Total Revenue", "Revenue"])
            previous_revenue = self._get_value(previous, ["Total Revenue", "Revenue"])

            if current_revenue and previous_revenue and previous_revenue != 0:
                metrics["revenue_growth_yoy"] = ((current_revenue - previous_revenue) / previous_revenue) * 100

            # Earnings growth
            current_earnings = self._get_value(current, ["Net Income", "NetIncome"])
            previous_earnings = self._get_value(previous, ["Net Income", "NetIncome"])

            if current_earnings and previous_earnings and previous_earnings != 0:
                metrics["earnings_growth_yoy"] = ((current_earnings - previous_earnings) / previous_earnings) * 100

        except Exception as e:
            self.logger.warning(f"Error calculating growth: {e}")
            metrics["error"] = str(e)

        return metrics

    def _calc_efficiency(self, income_stmt: Dict, balance_sheet: Dict) -> Dict[str, float]:
        """Calculate efficiency metrics."""
        metrics = {}

        try:
            if not income_stmt or not balance_sheet:
                return {"error": "Insufficient data"}

            # Get latest periods
            is_periods = list(income_stmt.keys())
            bs_periods = list(balance_sheet.keys())

            if not is_periods or not bs_periods:
                return {"error": "No periods available"}

            is_data = income_stmt.get(is_periods[0], {})
            bs_data = balance_sheet.get(bs_periods[0], {})

            # Asset Turnover
            revenue = self._get_value(is_data, ["Total Revenue", "Revenue"])
            total_assets = self._get_value(bs_data, ["Total Assets", "TotalAssets"])

            if revenue and total_assets and total_assets != 0:
                metrics["asset_turnover"] = revenue / total_assets

        except Exception as e:
            self.logger.warning(f"Error calculating efficiency: {e}")
            metrics["error"] = str(e)

        return metrics

    def _calc_health(self, balance_sheet: Dict, cash_flow: Dict) -> Dict[str, float]:
        """Calculate financial health metrics."""
        metrics = {}

        try:
            if not balance_sheet:
                return {"error": "No balance sheet data"}

            bs_periods = list(balance_sheet.keys())
            if not bs_periods:
                return {"error": "No periods available"}

            bs_data = balance_sheet.get(bs_periods[0], {})

            # Current Ratio
            current_assets = self._get_value(bs_data, ["Current Assets", "CurrentAssets"])
            current_liabilities = self._get_value(bs_data, ["Current Liabilities", "CurrentLiabilities"])

            if current_assets and current_liabilities and current_liabilities != 0:
                metrics["current_ratio"] = current_assets / current_liabilities

            # Debt to Equity
            total_debt = self._get_value(bs_data, ["Total Debt", "TotalDebt", "Long Term Debt"])
            shareholders_equity = self._get_value(bs_data, ["Stockholders Equity", "StockholdersEquity"])

            if total_debt and shareholders_equity and shareholders_equity != 0:
                metrics["debt_to_equity"] = total_debt / shareholders_equity

            # Free Cash Flow
            if cash_flow:
                cf_periods = list(cash_flow.keys())
                if cf_periods:
                    cf_data = cash_flow.get(cf_periods[0], {})
                    operating_cf = self._get_value(cf_data, ["Operating Cash Flow", "OperatingCashFlow"])
                    capex = self._get_value(cf_data, ["Capital Expenditure", "CapitalExpenditure", "CapEx"])

                    if operating_cf:
                        metrics["free_cash_flow"] = operating_cf - (capex if capex else 0)

        except Exception as e:
            self.logger.warning(f"Error calculating health: {e}")
            metrics["error"] = str(e)

        return metrics

    def _calc_valuation(self, income_stmt: Dict, basic_info: Dict, price_data: Dict) -> Dict[str, float]:
        """Calculate valuation metrics."""
        metrics = {}

        try:
            market_cap = basic_info.get("market_cap", 0)
            current_price = price_data.get("current_price", 0)

            if not market_cap or not current_price:
                return {"error": "Insufficient data for valuation"}

            # P/E Ratio
            if income_stmt:
                is_periods = list(income_stmt.keys())
                if is_periods:
                    is_data = income_stmt.get(is_periods[0], {})
                    net_income = self._get_value(is_data, ["Net Income", "NetIncome"])

                    if net_income and net_income != 0:
                        metrics["pe_ratio"] = market_cap / net_income

        except Exception as e:
            self.logger.warning(f"Error calculating valuation: {e}")
            metrics["error"] = str(e)

        return metrics

    def _get_value(self, data: Dict, keys: List[str]) -> float:
        """Get value from dict by trying multiple possible keys."""
        for key in keys:
            if key in data:
                value = data[key]
                if isinstance(value, (int, float)):
                    return float(value)
        return 0.0

    async def _analyze_with_llm(
        self,
        ticker: str,
        data: Dict[str, Any],
        metrics: Dict[str, Dict[str, float]]
    ) -> Dict[str, Any]:
        """
        Use LLM to analyze metrics and provide insights.

        Args:
            ticker: Stock ticker
            data: Collected data
            metrics: Calculated metrics

        Returns:
            Analysis dictionary
        """
        # Prepare data summary for LLM
        data_summary = {
            "ticker": ticker,
            "company": data.get("basic_info", {}).get("name", ticker),
            "sector": data.get("basic_info", {}).get("sector", "Unknown"),
            "market_cap": data.get("basic_info", {}).get("market_cap", 0),
            "metrics": metrics
        }

        messages = [
            {"role": "system", "content": FUNDAMENTAL_ANALYST_PROMPT},
            {"role": "user", "content": f"""
Analyze the following fundamental data for {ticker}:

{self._format_data_for_prompt(data_summary)}

Provide your analysis in JSON format with the following structure:
{{
    "score": <float 0-10>,
    "strengths": [<list of 3-5 strengths>],
    "weaknesses": [<list of 3-5 weaknesses>],
    "recommendation": "<strong_buy|buy|hold|sell|strong_sell>",
    "confidence": <float 0-1>,
    "reasoning": "<detailed explanation>"
}}
"""}
        ]

        try:
            response = await self._call_llm_json(messages, temperature=0.7)
            return response
        except Exception as e:
            self.logger.error(f"LLM analysis failed: {e}")
            return {
                "score": 5.0,
                "strengths": ["Unable to analyze"],
                "weaknesses": ["Analysis failed"],
                "recommendation": "hold",
                "confidence": 0.3,
                "reasoning": f"Analysis failed: {str(e)}"
            }
