"""Technical Analyst Agent - Analyzes price action and technical indicators."""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from src.agents.base_agent import BaseAgent
from src.models.prompts import TECHNICAL_ANALYST_PROMPT
from src.utils.validators import validate_confidence


class TechnicalAnalystAgent(BaseAgent):
    """
    Agent responsible for technical analysis of stocks.

    Calculates technical indicators, identifies trends,
    and provides trading signals based on price action.
    """

    def __init__(self, openai_client, db_client):
        """Initialize Technical Analyst Agent."""
        super().__init__("technical_analyst", openai_client, db_client)

    async def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute technical analysis.

        Args:
            inputs: Dictionary with 'ticker' and 'collected_data'

        Returns:
            Technical analysis results
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

        self.logger.info(f"Performing technical analysis for {ticker}")

        # Get price data
        price_data = collected_data.get("price_data", {})
        historical = price_data.get("historical_data", {})

        if not historical:
            return {
                "ticker": ticker,
                "error": "No historical price data available",
                "confidence": 0.0,
                "reasoning": "Missing historical data"
            }

        # Convert historical data to DataFrame
        df = self._prepare_dataframe(historical)

        if df is None or len(df) < 50:
            return {
                "ticker": ticker,
                "error": "Insufficient historical data (need at least 50 days)",
                "confidence": 0.0,
                "reasoning": "Not enough data points for technical analysis"
            }

        # Calculate indicators
        indicators = self._calculate_indicators(df)

        # Determine trend and signals
        trend = self._determine_trend(indicators, df)
        signals = self._generate_signals(indicators, trend)
        support_resistance = self._find_support_resistance(df)

        # Use LLM for interpretation
        analysis = await self._analyze_with_llm(ticker, indicators, trend, signals)

        return {
            "ticker": ticker,
            "trend": trend,
            "indicators": indicators,
            "signals": signals,
            "support_resistance": support_resistance,
            "confidence": validate_confidence(analysis.get("confidence", 0.7)),
            "reasoning": analysis.get("reasoning", "")
        }

    def _prepare_dataframe(self, historical: Dict[str, Any]) -> Optional[pd.DataFrame]:
        """
        Prepare DataFrame from historical data.

        Args:
            historical: Historical price data

        Returns:
            pandas DataFrame or None
        """
        try:
            if not historical:
                return None

            # Convert to DataFrame
            df = pd.DataFrame.from_dict(historical, orient='index')

            # Ensure we have required columns
            required_cols = ['Close', 'High', 'Low', 'Volume']
            if not all(col in df.columns for col in required_cols):
                self.logger.warning("Missing required columns in historical data")
                return None

            # Sort by date - use utc=True to handle mixed timezones
            df.index = pd.to_datetime(df.index, utc=True)
            df = df.sort_index()

            # Clean data
            df = df.dropna(subset=['Close'])

            return df

        except Exception as e:
            self.logger.error(f"Error preparing DataFrame: {e}")
            return None

    def _calculate_indicators(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate technical indicators.

        Args:
            df: Price data DataFrame

        Returns:
            Dictionary of indicators
        """
        indicators = {}

        try:
            # SMA - Simple Moving Averages
            if len(df) >= 50:
                indicators["SMA_50"] = float(df['Close'].rolling(window=50).mean().iloc[-1])

            if len(df) >= 200:
                indicators["SMA_200"] = float(df['Close'].rolling(window=200).mean().iloc[-1])

            # EMA - Exponential Moving Averages
            if len(df) >= 12:
                indicators["EMA_12"] = float(df['Close'].ewm(span=12, adjust=False).mean().iloc[-1])

            if len(df) >= 26:
                indicators["EMA_26"] = float(df['Close'].ewm(span=26, adjust=False).mean().iloc[-1])

            # RSI - Relative Strength Index
            rsi = self._calculate_rsi(df['Close'], period=14)
            if rsi is not None:
                indicators["RSI"] = float(rsi)

            # MACD
            if len(df) >= 26:
                macd_data = self._calculate_macd(df['Close'])
                indicators["MACD"] = macd_data

            # Volume analysis
            if 'Volume' in df.columns:
                avg_volume = df['Volume'].rolling(window=20).mean().iloc[-1]
                current_volume = df['Volume'].iloc[-1]
                indicators["avg_volume_20d"] = float(avg_volume)
                indicators["current_volume"] = float(current_volume)
                indicators["volume_ratio"] = float(current_volume / avg_volume) if avg_volume > 0 else 1.0

            # Golden/Death Cross detection
            if "SMA_50" in indicators and "SMA_200" in indicators:
                if indicators["SMA_50"] > indicators["SMA_200"]:
                    indicators["golden_death_cross"] = "Golden Cross (Bullish)"
                else:
                    indicators["golden_death_cross"] = "Death Cross (Bearish)"

        except Exception as e:
            self.logger.error(f"Error calculating indicators: {e}")
            indicators["error"] = str(e)

        return indicators

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> Optional[float]:
        """Calculate Relative Strength Index."""
        try:
            if len(prices) < period + 1:
                return None

            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))

            return float(rsi.iloc[-1])

        except Exception as e:
            self.logger.error(f"Error calculating RSI: {e}")
            return None

    def _calculate_macd(self, prices: pd.Series) -> Dict[str, Any]:
        """Calculate MACD indicator."""
        try:
            ema_12 = prices.ewm(span=12, adjust=False).mean()
            ema_26 = prices.ewm(span=26, adjust=False).mean()
            macd_line = ema_12 - ema_26
            signal_line = macd_line.ewm(span=9, adjust=False).mean()
            histogram = macd_line - signal_line

            current_macd = float(macd_line.iloc[-1])
            current_signal = float(signal_line.iloc[-1])
            current_histogram = float(histogram.iloc[-1])

            # Determine signal
            if current_histogram > 0:
                signal = "Bullish"
            elif current_histogram < 0:
                signal = "Bearish"
            else:
                signal = "Neutral"

            return {
                "value": current_macd,
                "signal_line": current_signal,
                "histogram": current_histogram,
                "signal": signal
            }

        except Exception as e:
            self.logger.error(f"Error calculating MACD: {e}")
            return {"error": str(e)}

    def _determine_trend(self, indicators: Dict[str, Any], df: pd.DataFrame) -> str:
        """
        Determine overall trend from indicators.

        Args:
            indicators: Calculated indicators
            df: Price DataFrame

        Returns:
            "bullish", "bearish", or "neutral"
        """
        bullish_signals = 0
        bearish_signals = 0

        # Check SMA trend
        if "SMA_50" in indicators and "SMA_200" in indicators:
            if indicators["SMA_50"] > indicators["SMA_200"]:
                bullish_signals += 1
            else:
                bearish_signals += 1

        # Check RSI
        if "RSI" in indicators:
            rsi = indicators["RSI"]
            if rsi > 70:
                bearish_signals += 1  # Overbought
            elif rsi < 30:
                bullish_signals += 1  # Oversold
            elif rsi > 50:
                bullish_signals += 0.5
            else:
                bearish_signals += 0.5

        # Check MACD
        macd = indicators.get("MACD", {})
        if isinstance(macd, dict) and "signal" in macd:
            if macd["signal"] == "Bullish":
                bullish_signals += 1
            elif macd["signal"] == "Bearish":
                bearish_signals += 1

        # Check price vs moving averages
        current_price = float(df['Close'].iloc[-1])
        if "SMA_50" in indicators:
            if current_price > indicators["SMA_50"]:
                bullish_signals += 0.5
            else:
                bearish_signals += 0.5

        # Determine overall trend
        if bullish_signals > bearish_signals + 1:
            return "bullish"
        elif bearish_signals > bullish_signals + 1:
            return "bearish"
        else:
            return "neutral"

    def _generate_signals(self, indicators: Dict[str, Any], trend: str) -> Dict[str, str]:
        """
        Generate trading signals for different timeframes.

        Args:
            indicators: Calculated indicators
            trend: Overall trend

        Returns:
            Dictionary of signals
        """
        signals = {
            "short_term": "hold",
            "medium_term": "hold",
            "long_term": "hold"
        }

        rsi = indicators.get("RSI", 50)
        macd = indicators.get("MACD", {})

        # Short-term signal (based on RSI and MACD)
        if rsi < 30 and macd.get("signal") == "Bullish":
            signals["short_term"] = "buy"
        elif rsi > 70 and macd.get("signal") == "Bearish":
            signals["short_term"] = "sell"
        elif macd.get("signal") == "Bullish":
            signals["short_term"] = "buy"
        elif macd.get("signal") == "Bearish":
            signals["short_term"] = "sell"

        # Medium-term signal (based on trend and indicators)
        if trend == "bullish" and rsi < 60:
            signals["medium_term"] = "buy"
        elif trend == "bearish" and rsi > 40:
            signals["medium_term"] = "sell"

        # Long-term signal (based on moving average crossovers)
        cross_signal = indicators.get("golden_death_cross", "")
        if "Golden Cross" in cross_signal:
            signals["long_term"] = "buy"
        elif "Death Cross" in cross_signal:
            signals["long_term"] = "sell"
        elif trend == "bullish":
            signals["long_term"] = "buy"
        elif trend == "bearish":
            signals["long_term"] = "sell"

        return signals

    def _find_support_resistance(self, df: pd.DataFrame) -> Dict[str, List[float]]:
        """
        Find support and resistance levels.

        Args:
            df: Price DataFrame

        Returns:
            Dictionary with support and resistance levels
        """
        try:
            # Use recent high and low points
            recent = df.tail(100)

            # Find local maxima (resistance)
            highs = recent['High'].values
            resistance_levels = []

            for i in range(2, len(highs) - 2):
                if highs[i] > highs[i-1] and highs[i] > highs[i-2] and \
                   highs[i] > highs[i+1] and highs[i] > highs[i+2]:
                    resistance_levels.append(float(highs[i]))

            # Find local minima (support)
            lows = recent['Low'].values
            support_levels = []

            for i in range(2, len(lows) - 2):
                if lows[i] < lows[i-1] and lows[i] < lows[i-2] and \
                   lows[i] < lows[i+1] and lows[i] < lows[i+2]:
                    support_levels.append(float(lows[i]))

            # Keep top 3 of each
            resistance_levels = sorted(set(resistance_levels), reverse=True)[:3]
            support_levels = sorted(set(support_levels), reverse=True)[:3]

            return {
                "support_levels": support_levels,
                "resistance_levels": resistance_levels
            }

        except Exception as e:
            self.logger.error(f"Error finding support/resistance: {e}")
            return {
                "support_levels": [],
                "resistance_levels": []
            }

    async def _analyze_with_llm(
        self,
        ticker: str,
        indicators: Dict[str, Any],
        trend: str,
        signals: Dict[str, str]
    ) -> Dict[str, Any]:
        """
        Use LLM to interpret technical analysis.

        Args:
            ticker: Stock ticker
            indicators: Calculated indicators
            trend: Overall trend
            signals: Trading signals

        Returns:
            Analysis dictionary
        """
        analysis_summary = {
            "ticker": ticker,
            "trend": trend,
            "indicators": indicators,
            "signals": signals
        }

        messages = [
            {"role": "system", "content": TECHNICAL_ANALYST_PROMPT},
            {"role": "user", "content": f"""
Analyze the following technical data for {ticker}:

{self._format_data_for_prompt(analysis_summary)}

Provide your interpretation in JSON format with:
{{
    "confidence": <float 0-1>,
    "reasoning": "<detailed technical analysis explanation>"
}}
"""}
        ]

        try:
            response = await self._call_llm_json(messages, temperature=0.7)
            return response
        except Exception as e:
            self.logger.error(f"LLM analysis failed: {e}")
            return {
                "confidence": 0.5,
                "reasoning": f"Technical analysis completed with calculated indicators. Trend: {trend}"
            }
