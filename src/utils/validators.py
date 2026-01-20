"""Input validation utilities."""

import re
from typing import List, Tuple


def validate_ticker(ticker: str) -> Tuple[bool, str]:
    """
    Validate a stock ticker symbol.

    Args:
        ticker: Stock ticker to validate

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not ticker:
        return False, "Ticker cannot be empty"

    # Clean the ticker
    ticker = ticker.strip().upper()

    # Basic validation: 1-5 alphanumeric characters, possibly with dots or hyphens
    if not re.match(r'^[A-Z0-9\.\-]{1,10}$', ticker):
        return False, f"Invalid ticker format: {ticker}"

    return True, ""


def validate_tickers(tickers: List[str]) -> Tuple[bool, str, List[str]]:
    """
    Validate a list of ticker symbols.

    Args:
        tickers: List of tickers to validate

    Returns:
        Tuple of (all_valid, error_message, valid_tickers)
    """
    if not tickers:
        return False, "No tickers provided", []

    valid_tickers = []
    invalid_tickers = []

    for ticker in tickers:
        is_valid, error = validate_ticker(ticker)
        if is_valid:
            valid_tickers.append(ticker.strip().upper())
        else:
            invalid_tickers.append(ticker)

    if invalid_tickers:
        return False, f"Invalid tickers: {', '.join(invalid_tickers)}", valid_tickers

    if len(valid_tickers) > 10:
        return False, "Maximum 10 tickers allowed per analysis", valid_tickers[:10]

    return True, "", valid_tickers


def parse_query(query: str) -> Tuple[List[str], str]:
    """
    Parse user query to extract tickers and analysis type.

    Args:
        query: User query string

    Returns:
        Tuple of (tickers, analysis_type)

    Examples:
        "Analyze AAPL" -> (["AAPL"], "single")
        "Compare AAPL, MSFT, GOOGL" -> (["AAPL", "MSFT", "GOOGL"], "multiple")
        "Top tech stocks" -> ([], "sector")
    """
    query = query.strip()

    # Look for explicit ticker symbols (1-5 uppercase letters)
    ticker_pattern = r'\b[A-Z]{1,5}\b'
    potential_tickers = re.findall(ticker_pattern, query)

    # Filter out common words that might look like tickers
    common_words = {
        'A', 'I', 'US', 'UK', 'IT', 'AI', 'API', 'CEO', 'CFO',
        'IPO', 'ETF', 'USD', 'GDP', 'PE', 'ROE', 'ROA', 'EPS'
    }
    tickers = [t for t in potential_tickers if t not in common_words]

    # Determine analysis type
    if len(tickers) == 0:
        return [], "sector"
    elif len(tickers) == 1:
        return tickers, "single"
    else:
        return tickers, "multiple"


def sanitize_input(text: str) -> str:
    """
    Sanitize user input to prevent injection attacks.

    Args:
        text: User input text

    Returns:
        Sanitized text
    """
    if not text:
        return ""

    # Remove potentially dangerous characters
    text = re.sub(r'[;<>|&$`]', '', text)

    # Limit length
    max_length = 500
    if len(text) > max_length:
        text = text[:max_length]

    return text.strip()


def validate_score(score: float, min_val: float = 0, max_val: float = 10) -> float:
    """
    Validate and clamp a score within valid range.

    Args:
        score: Score to validate
        min_val: Minimum valid value
        max_val: Maximum valid value

    Returns:
        Valid score clamped to range
    """
    if score < min_val:
        return min_val
    if score > max_val:
        return max_val
    return score


def validate_confidence(confidence: float) -> float:
    """
    Validate and clamp a confidence value to [0, 1].

    Args:
        confidence: Confidence value to validate

    Returns:
        Valid confidence value
    """
    return max(0.0, min(1.0, confidence))
