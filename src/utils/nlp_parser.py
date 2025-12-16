"""
Natural Language Processing utilities for parsing user queries.
Extracts symbols, time horizons, and other parameters from natural language.
"""

import re
from typing import Dict, Optional, Tuple
import structlog

logger = structlog.get_logger(__name__)

# Common stock symbols and patterns
COMMON_SYMBOLS = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'JPM', 'V', 'JNJ',
    'WMT', 'PG', 'MA', 'UNH', 'HD', 'DIS', 'BAC', 'ADBE', 'PYPL', 'NFLX',
    'SPY', 'QQQ', 'DIA', 'IWM', 'DJI', 'GSPC', 'IXIC', 'VIX'
]

# Time horizon patterns
TIME_PATTERNS = [
    (r'(\d+)\s*(?:day|days|d)', 'days'),
    (r'(\d+)\s*(?:week|weeks|w)', 'weeks'),
    (r'(\d+)\s*(?:month|months|m)', 'months'),
    (r'(\d+)\s*(?:year|years|y)', 'years'),
    (r'next\s+(\d+)\s*(?:day|days)', 'days'),
    (r'(\d+)\s*(?:hour|hours|h)', 'hours'),
    (r'tomorrow', 'days'),
    (r'next\s+week', 'weeks'),
    (r'next\s+month', 'months'),
]


def extract_symbol(text: str) -> Optional[str]:
    """
    Extract stock symbol from natural language text.
    
    Args:
        text: User's natural language input
        
    Returns:
        Symbol if found, None otherwise
    """
    text_upper = text.upper()
    
    # Look for common symbols (3-5 letter uppercase patterns)
    # Check for exact matches first
    for symbol in COMMON_SYMBOLS:
        if symbol in text_upper:
            # Make sure it's not part of a word
            pattern = r'\b' + re.escape(symbol) + r'\b'
            if re.search(pattern, text_upper):
                logger.debug("Symbol extracted", symbol=symbol, text=text)
                return symbol
    
    # Look for patterns like "AAPL", "for AAPL", "AAPL stock", etc.
    symbol_pattern = r'\b([A-Z]{1,5})\b'
    matches = re.findall(symbol_pattern, text_upper)
    
    for match in matches:
        # Filter out common words
        if match not in ['THE', 'FOR', 'AND', 'BUT', 'ARE', 'WAS', 'HAS', 'HAD', 'WILL', 'CAN', 'MAY']:
            # Check if it looks like a stock symbol (2-5 uppercase letters)
            if 2 <= len(match) <= 5 and match.isalpha():
                logger.debug("Symbol pattern extracted", symbol=match, text=text)
                return match
    
    # Look for patterns like "Apple", "Microsoft", etc. and map to symbols
    company_to_symbol = {
        'APPLE': 'AAPL',
        'MICROSOFT': 'MSFT',
        'GOOGLE': 'GOOGL',
        'AMAZON': 'AMZN',
        'META': 'META',
        'FACEBOOK': 'META',
        'TESLA': 'TSLA',
        'NVIDIA': 'NVDA',
        'JPMORGAN': 'JPM',
        'VISA': 'V',
        'JOHNSON': 'JNJ',
        'WALMART': 'WMT',
        'PROCTER': 'PG',
        'MASTERCARD': 'MA',
        'UNITEDHEALTH': 'UNH',
        'HOMEDEPOT': 'HD',
        'DISNEY': 'DIS',
        'BANKOFAMERICA': 'BAC',
        'ADOBE': 'ADBE',
        'PAYPAL': 'PYPL',
        'NETFLIX': 'NFLX',
        'SP500': 'SPY',
        'S&P500': 'SPY',
        'DOW': 'DJI',
        'DOWJONES': 'DJI',
        'NASDAQ': 'IXIC',
    }
    
    for company, symbol in company_to_symbol.items():
        if company in text_upper.replace(' ', ''):
            logger.debug("Company name mapped to symbol", company=company, symbol=symbol, text=text)
            return symbol
    
    logger.debug("No symbol found in text", text=text)
    return None


def extract_horizon_days(text: str) -> int:
    """
    Extract forecast horizon in days from natural language.
    
    Args:
        text: User's natural language input
        
    Returns:
        Number of days (default: 5)
    """
    text_lower = text.lower()
    
    # Check for specific time patterns
    for pattern, unit in TIME_PATTERNS:
        match = re.search(pattern, text_lower)
        if match:
            if unit == 'days':
                if 'tomorrow' in text_lower:
                    return 1
                number = int(match.group(1)) if match.groups() else 1
                return number
            elif unit == 'weeks':
                number = int(match.group(1)) if match.groups() else 1
                return number * 7
            elif unit == 'months':
                number = int(match.group(1)) if match.groups() else 1
                return number * 30
            elif unit == 'years':
                number = int(match.group(1)) if match.groups() else 1
                return number * 365
            elif unit == 'hours':
                number = int(match.group(1)) if match.groups() else 1
                return max(1, number // 24)  # Convert to days, minimum 1
    
    # Check for keywords
    if 'short term' in text_lower or 'short-term' in text_lower:
        return 3
    elif 'medium term' in text_lower or 'medium-term' in text_lower:
        return 7
    elif 'long term' in text_lower or 'long-term' in text_lower:
        return 30
    elif 'week' in text_lower and 'next' in text_lower:
        return 7
    elif 'month' in text_lower and 'next' in text_lower:
        return 30
    
    # Default
    return 5


def extract_include_events(text: str) -> bool:
    """
    Determine if user wants to include events in prediction.
    
    Args:
        text: User's natural language input
        
    Returns:
        True if events should be included (default: True)
    """
    text_lower = text.lower()
    
    # Negative patterns
    negative_patterns = [
        'no events', 'without events', 'exclude events',
        'ignore events', 'no event', 'skip events'
    ]
    
    for pattern in negative_patterns:
        if pattern in text_lower:
            return False
    
    # Positive patterns
    positive_patterns = [
        'with events', 'include events', 'consider events',
        'event impact', 'upcoming events'
    ]
    
    for pattern in positive_patterns:
        if pattern in text_lower:
            return True
    
    # Default: include events
    return True


def parse_query(text: str) -> Dict[str, any]:
    """
    Parse natural language query and extract parameters.
    
    Args:
        text: User's natural language input
        
    Returns:
        Dictionary with extracted parameters:
        - symbol: str (required)
        - horizon_days: int (default: 5)
        - include_events: bool (default: True)
        - original_text: str
    """
    symbol = extract_symbol(text)
    horizon_days = extract_horizon_days(text)
    include_events = extract_include_events(text)
    
    result = {
        'symbol': symbol,
        'horizon_days': horizon_days,
        'include_events': include_events,
        'original_text': text,
    }
    
    logger.info("Query parsed", **result)
    return result


def format_prediction_response(prediction: Dict, query_text: str) -> str:
    """
    Format prediction response as natural language.
    
    Args:
        prediction: Prediction response dictionary
        query_text: Original user query
        
    Returns:
        Formatted natural language response
    """
    symbol = prediction.get('symbol', 'Unknown')
    horizon_days = prediction.get('horizon_days', 5)
    avg_confidence = prediction.get('avg_confidence', 0.0)
    forecast = prediction.get('forecast', [])
    rationale = prediction.get('rationale', 'No rationale available.')
    
    # Calculate average volatility
    avg_volatility = sum(forecast) / len(forecast) if forecast else 0.0
    max_volatility = max(forecast) if forecast else 0.0
    min_volatility = min(forecast) if forecast else 0.0
    
    response = f"""
📊 **Prediction for {symbol}** (Next {horizon_days} days)

**Volatility Forecast:**
• Average: {avg_volatility:.4f}
• Peak: {max_volatility:.4f}
• Minimum: {min_volatility:.4f}

**Confidence:** {avg_confidence:.1%}

**Analysis:**
{rationale}

**Daily Forecast:**
"""
    
    for i, vol in enumerate(forecast[:10], 1):  # Show first 10 days
        response += f"Day {i}: {vol:.4f}\n"
    
    if len(forecast) > 10:
        response += f"... and {len(forecast) - 10} more days\n"
    
    # Add event adjustments if available
    event_adjustments = prediction.get('event_adjustments', [])
    if event_adjustments:
        response += f"\n**Event Impact:** {len(event_adjustments)} events considered\n"
    
    return response.strip()

