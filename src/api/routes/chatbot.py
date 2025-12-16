"""
Chatbot API routes for natural language market predictions.
"""

from typing import Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from datetime import datetime, timedelta
import structlog

from src.api.schemas import (
    PredictionResponse,
    ErrorResponse,
)
from src.models.ensemble_predictor import EnsemblePredictor
from src.data.bigquery_client import BigQueryClient
from src.utils.nlp_parser import parse_query, format_prediction_response


class ChatRequest(BaseModel):
    """Request schema for chatbot."""
    message: str = Field(..., description="Natural language query from user")

logger = structlog.get_logger(__name__)
router = APIRouter(prefix="/chatbot", tags=["chatbot"])

# Initialize components
ensemble_predictor = EnsemblePredictor()
bq_client: Optional[BigQueryClient] = None


def get_bq_client() -> BigQueryClient:
    """Get BigQuery client instance."""
    global bq_client
    if bq_client is None:
        bq_client = BigQueryClient()
    return bq_client


@router.post("/chat")
async def chat(
    request: ChatRequest,
) -> dict:
    """
    Chat endpoint that accepts natural language queries and returns predictions.
    
    Examples:
    - "What's the volatility forecast for AAPL?"
    - "Predict SPY for the next 7 days"
    - "Show me Apple stock prediction for next week"
    - "What will happen to Tesla in the next month?"
    
    Args:
        message: Natural language query from user
        
    Returns:
        Dictionary with:
        - response: Natural language formatted prediction
        - prediction: Full prediction data
        - parsed_query: Extracted parameters
    """
    try:
        message = request.message
        logger.info("Chat request received", message=message)
        
        # Parse the natural language query
        parsed = parse_query(message)
        
        if not parsed['symbol']:
            return {
                "response": "I couldn't find a stock symbol in your message. Please specify a symbol like AAPL, SPY, or Tesla.",
                "error": "No symbol found",
                "suggestions": [
                    "Try: 'What's the prediction for AAPL?'",
                    "Try: 'Show me SPY volatility forecast'",
                    "Try: 'Predict Tesla for next week'"
                ]
            }
        
        symbol = parsed['symbol']
        horizon_days = parsed['horizon_days']
        include_events = parsed['include_events']
        
        # Get market data
        bq = get_bq_client()
        end_date = datetime.now()
        start_date = end_date - timedelta(days=1000)
        
        market_data = bq.get_market_data(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
        )
        
        if len(market_data) == 0:
            return {
                "response": f"Sorry, I couldn't find any market data for {symbol}. Please check if the symbol is correct.",
                "error": f"No market data found for symbol: {symbol}",
                "symbol": symbol
            }
        
        # Get upcoming events if requested
        events = None
        if include_events:
            try:
                events = bq.get_upcoming_events(days_ahead=horizon_days)
                if events is not None and len(events) > 0 and 'symbol' in events.columns:
                    symbol_mask = (events['symbol'] == symbol) | (events['symbol'].isna())
                    events = events[symbol_mask] if symbol_mask.any() else events
            except Exception as e:
                logger.warning("Failed to fetch events", error=str(e))
        
        # Make prediction
        prediction = ensemble_predictor.predict_volatility(
            market_data=market_data,
            symbol=symbol,
            horizon_days=horizon_days,
            events=events,
        )
        
        # Format response
        prediction_response = PredictionResponse(**prediction)
        formatted_response = format_prediction_response(
            prediction_response.dict(),
            message
        )
        
        return {
            "response": formatted_response,
            "prediction": prediction_response.dict(),
            "parsed_query": parsed,
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error("Chat request failed", error=str(e), message=message)
        return {
            "response": f"I encountered an error processing your request: {str(e)}. Please try rephrasing your question.",
            "error": str(e),
            "message": message
        }


@router.get("/help")
async def get_help() -> dict:
    """
    Get help information about how to use the chatbot.
    
    Returns:
        Dictionary with help information and examples
    """
    return {
        "title": "Market Prediction Chatbot Help",
        "description": "Ask me about stock volatility predictions using natural language!",
        "examples": [
            "What's the volatility forecast for AAPL?",
            "Predict SPY for the next 7 days",
            "Show me Apple stock prediction for next week",
            "What will happen to Tesla in the next month?",
            "Forecast Microsoft volatility for 10 days",
            "Tell me about DJI prediction for next 2 weeks",
        ],
        "supported_symbols": [
            "Individual stocks: AAPL, MSFT, GOOGL, AMZN, TSLA, etc.",
            "Indices: SPY, DJI, GSPC, IXIC, VIX",
            "Company names: Apple, Microsoft, Google, Amazon, Tesla, etc."
        ],
        "time_horizons": [
            "Days: 'next 5 days', '7 days', '10 days'",
            "Weeks: 'next week', '2 weeks'",
            "Months: 'next month', '3 months'",
            "Keywords: 'short term' (3 days), 'medium term' (7 days), 'long term' (30 days)"
        ],
        "features": [
            "Natural language processing",
            "Automatic symbol detection",
            "Time horizon extraction",
            "Event impact analysis",
            "Confidence scoring"
        ]
    }

