"""
API routes for volatility predictions.
"""

from typing import Optional
from fastapi import APIRouter, HTTPException, Depends, Query
from datetime import datetime, timedelta
import structlog

from src.api.schemas import (
    PredictionRequest,
    PredictionResponse,
    ErrorResponse,
)
from src.models.ensemble_predictor import EnsemblePredictor
from src.data.bigquery_client import BigQueryClient
# Authentication disabled
# from src.api.auth import get_current_user

logger = structlog.get_logger(__name__)
router = APIRouter(prefix="/predictions", tags=["predictions"])

# Initialize components (in production, use dependency injection)
ensemble_predictor = EnsemblePredictor()
bq_client: Optional[BigQueryClient] = None


def get_bq_client() -> BigQueryClient:
    """Get BigQuery client instance."""
    global bq_client
    if bq_client is None:
        bq_client = BigQueryClient()
    return bq_client


@router.get("/{symbol}", response_model=PredictionResponse)
async def get_prediction(
    symbol: str,
    horizon_days: int = Query(default=5, ge=1, le=30),
    include_events: bool = Query(default=True),
) -> PredictionResponse:
    """
    Get volatility prediction for a symbol.
    
    Args:
        symbol: Trading symbol
        horizon_days: Forecast horizon in days
        include_events: Whether to include event impact adjustments
        
    Returns:
        Prediction response with forecast and metadata
    """
    try:
        logger.info(
            "Prediction request received",
            symbol=symbol,
            horizon_days=horizon_days,
            user="anonymous",
        )
        
        # Get market data
        bq = get_bq_client()
        end_date = datetime.now()
        start_date = end_date - timedelta(days=1000)  # 1 year of data
        
        market_data = bq.get_market_data(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
        )
        
        if len(market_data) == 0:
            raise HTTPException(
                status_code=404,
                detail=f"No market data found for symbol: {symbol}",
            )
        
        # Get upcoming events if requested
        events = None
        if include_events:
            try:
                events = bq.get_upcoming_events(days_ahead=horizon_days)
                # Filter by symbol if symbol column exists, but keep events without symbol (economic indicators)
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
        
        return PredictionResponse(**prediction)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Prediction failed", error=str(e), symbol=symbol)
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}",
        )


@router.post("/", response_model=PredictionResponse)
async def create_prediction(
    request: PredictionRequest,
) -> PredictionResponse:
    """
    Create volatility prediction with custom parameters.
    
    Args:
        request: Prediction request parameters
        
    Returns:
        Prediction response
    """
    return await get_prediction(
        symbol=request.symbol,
        horizon_days=request.horizon_days,
        include_events=request.include_events,
    )

