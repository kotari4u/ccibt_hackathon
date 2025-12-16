"""
API routes for backtesting predictions.
"""

from typing import Optional
from fastapi import APIRouter, HTTPException, Depends
from datetime import datetime, timedelta
import structlog

from src.api.schemas import BacktestRequest, BacktestResponse
from src.models.ensemble_predictor import EnsemblePredictor
from src.data.bigquery_client import BigQueryClient
# Authentication disabled
# from src.api.auth import get_current_user

logger = structlog.get_logger(__name__)
router = APIRouter(prefix="/backtest", tags=["backtest"])

ensemble_predictor = EnsemblePredictor()
bq_client: Optional[BigQueryClient] = None


def get_bq_client() -> BigQueryClient:
    """Get BigQuery client instance."""
    global bq_client
    if bq_client is None:
        bq_client = BigQueryClient()
    return bq_client


@router.post("/", response_model=BacktestResponse)
async def run_backtest(
    request: BacktestRequest,
    # Authentication disabled
    # current_user: dict = Depends(get_current_user),
) -> BacktestResponse:
    """
    Backtest prediction accuracy on historical data.
    
    Args:
        request: Backtest parameters
        current_user: Authenticated user
        
    Returns:
        Backtest results
    """
    try:
        logger.info(
            "Backtest request",
            symbol=request.symbol,
            prediction_date=request.prediction_date,
        )
        
        # Parse prediction date
        prediction_date = datetime.fromisoformat(request.prediction_date.replace('Z', '+00:00'))
        
        # Get market data
        bq = get_bq_client()
        end_date = datetime.now()
        start_date = end_date - timedelta(days=500)  # Get enough historical data
        
        market_data = bq.get_market_data(
            symbol=request.symbol,
            start_date=start_date,
            end_date=end_date,
        )
        
        if len(market_data) == 0:
            raise HTTPException(
                status_code=404,
                detail=f"No data found for symbol: {request.symbol}",
            )
        
        # Run backtest
        backtest_result = ensemble_predictor.backtest_prediction(
            market_data=market_data,
            symbol=request.symbol,
            prediction_date=prediction_date,
            horizon_days=request.horizon_days,
        )
        
        if "error" in backtest_result:
            raise HTTPException(
                status_code=400,
                detail=backtest_result["error"],
            )
        
        return BacktestResponse(**backtest_result)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Backtest failed", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Backtest failed: {str(e)}",
        )

