"""
API routes for pattern detection and anomaly detection.
"""

from fastapi import APIRouter, HTTPException, Depends, Query
from datetime import datetime, timedelta
import pandas as pd
import structlog

from src.api.schemas import PatternDetectionRequest, PatternDetectionResponse
from src.models.anomaly_detector import AnomalyDetector
from src.data.bigquery_client import BigQueryClient
from src.data.feature_engineering import FeatureEngineer
# Authentication disabled
# from src.api.auth import get_current_user

logger = structlog.get_logger(__name__)
router = APIRouter(prefix="/patterns", tags=["patterns"])

from typing import Optional

anomaly_detector = AnomalyDetector()
feature_engineer = FeatureEngineer()
bq_client: Optional[BigQueryClient] = None


def get_bq_client() -> BigQueryClient:
    """Get BigQuery client instance."""
    global bq_client
    if bq_client is None:
        bq_client = BigQueryClient()
    return bq_client


@router.get("/detect", response_model=PatternDetectionResponse)
async def detect_patterns(
    symbol: str,
    lookback_days: int = Query(default=60, ge=1, le=252),
    # Authentication disabled
    # current_user: dict = Depends(get_current_user),
) -> PatternDetectionResponse:
    """
    Detect patterns and anomalies in market data.
    
    Args:
        symbol: Trading symbol
        lookback_days: Number of days to look back
        current_user: Authenticated user
        
    Returns:
        Pattern detection results
    """
    try:
        logger.info(
            "Pattern detection request",
            symbol=symbol,
            lookback_days=lookback_days,
        )
        
        # Get market data
        bq = get_bq_client()
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days)
        
        market_data = bq.get_market_data(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
        )
        
        if len(market_data) == 0:
            raise HTTPException(
                status_code=404,
                detail=f"No data found for symbol: {symbol}",
            )
        
        # Create features
        market_data_features = feature_engineer.create_all_features(market_data)
        
        # Detect anomalies
        if 'returns' not in market_data_features.columns:
            market_data_features['returns'] = market_data_features['close'].pct_change()
        
        anomaly_score = anomaly_detector.calculate_anomaly_score(
            market_data_features,
            features=['rsi', 'macd', 'volatility_20', 'volume_ratio'],
        )
        
        # Price anomalies
        price_anomalies = anomaly_detector.detect_price_anomalies(
            market_data_features['close'],
            method="combined",
        )
        
        # Volume anomalies
        volume_anomalies = {}
        if 'volume' in market_data_features.columns:
            volume_anomalies = {
                "volume_spikes": anomaly_detector.detect_volume_anomalies(
                    market_data_features['volume'],
                    market_data_features['close'],
                ).sum(),
            }
        
        # Regime changes
        regime_changes = anomaly_detector.detect_regime_changes(
            market_data_features['returns'],
        )
        
        anomalies_dict = {
            "zscore": int(price_anomalies.get("zscore", pd.Series()).sum()),
            "iqr": int(price_anomalies.get("iqr", pd.Series()).sum()),
            "combined": int(price_anomalies.get("combined", pd.Series()).sum()),
            **volume_anomalies,
        }
        
        regime_changes_list = []
        if regime_changes.any():
            change_dates = market_data_features.loc[regime_changes, 'timestamp'].tolist() if 'timestamp' in market_data_features.columns else []
            regime_changes_list = [{"date": str(d)} for d in change_dates]
        
        return PatternDetectionResponse(
            symbol=symbol,
            anomalies=anomalies_dict,
            anomaly_score=float(anomaly_score.iloc[-1]) if len(anomaly_score) > 0 else 0.0,
            regime_changes=regime_changes_list,
            timestamp=datetime.now().isoformat(),
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Pattern detection failed", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Pattern detection failed: {str(e)}",
        )

