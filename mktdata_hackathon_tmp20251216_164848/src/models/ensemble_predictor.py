"""
Ensemble prediction system combining multiple models.
Provides unified interface for volatility forecasting with confidence scoring.
"""

from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import structlog

from src.models.volatility_forecaster import VolatilityForecaster
from src.models.event_impact_model import EventImpactModel
from src.models.anomaly_detector import AnomalyDetector
from src.data.feature_engineering import FeatureEngineer

logger = structlog.get_logger(__name__)


class EnsemblePredictor:
    """
    Ensemble prediction system combining multiple forecasting approaches.
    
    Integrates volatility forecasts, event impacts, and anomaly detection
    to provide comprehensive market predictions.
    """
    
    def __init__(self):
        """Initialize ensemble predictor."""
        self.volatility_forecaster = VolatilityForecaster()
        self.event_impact_model = EventImpactModel()
        self.anomaly_detector = AnomalyDetector()
        self.feature_engineer = FeatureEngineer()
        self.logger = logger
    
    def predict_volatility(
        self,
        market_data: pd.DataFrame,
        symbol: str,
        horizon_days: int = 5,
        events: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        """
        Predict volatility using ensemble approach.
        
        Args:
            market_data: Historical market data
            symbol: Trading symbol
            horizon_days: Forecast horizon in days
            events: Optional upcoming events DataFrame
            
        Returns:
            Dictionary with prediction and metadata
        """
        # Filter data for symbol
        symbol_data = market_data[market_data['symbol'] == symbol].copy()
        
        if len(symbol_data) == 0:
            raise ValueError(f"No data found for symbol: {symbol}")
        
        # Ensure sorted by timestamp
        if 'timestamp' in symbol_data.columns:
            symbol_data['timestamp'] = pd.to_datetime(symbol_data['timestamp'])
            symbol_data = symbol_data.sort_values('timestamp').reset_index(drop=True)
        elif 'date' in symbol_data.columns:
            symbol_data['timestamp'] = pd.to_datetime(symbol_data['date'])
            symbol_data = symbol_data.sort_values('timestamp').reset_index(drop=True)
        
        # Calculate returns
        if 'returns' not in symbol_data.columns:
            symbol_data['returns'] = symbol_data['close'].pct_change()
        
        returns = symbol_data['returns'].dropna()
        
        if len(returns) < 30:
            raise ValueError("Insufficient data for prediction")
        
        # Create features
        symbol_data_features = self.feature_engineer.create_all_features(symbol_data)
        
        # Get base volatility forecast
        try:
            forecast_result = self.volatility_forecaster.ensemble_forecast(
                returns=returns,
                features=symbol_data_features.select_dtypes(include=[np.number]),
                horizon=horizon_days,
            )
            base_forecast = forecast_result["forecast"]
            base_confidence = forecast_result["confidence"]
        except Exception as e:
            logger.warning("Ensemble forecast failed, using simple method", error=str(e))
            # Fallback to simple realized volatility
            current_vol = returns.rolling(20).std().iloc[-1] * np.sqrt(252)
            base_forecast = [current_vol] * horizon_days
            base_confidence = [0.5] * horizon_days
        
        # Adjust for upcoming events
        event_adjustments = []
        if events is not None and len(events) > 0:
            # Ensure event_date is datetime
            if 'event_date' in events.columns:
                events['event_date'] = pd.to_datetime(events['event_date'])
            
            upcoming_events = events[
                (events['event_date'] >= datetime.now()) &
                (events['event_date'] <= datetime.now() + timedelta(days=horizon_days))
            ]
            
            for _, event in upcoming_events.iterrows():
                event_date = pd.to_datetime(event['event_date'])
                days_until_event = (event_date - datetime.now()).days
                
                if 0 <= days_until_event < horizon_days:
                    impact = self.event_impact_model.predict_event_impact(
                        event_type=event.get('event_type', 'unknown'),
                        surprise_factor=event.get('surprise_factor'),
                    )
                    
                    multiplier = impact.get('predicted_volatility_multiplier', 1.0)
                    event_adjustments.append({
                        "day": days_until_event,
                        "multiplier": multiplier,
                        "event_type": event.get('event_type'),
                    })
        
        # Apply event adjustments
        adjusted_forecast = np.array(base_forecast).copy()
        for adj in event_adjustments:
            day_idx = adj["day"]
            if day_idx < len(adjusted_forecast):
                adjusted_forecast[day_idx] *= adj["multiplier"]
        
        # Detect anomalies
        try:
            anomaly_score = self.anomaly_detector.calculate_anomaly_score(
                symbol_data_features,
                features=['rsi', 'macd', 'volatility_20', 'volume_ratio'],
            )
            recent_anomaly_score = anomaly_score.iloc[-1] if len(anomaly_score) > 0 else 0.0
        except Exception as e:
            logger.warning("Anomaly detection failed", error=str(e))
            recent_anomaly_score = 0.0
        
        # Adjust confidence based on anomalies
        if recent_anomaly_score > 0.7:
            base_confidence = [c * 0.8 for c in base_confidence]  # Lower confidence with anomalies
        
        # Calculate overall confidence
        avg_confidence = np.mean(base_confidence)
        
        # Generate prediction rationale
        rationale = self._generate_rationale(
            base_forecast,
            adjusted_forecast,
            event_adjustments,
            recent_anomaly_score,
            symbol_data,
        )
        
        return {
            "symbol": symbol,
            "forecast": adjusted_forecast.tolist(),
            "base_forecast": base_forecast,
            "confidence": base_confidence,
            "avg_confidence": float(avg_confidence),
            "horizon_days": horizon_days,
            "event_adjustments": event_adjustments,
            "anomaly_score": float(recent_anomaly_score),
            "rationale": rationale,
            "timestamp": datetime.now().isoformat(),
        }
    
    def _generate_rationale(
        self,
        base_forecast: List[float],
        adjusted_forecast: np.ndarray,
        event_adjustments: List[Dict],
        anomaly_score: float,
        market_data: pd.DataFrame,
    ) -> str:
        """
        Generate natural language rationale for prediction.
        
        Args:
            base_forecast: Base volatility forecast
            adjusted_forecast: Event-adjusted forecast
            event_adjustments: List of event adjustments
            anomaly_score: Current anomaly score
            market_data: Market data DataFrame
            
        Returns:
            Natural language rationale string
        """
        rationale_parts = []
        
        # Base forecast description
        avg_vol = np.mean(base_forecast)
        rationale_parts.append(
            f"Base volatility forecast: {avg_vol:.2%} average over forecast period"
        )
        
        # Event adjustments
        if event_adjustments:
            event_types = [adj["event_type"] for adj in event_adjustments]
            rationale_parts.append(
                f"Upcoming events detected: {', '.join(set(event_types))} "
                f"(volatility multiplier: {np.mean([adj['multiplier'] for adj in event_adjustments]):.2f}x)"
            )
        
        # Anomaly detection
        if anomaly_score > 0.7:
            rationale_parts.append(
                f"High anomaly score detected ({anomaly_score:.2f}) - unusual market patterns observed"
            )
        elif anomaly_score > 0.5:
            rationale_parts.append(
                f"Moderate anomaly score ({anomaly_score:.2f}) - some unusual patterns"
            )
        
        # Current market state
        if 'volatility_20' in market_data.columns:
            current_vol = market_data['volatility_20'].iloc[-1]
            if not pd.isna(current_vol):
                rationale_parts.append(
                    f"Current 20-day volatility: {current_vol:.2%}"
                )
        
        return ". ".join(rationale_parts) + "."
    
    def calculate_confidence_score(
        self,
        prediction: Dict[str, Any],
        historical_accuracy: Optional[float] = None,
    ) -> float:
        """
        Calculate overall confidence score for prediction.
        
        Args:
            prediction: Prediction dictionary
            historical_accuracy: Historical prediction accuracy (optional)
            
        Returns:
            Confidence score (0-1)
        """
        # Start with base confidence
        confidence = prediction.get("avg_confidence", 0.5)
        
        # Adjust for anomaly score (lower confidence with anomalies)
        anomaly_score = prediction.get("anomaly_score", 0.0)
        confidence *= (1 - anomaly_score * 0.3)
        
        # Adjust for historical accuracy if available
        if historical_accuracy is not None:
            confidence = (confidence + historical_accuracy) / 2
        
        # Ensure bounds
        confidence = max(0.0, min(1.0, confidence))
        
        return confidence
    
    def backtest_prediction(
        self,
        market_data: pd.DataFrame,
        symbol: str,
        prediction_date: datetime,
        horizon_days: int = 5,
    ) -> Dict[str, Any]:
        """
        Backtest prediction accuracy on historical data.
        
        Args:
            market_data: Historical market data
            symbol: Trading symbol
            prediction_date: Date to simulate prediction from
            horizon_days: Forecast horizon
            
        Returns:
            Dictionary with backtest results
        """
        # Filter data up to prediction date
        symbol_data = market_data[market_data['symbol'] == symbol].copy()
        
        # Handle timestamp/date column
        if 'timestamp' in symbol_data.columns:
            symbol_data['timestamp'] = pd.to_datetime(symbol_data['timestamp'])
        elif 'date' in symbol_data.columns:
            symbol_data['timestamp'] = pd.to_datetime(symbol_data['date'])
        
        historical_data = symbol_data[symbol_data['timestamp'] <= prediction_date].copy()
        actual_data = symbol_data[
            (symbol_data['timestamp'] > prediction_date) &
            (symbol_data['timestamp'] <= prediction_date + timedelta(days=horizon_days))
        ].copy()
        
        if len(historical_data) < 30 or len(actual_data) == 0:
            return {"error": "Insufficient data for backtest"}
        
        # Make prediction
        try:
            prediction = self.predict_volatility(
                market_data=historical_data,
                symbol=symbol,
                horizon_days=horizon_days,
            )
        except Exception as e:
            return {"error": f"Prediction failed: {str(e)}"}
        
        # Calculate actual volatility
        if 'returns' not in actual_data.columns:
            actual_data['returns'] = actual_data['close'].pct_change()
        
        actual_returns = actual_data['returns'].dropna()
        actual_volatility = actual_returns.std() * np.sqrt(252)  # Annualized
        
        predicted_volatility = np.mean(prediction["forecast"])
        
        # Calculate metrics
        error = abs(predicted_volatility - actual_volatility)
        error_pct = (error / actual_volatility * 100) if actual_volatility > 0 else 0
        
        return {
            "prediction_date": prediction_date.isoformat(),
            "predicted_volatility": float(predicted_volatility),
            "actual_volatility": float(actual_volatility),
            "error": float(error),
            "error_percentage": float(error_pct),
            "horizon_days": horizon_days,
            "confidence": prediction.get("avg_confidence", 0.5),
        }

