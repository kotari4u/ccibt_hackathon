"""
Volatility forecasting models.
Implements GARCH, Prophet, and ensemble methods for volatility prediction.
"""

from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import structlog
import warnings

# Optional imports - handle gracefully if not available
try:
    from arch import arch_model
    ARCH_AVAILABLE = True
except ImportError:
    ARCH_AVAILABLE = False
    logger = structlog.get_logger(__name__)
    logger.warning("arch package not available - GARCH models will be disabled")

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    if 'logger' not in locals():
        logger = structlog.get_logger(__name__)
    logger.warning("prophet package not available - Prophet forecasts will be disabled")

warnings.filterwarnings('ignore')
if 'logger' not in locals():
    logger = structlog.get_logger(__name__)


class VolatilityForecaster:
    """
    Forecasts market volatility using multiple models.
    
    Supports GARCH for short-term, Prophet for medium-term, and ensemble methods.
    """
    
    def __init__(self):
        """Initialize volatility forecaster."""
        self.garch_model = None
        self.prophet_model = None
        self.ensemble_model = None
        self.logger = logger
    
    def calculate_realized_volatility(
        self,
        returns: pd.Series,
        window: int = 20,
        annualized: bool = True,
    ) -> pd.Series:
        """
        Calculate realized volatility from returns.
        
        Args:
            returns: Returns series
            window: Rolling window size
            annualized: Whether to annualize
            
        Returns:
            Realized volatility series
        """
        volatility = returns.rolling(window=window).std()
        
        if annualized:
            volatility = volatility * np.sqrt(252)  # Annualize
        
        return volatility
    
    def fit_garch(
        self,
        returns: pd.Series,
        p: int = 1,
        q: int = 1,
        dist: str = "t",
    ) -> Dict[str, Any]:
        """
        Fit GARCH model for volatility forecasting.
        
        Args:
            returns: Returns series
            p: GARCH lag order
            q: ARCH lag order
            dist: Error distribution ('normal', 't', 'skewt')
            
        Returns:
            Dictionary with model and metadata
        """
        if not ARCH_AVAILABLE:
            raise ImportError(
                "arch package not available. Install with: pip install arch"
            )
        
        try:
            # Remove NaN values
            returns_clean = returns.dropna()
            
            if len(returns_clean) < 100:
                raise ValueError("Insufficient data for GARCH model")
            
            # Fit GARCH model
            model = arch_model(
                returns_clean * 100,  # Scale for numerical stability
                vol="GARCH",
                p=p,
                q=q,
                dist=dist,
            )
            
            fitted_model = model.fit(disp="off")
            
            self.garch_model = fitted_model
            
            # Get fitted volatility
            conditional_vol = fitted_model.conditional_volatility / 100  # Rescale
            
            metadata = {
                "model_type": "GARCH",
                "aic": float(fitted_model.aic),
                "bic": float(fitted_model.bic),
                "log_likelihood": float(fitted_model.loglikelihood),
                "n_observations": len(returns_clean),
            }
            
            logger.info("GARCH model fitted successfully", **metadata)
            
            return {
                "model": fitted_model,
                "fitted_volatility": conditional_vol,
                "metadata": metadata,
            }
        
        except Exception as e:
            logger.error("GARCH model fitting failed", error=str(e))
            raise
    
    def forecast_garch(
        self,
        horizon: int = 5,
        start: Optional[pd.Timestamp] = None,
    ) -> Dict[str, Any]:
        """
        Forecast volatility using GARCH model.
        
        Args:
            horizon: Forecast horizon in periods
            start: Start date for forecast
            
        Returns:
            Dictionary with forecast and confidence intervals
        """
        if self.garch_model is None:
            raise ValueError("GARCH model not fitted. Call fit_garch() first.")
        
        try:
            forecast = self.garch_model.forecast(horizon=horizon, start=start)
            
            # Extract forecasted variance and convert to volatility
            forecast_variance = forecast.variance.values[-1] / 10000  # Rescale
            forecast_vol = np.sqrt(forecast_variance)
            
            # Calculate confidence intervals
            lower_ci = forecast_variance * 0.8  # Approximate 80% CI
            upper_ci = forecast_variance * 1.2
            lower_vol = np.sqrt(lower_ci)
            upper_vol = np.sqrt(upper_ci)
            
            return {
                "forecast": forecast_vol,
                "lower_ci": lower_vol,
                "upper_ci": upper_vol,
                "horizon": horizon,
            }
        
        except Exception as e:
            logger.error("GARCH forecast failed", error=str(e))
            raise
    
    def fit_prophet(
        self,
        volatility: pd.Series,
        daily_seasonality: bool = True,
        weekly_seasonality: bool = True,
    ) -> Dict[str, Any]:
        """
        Fit Prophet model for volatility forecasting.
        
        Args:
            volatility: Historical volatility series
            daily_seasonality: Include daily seasonality
            weekly_seasonality: Include weekly seasonality
            
        Returns:
            Dictionary with model and metadata
        """
        if not PROPHET_AVAILABLE:
            raise ImportError(
                "prophet package not available. Install with: pip install prophet"
            )
        
        try:
            # Prepare data for Prophet
            df = pd.DataFrame({
                'ds': volatility.index,
                'y': volatility.values,
            })
            df = df.dropna()
            
            if len(df) < 30:
                raise ValueError("Insufficient data for Prophet model")
            
            # Initialize and fit Prophet
            model = Prophet(
                daily_seasonality=daily_seasonality,
                weekly_seasonality=weekly_seasonality,
                yearly_seasonality=False,
            )
            
            model.fit(df)
            self.prophet_model = model
            
            metadata = {
                "model_type": "Prophet",
                "n_observations": len(df),
                "date_range": {
                    "start": str(df['ds'].min()),
                    "end": str(df['ds'].max()),
                },
            }
            
            logger.info("Prophet model fitted successfully", **metadata)
            
            return {
                "model": model,
                "metadata": metadata,
            }
        
        except Exception as e:
            logger.error("Prophet model fitting failed", error=str(e))
            raise
    
    def forecast_prophet(
        self,
        periods: int = 30,
        freq: str = "D",
    ) -> Dict[str, Any]:
        """
        Forecast volatility using Prophet model.
        
        Args:
            periods: Number of periods to forecast
            freq: Frequency string ('D', 'H', etc.)
            
        Returns:
            Dictionary with forecast and confidence intervals
        """
        if self.prophet_model is None:
            raise ValueError("Prophet model not fitted. Call fit_prophet() first.")
        
        try:
            # Create future dataframe
            future = self.prophet_model.make_future_dataframe(periods=periods, freq=freq)
            
            # Forecast
            forecast = self.prophet_model.predict(future)
            
            # Extract forecast
            forecast_df = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(periods)
            
            return {
                "forecast": forecast_df['yhat'].values,
                "lower_ci": forecast_df['yhat_lower'].values,
                "upper_ci": forecast_df['yhat_upper'].values,
                "dates": forecast_df['ds'].values,
                "horizon": periods,
            }
        
        except Exception as e:
            logger.error("Prophet forecast failed", error=str(e))
            raise
    
    def fit_ensemble(
        self,
        features: pd.DataFrame,
        target: pd.Series,
        model_type: str = "xgboost",
    ) -> Dict[str, Any]:
        """
        Fit ensemble model for volatility forecasting.
        
        Args:
            features: Feature DataFrame
            target: Target volatility series
            model_type: Model type ('xgboost', 'random_forest')
            
        Returns:
            Dictionary with model and metadata
        """
        try:
            # Align features and target
            aligned_data = pd.concat([features, target], axis=1).dropna()
            
            if len(aligned_data) < 50:
                raise ValueError("Insufficient data for ensemble model")
            
            X = aligned_data[features.columns]
            y = aligned_data[target.name]
            
            if model_type == "xgboost":
                model = xgb.XGBRegressor(
                    n_estimators=100,
                    max_depth=5,
                    learning_rate=0.1,
                    random_state=42,
                )
            elif model_type == "random_forest":
                model = RandomForestRegressor(
                    n_estimators=100,
                    max_depth=10,
                    random_state=42,
                )
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            model.fit(X, y)
            self.ensemble_model = model
            
            # Calculate feature importance
            if hasattr(model, 'feature_importances_'):
                feature_importance = dict(zip(features.columns, model.feature_importances_))
            else:
                feature_importance = {}
            
            metadata = {
                "model_type": model_type,
                "n_features": len(features.columns),
                "n_samples": len(aligned_data),
                "feature_importance": feature_importance,
            }
            
            logger.info("Ensemble model fitted successfully", **metadata)
            
            return {
                "model": model,
                "metadata": metadata,
            }
        
        except Exception as e:
            logger.error("Ensemble model fitting failed", error=str(e))
            raise
    
    def forecast_ensemble(
        self,
        features: pd.DataFrame,
    ) -> Dict[str, Any]:
        """
        Forecast volatility using ensemble model.
        
        Args:
            features: Feature DataFrame for prediction
            
        Returns:
            Dictionary with forecast
        """
        if self.ensemble_model is None:
            raise ValueError("Ensemble model not fitted. Call fit_ensemble() first.")
        
        try:
            # Ensure features match training features
            missing_cols = set(self.ensemble_model.feature_names_in_) - set(features.columns)
            if missing_cols:
                raise ValueError(f"Missing features: {missing_cols}")
            
            # Select and order features
            X = features[self.ensemble_model.feature_names_in_]
            
            # Predict
            predictions = self.ensemble_model.predict(X)
            
            return {
                "forecast": predictions,
                "n_predictions": len(predictions),
            }
        
        except Exception as e:
            logger.error("Ensemble forecast failed", error=str(e))
            raise
    
    def ensemble_forecast(
        self,
        returns: pd.Series,
        features: Optional[pd.DataFrame] = None,
        horizon: int = 5,
        use_garch: bool = True,
        use_prophet: bool = True,
        use_ensemble: bool = True,
    ) -> Dict[str, Any]:
        """
        Create ensemble forecast combining multiple models.
        
        Args:
            returns: Returns series
            features: Optional feature DataFrame for ensemble model
            horizon: Forecast horizon
            use_garch: Include GARCH forecast
            use_prophet: Include Prophet forecast
            use_ensemble: Include ML ensemble forecast
            
        Returns:
            Dictionary with combined forecast and confidence scores
        """
        forecasts = []
        weights = []
        
        # GARCH forecast (short-term)
        if use_garch:
            try:
                if self.garch_model is None:
                    self.fit_garch(returns)
                garch_result = self.forecast_garch(horizon=min(horizon, 5))
                forecasts.append(garch_result["forecast"])
                weights.append(0.4)  # Higher weight for short-term
            except Exception as e:
                logger.warning("GARCH forecast failed, skipping", error=str(e))
        
        # Prophet forecast (medium-term)
        if use_prophet:
            try:
                volatility = self.calculate_realized_volatility(returns)
                if self.prophet_model is None:
                    self.fit_prophet(volatility)
                prophet_result = self.forecast_prophet(periods=horizon)
                forecasts.append(prophet_result["forecast"][:horizon])
                weights.append(0.3)
            except Exception as e:
                logger.warning("Prophet forecast failed, skipping", error=str(e))
        
        # Ensemble forecast
        if use_ensemble and features is not None:
            try:
                if self.ensemble_model is None:
                    target = self.calculate_realized_volatility(returns)
                    self.fit_ensemble(features, target)
                ensemble_result = self.forecast_ensemble(features.tail(horizon))
                forecasts.append(ensemble_result["forecast"])
                weights.append(0.3)
            except Exception as e:
                logger.warning("Ensemble forecast failed, skipping", error=str(e))
        
        if not forecasts:
            raise ValueError("All forecast methods failed")
        
        # Normalize weights
        weights = np.array(weights) / sum(weights)
        
        # Combine forecasts
        min_len = min(len(f) for f in forecasts)
        forecasts_aligned = [f[:min_len] for f in forecasts]
        
        combined_forecast = np.zeros(min_len)
        for forecast, weight in zip(forecasts_aligned, weights):
            combined_forecast += np.array(forecast) * weight
        
        # Calculate confidence (based on agreement between models)
        if len(forecasts) > 1:
            forecast_std = np.std([f[:min_len] for f in forecasts], axis=0)
            confidence = 1 / (1 + forecast_std)  # Higher agreement = higher confidence
        else:
            confidence = np.array([0.7] * min_len)  # Default confidence
        
        return {
            "forecast": combined_forecast.tolist(),
            "confidence": confidence.tolist(),
            "horizon": min_len,
            "models_used": len(forecasts),
            "weights": weights.tolist(),
        }

