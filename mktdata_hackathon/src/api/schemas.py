"""
Pydantic schemas for API request/response models.
"""

from typing import List, Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    """Request schema for volatility prediction."""
    symbol: str = Field(..., description="Trading symbol (e.g., SPY, AAPL)")
    horizon_days: int = Field(default=5, ge=1, le=30, description="Forecast horizon in days")
    include_events: bool = Field(default=True, description="Include event impact adjustments")


class PredictionResponse(BaseModel):
    """Response schema for volatility prediction."""
    symbol: str
    forecast: List[float] = Field(..., description="Volatility forecast for each day")
    base_forecast: List[float]
    confidence: List[float] = Field(..., description="Confidence scores for each day")
    avg_confidence: float
    horizon_days: int
    event_adjustments: List[Dict[str, Any]]
    anomaly_score: float
    rationale: str
    timestamp: str


class ScenarioRequest(BaseModel):
    """Request schema for scenario simulation."""
    symbol: str
    current_price: float
    volatility: float
    days: int = Field(default=5, ge=1, le=30)
    scenarios: Optional[Dict[str, Dict[str, float]]] = None
    n_simulations: Optional[int] = Field(default=10000, ge=1000, le=100000)


class ScenarioResponse(BaseModel):
    """Response schema for scenario simulation."""
    symbol: str
    current_price: float
    scenarios: Dict[str, Any]
    monte_carlo_results: Optional[Dict[str, Any]] = None
    timestamp: str


class EventRequest(BaseModel):
    """Request schema for event impact prediction."""
    event_type: str
    symbol: Optional[str] = None
    surprise_factor: Optional[float] = None
    event_date: Optional[datetime] = None


class EventResponse(BaseModel):
    """Response schema for event impact."""
    event_type: str
    predicted_volatility_multiplier: float
    method: str
    confidence: Optional[float] = None
    timestamp: str


class UpcomingEventsResponse(BaseModel):
    """Response schema for upcoming events."""
    events: List[Dict[str, Any]]
    count: int
    timestamp: str


class AlertRequest(BaseModel):
    """Request schema for creating alerts."""
    symbol: str
    alert_type: str = Field(..., description="Type: volatility, anomaly, event, pattern")
    threshold: float
    condition: str = Field(default="above", description="above or below")


class AlertResponse(BaseModel):
    """Response schema for alerts."""
    alert_id: str
    symbol: str
    alert_type: str
    threshold: float
    condition: str
    status: str
    created_at: str
    triggered_at: Optional[str] = None


class ActiveAlertsResponse(BaseModel):
    """Response schema for active alerts."""
    alerts: List[AlertResponse]
    count: int
    timestamp: str


class PatternDetectionRequest(BaseModel):
    """Request schema for pattern detection."""
    symbol: str
    lookback_days: int = Field(default=60, ge=1, le=252)


class PatternDetectionResponse(BaseModel):
    """Response schema for pattern detection."""
    symbol: str
    anomalies: Dict[str, Any]
    anomaly_score: float
    regime_changes: List[Dict[str, Any]]
    timestamp: str


class BacktestRequest(BaseModel):
    """Request schema for backtesting."""
    symbol: str
    prediction_date: str = Field(..., description="ISO format date")
    horizon_days: int = Field(default=5, ge=1, le=30)


class BacktestResponse(BaseModel):
    """Response schema for backtest results."""
    prediction_date: str
    predicted_volatility: float
    actual_volatility: float
    error: float
    error_percentage: float
    horizon_days: int
    confidence: float


class ErrorResponse(BaseModel):
    """Error response schema."""
    error: str
    detail: Optional[str] = None
    timestamp: str

