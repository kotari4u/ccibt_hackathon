"""
Real-time alert system for market activity.
Monitors thresholds, events, and patterns to generate alerts.
"""

from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta
from enum import Enum
import pandas as pd
import structlog

from src.utils.config import settings

logger = structlog.get_logger(__name__)


class AlertSeverity(str, Enum):
    """Alert severity levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class AlertType(str, Enum):
    """Alert types."""
    VOLATILITY = "volatility"
    ANOMALY = "anomaly"
    EVENT = "event"
    PATTERN = "pattern"
    THRESHOLD = "threshold"


class Alert:
    """Alert data structure."""
    
    def __init__(
        self,
        alert_id: str,
        symbol: str,
        alert_type: AlertType,
        severity: AlertSeverity,
        message: str,
        threshold: Optional[float] = None,
        current_value: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Initialize alert."""
        self.alert_id = alert_id
        self.symbol = symbol
        self.alert_type = alert_type
        self.severity = severity
        self.message = message
        self.threshold = threshold
        self.current_value = current_value
        self.metadata = metadata or {}
        self.created_at = datetime.now()
        self.triggered_at = datetime.now()
        self.status = "active"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary."""
        return {
            "alert_id": self.alert_id,
            "symbol": self.symbol,
            "alert_type": self.alert_type.value,
            "severity": self.severity.value,
            "message": self.message,
            "threshold": self.threshold,
            "current_value": self.current_value,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "triggered_at": self.triggered_at.isoformat(),
            "status": self.status,
        }


class AlertEngine:
    """
    Engine for generating and managing market alerts.
    
    Monitors volatility, anomalies, events, and patterns to trigger alerts.
    """
    
    def __init__(self):
        """Initialize alert engine."""
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        self.logger = logger
    
    def check_volatility_threshold(
        self,
        symbol: str,
        current_volatility: float,
        threshold_multiplier: float = None,
        baseline_volatility: Optional[float] = None,
    ) -> Optional[Alert]:
        """
        Check if volatility exceeds threshold.
        
        Args:
            symbol: Trading symbol
            current_volatility: Current volatility level
            threshold_multiplier: Multiplier for baseline volatility
            baseline_volatility: Baseline volatility (if None, uses threshold_multiplier)
            
        Returns:
            Alert if threshold exceeded, None otherwise
        """
        threshold_multiplier = threshold_multiplier or settings.alert_volatility_threshold
        
        if baseline_volatility is None:
            # Use default threshold
            threshold = threshold_multiplier * 0.15  # Assume 15% baseline
        else:
            threshold = baseline_volatility * threshold_multiplier
        
        if current_volatility > threshold:
            severity = (
                AlertSeverity.CRITICAL
                if current_volatility > threshold * 1.5
                else AlertSeverity.HIGH
            )
            
            alert = Alert(
                alert_id=f"vol_{symbol}_{datetime.now().timestamp()}",
                symbol=symbol,
                alert_type=AlertType.VOLATILITY,
                severity=severity,
                message=f"High volatility detected: {current_volatility:.2%} (threshold: {threshold:.2%})",
                threshold=threshold,
                current_value=current_volatility,
                metadata={"multiplier": threshold_multiplier},
            )
            
            return alert
        
        return None
    
    def check_anomaly(
        self,
        symbol: str,
        anomaly_score: float,
        threshold: float = 0.7,
    ) -> Optional[Alert]:
        """
        Check if anomaly score exceeds threshold.
        
        Args:
            symbol: Trading symbol
            anomaly_score: Current anomaly score (0-1)
            threshold: Anomaly threshold
            
        Returns:
            Alert if anomaly detected, None otherwise
        """
        if anomaly_score > threshold:
            severity = (
                AlertSeverity.CRITICAL
                if anomaly_score > 0.9
                else AlertSeverity.HIGH if anomaly_score > 0.8 else AlertSeverity.MEDIUM
            )
            
            alert = Alert(
                alert_id=f"anom_{symbol}_{datetime.now().timestamp()}",
                symbol=symbol,
                alert_type=AlertType.ANOMALY,
                severity=severity,
                message=f"Anomaly detected: score {anomaly_score:.2f} (threshold: {threshold:.2f})",
                threshold=threshold,
                current_value=anomaly_score,
            )
            
            return alert
        
        return None
    
    def check_upcoming_event(
        self,
        event_date: datetime,
        event_type: str,
        symbol: str,
        hours_before: int = 24,
    ) -> Optional[Alert]:
        """
        Check if high-impact event is approaching.
        
        Args:
            event_date: Event date/time
            event_type: Type of event
            symbol: Trading symbol
            hours_before: Hours before event to alert
            
        Returns:
            Alert if event is approaching, None otherwise
        """
        time_until_event = event_date - datetime.now()
        
        if timedelta(hours=0) < time_until_event <= timedelta(hours=hours_before):
            hours_remaining = time_until_event.total_seconds() / 3600
            
            alert = Alert(
                alert_id=f"event_{symbol}_{datetime.now().timestamp()}",
                symbol=symbol,
                alert_type=AlertType.EVENT,
                severity=AlertSeverity.HIGH,
                message=f"Upcoming {event_type} event in {hours_remaining:.1f} hours",
                metadata={
                    "event_type": event_type,
                    "event_date": event_date.isoformat(),
                    "hours_remaining": hours_remaining,
                },
            )
            
            return alert
        
        return None
    
    def check_pattern_match(
        self,
        symbol: str,
        pattern_type: str,
        confidence: float,
        threshold: float = 0.7,
    ) -> Optional[Alert]:
        """
        Check if pattern matches high-volatility scenario.
        
        Args:
            symbol: Trading symbol
            pattern_type: Type of pattern detected
            confidence: Pattern match confidence
            threshold: Confidence threshold
            
        Returns:
            Alert if pattern matches, None otherwise
        """
        if confidence > threshold:
            alert = Alert(
                alert_id=f"pattern_{symbol}_{datetime.now().timestamp()}",
                symbol=symbol,
                alert_type=AlertType.PATTERN,
                severity=AlertSeverity.MEDIUM,
                message=f"Pattern match detected: {pattern_type} (confidence: {confidence:.2f})",
                current_value=confidence,
                threshold=threshold,
                metadata={"pattern_type": pattern_type},
            )
            
            return alert
        
        return None
    
    def add_alert(self, alert: Alert) -> None:
        """
        Add alert to active alerts.
        
        Args:
            alert: Alert object
        """
        self.active_alerts[alert.alert_id] = alert
        self.alert_history.append(alert)
        
        logger.info(
            "Alert created",
            alert_id=alert.alert_id,
            symbol=alert.symbol,
            type=alert.alert_type.value,
            severity=alert.severity.value,
        )
    
    def get_active_alerts(
        self,
        symbol: Optional[str] = None,
        severity: Optional[AlertSeverity] = None,
    ) -> List[Alert]:
        """
        Get active alerts with optional filters.
        
        Args:
            symbol: Filter by symbol
            severity: Filter by severity
            
        Returns:
            List of active alerts
        """
        alerts = list(self.active_alerts.values())
        
        if symbol:
            alerts = [a for a in alerts if a.symbol == symbol]
        
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        
        return sorted(alerts, key=lambda x: x.created_at, reverse=True)
    
    def resolve_alert(self, alert_id: str) -> bool:
        """
        Resolve (deactivate) an alert.
        
        Args:
            alert_id: Alert ID
            
        Returns:
            True if alert was resolved, False if not found
        """
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.status = "resolved"
            del self.active_alerts[alert_id]
            
            logger.info("Alert resolved", alert_id=alert_id)
            return True
        
        return False
    
    def get_alert_history(
        self,
        symbol: Optional[str] = None,
        limit: int = 100,
    ) -> List[Alert]:
        """
        Get alert history.
        
        Args:
            symbol: Filter by symbol
            limit: Maximum number of alerts to return
            
        Returns:
            List of historical alerts
        """
        alerts = self.alert_history
        
        if symbol:
            alerts = [a for a in alerts if a.symbol == symbol]
        
        return sorted(alerts, key=lambda x: x.created_at, reverse=True)[:limit]

