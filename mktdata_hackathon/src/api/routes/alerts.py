"""
API routes for alerts management.
"""

from typing import Optional
from fastapi import APIRouter, HTTPException, Depends, Query
from datetime import datetime
import structlog

from src.api.schemas import AlertRequest, AlertResponse, ActiveAlertsResponse
from src.alerts.alert_engine import AlertEngine, AlertSeverity
# Authentication disabled
# from src.api.auth import get_current_user

logger = structlog.get_logger(__name__)
router = APIRouter(prefix="/alerts", tags=["alerts"])

alert_engine = AlertEngine()


@router.get("/active", response_model=ActiveAlertsResponse)
async def get_active_alerts(
    symbol: Optional[str] = Query(default=None),
    severity: Optional[str] = Query(default=None),
    # Authentication disabled
    # current_user: dict = Depends(get_current_user),
) -> ActiveAlertsResponse:
    """
    Get active alerts.
    
    Args:
        symbol: Filter by symbol (optional)
        severity: Filter by severity (optional)
        current_user: Authenticated user
        
    Returns:
        List of active alerts
    """
    try:
        severity_enum = None
        if severity:
            try:
                severity_enum = AlertSeverity(severity.lower())
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid severity: {severity}",
                )
        
        alerts = alert_engine.get_active_alerts(
            symbol=symbol,
            severity=severity_enum,
        )
        
        alert_responses = [AlertResponse(**alert.to_dict()) for alert in alerts]
        
        return ActiveAlertsResponse(
            alerts=alert_responses,
            count=len(alert_responses),
            timestamp=datetime.now().isoformat(),
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get active alerts", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get alerts: {str(e)}",
        )


@router.post("/", response_model=AlertResponse)
async def create_alert(
    request: AlertRequest,
    # Authentication disabled
    # current_user: dict = Depends(get_current_user),
) -> AlertResponse:
    """
    Create a new alert.
    
    Args:
        request: Alert configuration
        current_user: Authenticated user
        
    Returns:
        Created alert
    """
    try:
        # This is a simplified implementation
        # In production, would integrate with monitoring system
        logger.info(
            "Alert creation requested",
            symbol=request.symbol,
            alert_type=request.alert_type,
        )
        
        # Return a mock alert for now
        # In production, would create actual alert in monitoring system
        from src.alerts.alert_engine import Alert, AlertType, AlertSeverity
        
        alert = Alert(
            alert_id=f"alert_{request.symbol}_{datetime.now().timestamp()}",
            symbol=request.symbol,
            alert_type=AlertType(request.alert_type),
            severity=AlertSeverity.MEDIUM,
            message=f"{request.alert_type} alert for {request.symbol}",
            threshold=request.threshold,
        )
        
        alert_engine.add_alert(alert)
        
        return AlertResponse(**alert.to_dict())
    
    except Exception as e:
        logger.error("Failed to create alert", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create alert: {str(e)}",
        )


@router.delete("/{alert_id}")
async def resolve_alert(
    alert_id: str,
    # Authentication disabled
    # current_user: dict = Depends(get_current_user),
) -> dict:
    """
    Resolve (deactivate) an alert.
    
    Args:
        alert_id: Alert ID
        current_user: Authenticated user
        
    Returns:
        Success message
    """
    try:
        success = alert_engine.resolve_alert(alert_id)
        
        if not success:
            raise HTTPException(
                status_code=404,
                detail=f"Alert not found: {alert_id}",
            )
        
        return {"message": "Alert resolved", "alert_id": alert_id}
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to resolve alert", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to resolve alert: {str(e)}",
        )

