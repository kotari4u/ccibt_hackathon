"""
API routes for financial events.
"""

from typing import Optional
from fastapi import APIRouter, HTTPException, Depends, Query
from datetime import datetime, timedelta
import structlog

from src.api.schemas import (
    EventRequest,
    EventResponse,
    UpcomingEventsResponse,
)
from src.models.event_impact_model import EventImpactModel
from src.data.bigquery_client import BigQueryClient
# Authentication disabled
# from src.api.auth import get_current_user

logger = structlog.get_logger(__name__)
router = APIRouter(prefix="/events", tags=["events"])

event_impact_model = EventImpactModel()
bq_client: Optional[BigQueryClient] = None


def get_bq_client() -> BigQueryClient:
    """Get BigQuery client instance."""
    global bq_client
    if bq_client is None:
        bq_client = BigQueryClient()
    return bq_client


@router.get("/upcoming", response_model=UpcomingEventsResponse)
async def get_upcoming_events(
    days_ahead: int = Query(default=30, ge=1, le=90),
    min_impact_score: Optional[float] = Query(default=None),
    # Authentication disabled
    # current_user: dict = Depends(get_current_user),
) -> UpcomingEventsResponse:
    """
    Get upcoming financial events.
    
    Args:
        days_ahead: Number of days to look ahead
        min_impact_score: Minimum impact score filter
        current_user: Authenticated user
        
    Returns:
        List of upcoming events
    """
    try:
        bq = get_bq_client()
        events_df = bq.get_upcoming_events(
            days_ahead=days_ahead,
            min_impact_score=min_impact_score,
        )
        
        events_list = events_df.to_dict('records') if len(events_df) > 0 else []
        
        return UpcomingEventsResponse(
            events=events_list,
            count=len(events_list),
            timestamp=datetime.now().isoformat(),
        )
    
    except Exception as e:
        logger.error("Failed to fetch upcoming events", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch events: {str(e)}",
        )


@router.post("/impact", response_model=EventResponse)
async def predict_event_impact(
    request: EventRequest,
    # Authentication disabled
    # current_user: dict = Depends(get_current_user),
) -> EventResponse:
    """
    Predict impact of a financial event.
    
    Args:
        request: Event impact prediction parameters
        current_user: Authenticated user
        
    Returns:
        Event impact prediction
    """
    try:
        impact = event_impact_model.predict_event_impact(
            event_type=request.event_type,
            surprise_factor=request.surprise_factor,
        )
        
        return EventResponse(
            event_type=request.event_type,
            predicted_volatility_multiplier=impact.get("predicted_volatility_multiplier", 1.0),
            method=impact.get("method", "multiplier"),
            confidence=impact.get("confidence"),
            timestamp=datetime.now().isoformat(),
        )
    
    except Exception as e:
        logger.error("Event impact prediction failed", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Event impact prediction failed: {str(e)}",
        )

