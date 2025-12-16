"""
Main FastAPI application for Market Activity Prediction Agent.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
import structlog

from src.utils.config import settings
from src.utils.logger import configure_logging, get_logger
from src.api.routes import predictions, scenarios, events, patterns, backtest, chatbot
from src.api.schemas import ErrorResponse

# Configure logging
configure_logging()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown tasks."""
    # Startup
    logger.info("Starting Market Activity Prediction Agent API")
    
    # Test BigQuery connection
    try:
        from src.data.bigquery_client import BigQueryClient
        bq_client = BigQueryClient()
        if bq_client.test_connection():
            logger.info("BigQuery connection verified")
        else:
            logger.warning("BigQuery connection test failed")
    except Exception as e:
        logger.warning("BigQuery connection not available", error=str(e))
    
    yield
    
    # Shutdown
    logger.info("Shutting down Market Activity Prediction Agent API")


# Create FastAPI app
app = FastAPI(
    title="Market Activity Prediction Agent API",
    description="API for market volatility forecasting, event impact modeling, and anomaly detection",
    version="1.0.0",
    lifespan=lifespan,
    # Authentication disabled - all endpoints are public
    swagger_ui_init_oauth=None,
    swagger_ui_oauth2_redirect_url=None,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(predictions.router, prefix="/api/v1")
app.include_router(scenarios.router, prefix="/api/v1")
app.include_router(events.router, prefix="/api/v1")
app.include_router(patterns.router, prefix="/api/v1")
app.include_router(backtest.router, prefix="/api/v1")

# Import alerts router
from src.api.routes import alerts
app.include_router(alerts.router, prefix="/api/v1")

# Include chatbot router
app.include_router(chatbot.router, prefix="/api/v1")


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Market Activity Prediction Agent API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "market-prediction-agent",
    }


@app.get("/chatbot")
async def chatbot_ui():
    """Serve the chatbot web interface."""
    chatbot_file = Path(__file__).parent.parent.parent / "chatbot_demo.html"
    if chatbot_file.exists():
        return FileResponse(chatbot_file)
    else:
        raise HTTPException(status_code=404, detail="Chatbot UI not found")


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    logger.error("Unhandled exception", error=str(exc), path=request.url.path)
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal server error",
            detail=str(exc),
            timestamp=datetime.now().isoformat(),
        ).dict(),
    )


if __name__ == "__main__":
    import uvicorn
    from datetime import datetime
    
    uvicorn.run(
        "src.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True,
    )

