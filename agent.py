"""
Vertex AI Agent Engine entry point for Market Activity Prediction Agent.
This file wraps the FastAPI application for Agent Engine deployment.
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.api.main import app
from src.utils.config import settings
from src.utils.logger import configure_logging, get_logger

# Configure logging
configure_logging()
logger = get_logger(__name__)

# Export the FastAPI app for Agent Engine
# Agent Engine will look for 'app' variable
__all__ = ['app']

# Initialize on import
logger.info("Market Activity Prediction Agent initialized for Vertex AI Agent Engine")

