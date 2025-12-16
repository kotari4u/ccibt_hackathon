#!/usr/bin/env python3
"""
Debug mode run script for Market Activity Prediction Agent.
Enables verbose logging, auto-reload, and debug features.
"""

import os
import sys

def main():
    """Run the FastAPI application in debug mode."""
    print("Market Activity Prediction Agent - DEBUG MODE")
    print("=" * 50)
    
    # Check Python version
    if sys.version_info < (3, 11):
        print("Error: Python 3.11 or higher is required")
        sys.exit(1)
    
    # Set debug environment variables
    os.environ["LOG_LEVEL"] = "DEBUG"
    os.environ["LOG_FORMAT"] = "text"  # Use text format for easier reading in debug
    
    # Check if .env exists
    if not os.path.exists(".env"):
        print("Warning: .env file not found")
        if os.path.exists(".env.example"):
            print("Creating .env from .env.example...")
            import shutil
            shutil.copy(".env.example", ".env")
            print("Please edit .env with your configuration")
        else:
            print("Error: .env.example not found")
            sys.exit(1)
    
    # Import and run
    try:
        import uvicorn
        from src.utils.config import settings
        
        print(f"\nStarting API server in DEBUG MODE...")
        print(f"Host: {settings.api_host}")
        print(f"Port: {settings.api_port}")
        print(f"Log Level: DEBUG")
        print(f"Log Format: TEXT (human-readable)")
        print(f"Auto-reload: ENABLED")
        print(f"\nAPI Documentation: http://localhost:{settings.api_port}/docs")
        print(f"Health Check: http://localhost:{settings.api_port}/health")
        print("\nPress Ctrl+C to stop\n")
        
        uvicorn.run(
            "src.api.main:app",
            host=settings.api_host,
            port=settings.api_port,
            reload=True,  # Auto-reload on code changes
            log_level="debug",  # Uvicorn debug logging
            access_log=True,  # Enable access logs
        )
    except ImportError as e:
        print(f"Error: Missing dependencies. Run: pip install -r requirements.txt")
        print(f"Details: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error starting server: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()

