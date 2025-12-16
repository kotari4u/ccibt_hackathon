#!/bin/bash

# Run script for Market Activity Prediction Agent

set -e

echo "Market Activity Prediction Agent - Starting..."
echo "================================================"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install/update dependencies
echo "Installing dependencies..."
pip install -q --upgrade pip
pip install -q -r requirements.txt

# Check if .env exists
if [ ! -f ".env" ]; then
    echo "Warning: .env file not found. Creating from .env.example..."
    if [ -f ".env.example" ]; then
        cp .env.example .env
        echo "Please edit .env file with your configuration before running."
    else
        echo "Error: .env.example not found. Please create .env file manually."
        exit 1
    fi
fi

# Run the application
echo ""
echo "Starting API server..."
echo "API will be available at http://localhost:8000"
echo "Documentation at http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop"
echo ""

uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

