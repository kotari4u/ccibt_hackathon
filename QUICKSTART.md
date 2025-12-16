# Quick Start Guide

## Prerequisites

1. Python 3.11 or higher
2. Google Cloud account with BigQuery access
3. (Optional) Docker for containerized deployment

## Installation Steps

### 1. Clone and Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
# Copy example environment file
cp .env.example .env

# Edit .env with your settings:
# - BIGQUERY_PROJECT_ID: Your GCP project ID
# - GOOGLE_APPLICATION_CREDENTIALS: Path to service account JSON
# - SECRET_KEY: Generate a secure random key for JWT
```

### 3. Set Up Google Cloud Credentials

```bash
# Option 1: Service Account Key File
# Download JSON key from Google Cloud Console
# Set GOOGLE_APPLICATION_CREDENTIALS=/path/to/key.json in .env

# Option 2: Application Default Credentials
gcloud auth application-default login
```

### 4. Verify BigQuery Access

```python
# Run Python script to test connection
python -c "from src.data.bigquery_client import BigQueryClient; \
           bq = BigQueryClient(); \
           print('Connected!' if bq.test_connection() else 'Failed')"
```

## Running the Application

### Development Mode

```bash
# Start API server
uvicorn src.api.main:app --reload

# API will be available at http://localhost:8000
# Documentation at http://localhost:8000/docs
```

### Using Docker

```bash
# Build image
docker build -t market-prediction-agent .

# Run container
docker run -p 8000:8000 --env-file .env market-prediction-agent
```

### Using Docker Compose

```bash
# Start all services (API + Redis)
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

## Testing the API

### 1. Get Authentication Token

```bash
# For testing, you can use a simple token
# In production, implement proper OAuth2 flow
```

### 2. Make API Calls

```bash
# Get volatility prediction
curl -X GET "http://localhost:8000/api/v1/predictions/SPY?horizon_days=5" \
  -H "Authorization: Bearer YOUR_TOKEN"

# Run scenario simulation
curl -X POST "http://localhost:8000/api/v1/scenarios/" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -d '{
    "symbol": "SPY",
    "current_price": 450.0,
    "volatility": 0.15,
    "days": 5
  }'

# Get upcoming events
curl -X GET "http://localhost:8000/api/v1/events/upcoming?days_ahead=30" \
  -H "Authorization: Bearer YOUR_TOKEN"
```

## Running Examples

```bash
# Run basic usage examples
python examples/basic_usage.py
```

## Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src tests/

# Run specific test file
pytest tests/test_data_processor.py
```

## Common Issues

### BigQuery Connection Failed

- Verify `GOOGLE_APPLICATION_CREDENTIALS` path is correct
- Check service account has BigQuery access
- Verify project ID is correct

### Import Errors

- Ensure virtual environment is activated
- Run `pip install -r requirements.txt` again
- Check Python version (3.11+)

### Port Already in Use

- Change `API_PORT` in `.env`
- Or use `--port` flag: `uvicorn src.api.main:app --port 8001`

## Next Steps

1. Review API documentation at `/docs`
2. Explore example notebooks (create in `notebooks/` directory)
3. Customize models and thresholds in configuration
4. Set up monitoring and alerting
5. Deploy to production environment

## Production Deployment

1. Set secure `SECRET_KEY` in environment
2. Configure CORS origins appropriately
3. Set up Redis for caching
4. Configure proper logging and monitoring
5. Set up CI/CD pipeline
6. Configure rate limiting
7. Set up SSL/TLS certificates

