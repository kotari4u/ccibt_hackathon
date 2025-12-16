# Market Activity Prediction Agent

A comprehensive system for analyzing historical market data, modeling event impacts, forecasting volatility, and providing real-time alerts for trading decisions.

## Features

- **Volatility Forecasting**: Multiple models (GARCH, Prophet, Ensemble) for short and medium-term volatility predictions
- **Event Impact Modeling**: Correlates financial events with historical market reactions
- **Anomaly Detection**: Statistical and ML-based anomaly detection for unusual market patterns
- **Scenario Simulation**: Monte Carlo simulations and stress testing capabilities
- **Real-time Alerts**: Threshold-based and event-driven alerting system
- **REST API**: FastAPI-based API with comprehensive endpoints
- **Model Explainability**: SHAP values and feature importance for interpretable predictions

## Architecture

```
market-prediction-agent/
├── src/
│   ├── data/              # Data ingestion and processing
│   │   ├── bigquery_client.py
│   │   ├── data_processor.py
│   │   └── feature_engineering.py
│   ├── models/            # ML models and forecasting
│   │   ├── volatility_forecaster.py
│   │   ├── event_impact_model.py
│   │   ├── anomaly_detector.py
│   │   ├── ensemble_predictor.py
│   │   ├── scenario_simulator.py
│   │   └── explainability.py
│   ├── api/               # FastAPI application
│   │   ├── main.py
│   │   ├── schemas.py
│   │   ├── auth.py
│   │   └── routes/
│   ├── alerts/            # Alert system
│   │   ├── alert_engine.py
│   │   └── notification_handler.py
│   └── utils/             # Utilities
│       ├── config.py
│       └── logger.py
├── tests/                 # Test suite
├── requirements.txt       # Python dependencies
├── Dockerfile            # Docker configuration
└── README.md
```

## Installation

### Prerequisites

- Python 3.11+
- Google Cloud account with BigQuery access
- (Optional) Redis for caching
- (Optional) Twilio account for SMS alerts

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd market-prediction-agent
```

2. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure environment variables:
```bash
cp .env.example .env
# Edit .env with your configuration
```

5. Set up Google Cloud credentials:
   - Create a service account in Google Cloud Console
   - Download the JSON key file
   - Set `GOOGLE_APPLICATION_CREDENTIALS` in `.env` to the path of the key file

## Configuration

Key configuration options in `.env`:

- `BIGQUERY_PROJECT_ID`: Your Google Cloud project ID
- `BIGQUERY_DATASET`: BigQuery dataset name (default: `market_data`)
- `ALERT_VOLATILITY_THRESHOLD`: Volatility multiplier for alerts (default: 1.5)
- `CONFIDENCE_THRESHOLD`: Minimum confidence for predictions (default: 0.70)

## Usage

### Running the API Server

```bash
# Development
uvicorn src.api.main:app --reload

# Production
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`
- API Documentation: `http://localhost:8000/docs`
- Alternative docs: `http://localhost:8000/redoc`

### Docker Deployment

```bash
# Build image
docker build -t market-prediction-agent .

# Run container
docker run -p 8000:8000 --env-file .env market-prediction-agent
```

## API Endpoints

### Predictions

- `GET /api/v1/predictions/{symbol}` - Get volatility prediction
- `POST /api/v1/predictions/` - Create custom prediction

### Scenarios

- `POST /api/v1/scenarios/` - Run scenario simulation

### Events

- `GET /api/v1/events/upcoming` - Get upcoming financial events
- `POST /api/v1/events/impact` - Predict event impact

### Patterns

- `GET /api/v1/patterns/detect` - Detect patterns and anomalies

### Backtesting

- `POST /api/v1/backtest/` - Backtest prediction accuracy

## Example Usage

### Get Volatility Prediction

```bash
curl -X GET "http://localhost:8000/api/v1/predictions/SPY?horizon_days=5" \
  -H "Authorization: Bearer YOUR_TOKEN"
```

### Run Scenario Simulation

```bash
curl -X POST "http://localhost:8000/api/v1/scenarios/" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -d '{
    "symbol": "SPY",
    "current_price": 450.0,
    "volatility": 0.15,
    "days": 5,
    "scenarios": {
      "bull": {"vol_multiplier": 0.8, "drift": 0.1},
      "bear": {"vol_multiplier": 1.5, "drift": -0.1}
    }
  }'
```

## BigQuery Schema

The system expects the following BigQuery tables:

### market_data
- `timestamp` (TIMESTAMP)
- `symbol` (STRING)
- `open` (FLOAT)
- `high` (FLOAT)
- `low` (FLOAT)
- `close` (FLOAT)
- `volume` (INTEGER)
- `bid_ask_spread` (FLOAT)
- `trade_count` (INTEGER)

### financial_events
- `event_id` (STRING)
- `event_type` (STRING)
- `event_date` (TIMESTAMP)
- `symbol` (STRING)
- `description` (STRING)
- `actual_value` (FLOAT)
- `expected_value` (FLOAT)
- `surprise_factor` (FLOAT)
- `impact_score` (FLOAT)

### news_sentiment (optional)
- `timestamp` (TIMESTAMP)
- `symbol` (STRING)
- `headline` (STRING)
- `sentiment_score` (FLOAT)
- `source` (STRING)

## Testing

```bash
# Run tests
pytest

# With coverage
pytest --cov=src tests/
```

## Performance Metrics

Target performance:
- **Prediction Accuracy**: >65% directional accuracy
- **API Latency**: <200ms (cached), <2s (uncached)
- **Alert Precision**: >70% of high-severity alerts lead to volatility spikes
- **API Uptime**: >99.5%

## Model Explainability

The system provides explainability through:
- SHAP values for feature importance
- Natural language prediction rationales
- Confidence scores based on model agreement
- Historical accuracy tracking

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

[Specify your license]

## Support

For issues and questions, please open an issue on GitHub.

