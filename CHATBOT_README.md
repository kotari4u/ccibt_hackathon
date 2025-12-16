# Market Prediction Chatbot

A natural language chatbot interface for market volatility predictions. Ask questions in plain English and get predictions for stocks and indices.

## Features

- **Natural Language Processing**: Understands queries in plain English
- **Automatic Symbol Detection**: Extracts stock symbols from your message
- **Time Horizon Extraction**: Understands time periods (days, weeks, months)
- **Event Impact Analysis**: Considers upcoming events in predictions
- **User-Friendly Interface**: Simple web-based chat interface

## API Endpoints

### POST `/api/v1/chatbot/chat`

Send a natural language query and get predictions.

**Request:**
```json
{
  "message": "What's the volatility forecast for AAPL?"
}
```

**Response:**
```json
{
  "response": "📊 **Prediction for AAPL** (Next 5 days)\n\n**Volatility Forecast:**\n• Average: 0.0234\n...",
  "prediction": {
    "symbol": "AAPL",
    "forecast": [0.0234, 0.0245, ...],
    "confidence": [0.85, 0.82, ...],
    ...
  },
  "parsed_query": {
    "symbol": "AAPL",
    "horizon_days": 5,
    "include_events": true
  }
}
```

### GET `/api/v1/chatbot/help`

Get help information and examples.

## Example Queries

- "What's the volatility forecast for AAPL?"
- "Predict SPY for the next 7 days"
- "Show me Apple stock prediction for next week"
- "What will happen to Tesla in the next month?"
- "Forecast Microsoft volatility for 10 days"
- "Tell me about DJI prediction for next 2 weeks"
- "What's the short-term forecast for GOOGL?"

## Supported Symbols

### Individual Stocks
- AAPL (Apple), MSFT (Microsoft), GOOGL (Google), AMZN (Amazon)
- TSLA (Tesla), NVDA (NVIDIA), META (Meta/Facebook)
- And many more...

### Indices
- SPY (S&P 500), DJI (Dow Jones), GSPC (S&P 500 Index)
- IXIC (NASDAQ), VIX (Volatility Index)

### Company Names
You can also use company names:
- "Apple" → AAPL
- "Microsoft" → MSFT
- "Tesla" → TSLA
- "Google" → GOOGL

## Time Horizons

The chatbot understands various time expressions:

- **Days**: "next 5 days", "7 days", "10 days", "tomorrow"
- **Weeks**: "next week", "2 weeks", "next 2 weeks"
- **Months**: "next month", "3 months"
- **Keywords**: 
  - "short term" → 3 days
  - "medium term" → 7 days
  - "long term" → 30 days

## Web Interface

A simple web interface is provided in `chatbot_demo.html`. To use it:

1. Start the API server:
   ```bash
   python3.12 run.py
   ```

2. Open `chatbot_demo.html` in your browser

3. Start chatting!

## Testing with curl

```bash
# Basic query
curl -X POST "http://localhost:8000/api/v1/chatbot/chat" \
  -H "Content-Type: application/json" \
  -d '{"message": "What is the volatility forecast for AAPL?"}'

# With time horizon
curl -X POST "http://localhost:8000/api/v1/chatbot/chat" \
  -H "Content-Type: application/json" \
  -d '{"message": "Predict SPY for the next 7 days"}'

# Get help
curl "http://localhost:8000/api/v1/chatbot/help"
```

## How It Works

1. **Natural Language Parsing**: The `nlp_parser.py` module extracts:
   - Stock symbols (from ticker codes or company names)
   - Time horizons (days, weeks, months)
   - Event inclusion preferences

2. **Symbol Detection**: Uses pattern matching and a symbol dictionary to identify:
   - Ticker symbols (AAPL, SPY, etc.)
   - Company names (Apple, Microsoft, etc.)

3. **Prediction Generation**: Calls the existing prediction API with extracted parameters

4. **Response Formatting**: Formats the prediction results in natural language

## Error Handling

The chatbot handles various error cases:

- **No symbol found**: Suggests example queries
- **No market data**: Informs user about invalid symbols
- **API errors**: Provides helpful error messages

## Integration

The chatbot integrates with:
- `EnsemblePredictor` for volatility predictions
- `BigQueryClient` for market data retrieval
- Event impact modeling for comprehensive analysis

## Future Enhancements

- Support for multiple symbols in one query
- Comparison queries ("Compare AAPL vs MSFT")
- Historical analysis queries
- More sophisticated NLP using LLMs
- Voice input support

