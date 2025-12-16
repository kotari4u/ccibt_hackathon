# How to Run Market Activity Prediction Agent

## Quick Start (3 Steps)

### Step 1: Install Dependencies

```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Configure Environment

```bash
# Copy example environment file
cp .env.example .env

# Edit .env file (minimum required):
# - BIGQUERY_PROJECT_ID=your-project-id
# - SECRET_KEY=your-secret-key-here
```

**Note**: For testing without BigQuery, you can leave BigQuery settings empty. The API will still run but prediction endpoints will need mock data.

### Step 3: Run the Application

**Option A: Using the run script**
```bash
# Make script executable (Linux/Mac)
chmod +x run.sh
./run.sh

# Or use Python script
python run.py
```

**Option B: Direct uvicorn command**
```bash
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

**Option C: Using Python directly**
```bash
python -m src.api.main
```

## Verify It's Running

1. **Check health endpoint:**
```bash
curl http://localhost:8000/health
```

Expected response:
```json
{"status": "healthy", "service": "market-prediction-agent"}
```

2. **View API documentation:**
Open in browser: http://localhost:8000/docs

3. **Check root endpoint:**
```bash
curl http://localhost:8000/
```

## Testing the API

### Important: Authentication Required

The API uses JWT authentication. For testing, you have two options:

**Option 1: Create a test token (for development)**

Create a simple script `create_test_token.py`:

```python
from datetime import datetime, timedelta
from jose import jwt
from src.utils.config import settings

# Create a test token
data = {"sub": "test_user"}
token = jwt.encode(data, settings.secret_key, algorithm=settings.algorithm)
print(f"Test token: {token}")
```

Run it:
```bash
python create_test_token.py
```

**Option 2: Temporarily disable auth (for testing only)**

Edit `src/api/routes/predictions.py` and comment out the auth dependency:

```python
# current_user: dict = Depends(get_current_user),
current_user: dict = {"username": "test"}  # Temporary
```

### Example API Calls

**1. Health Check (No auth required)**
```bash
curl http://localhost:8000/health
```

**2. Get Prediction (Requires auth token)**
```bash
# Replace YOUR_TOKEN with actual token
curl -X GET "http://localhost:8000/api/v1/predictions/SPY?horizon_days=5" \
  -H "Authorization: Bearer YOUR_TOKEN"
```

**3. Run Scenario Simulation**
```bash
curl -X POST "http://localhost:8000/api/v1/scenarios/" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -d '{
    "symbol": "SPY",
    "current_price": 450.0,
    "volatility": 0.15,
    "days": 5
  }'
```

## Running with Docker

### Build and Run

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
docker-compose logs -f api

# Stop services
docker-compose down
```

## Common Issues & Solutions

### Issue 1: "Module not found" errors

**Solution:**
```bash
# Ensure virtual environment is activated
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Reinstall dependencies
pip install -r requirements.txt
```

### Issue 2: "BigQuery connection failed"

**Solution:**
- This is OK for testing! The API will still run
- For production, ensure:
  - `GOOGLE_APPLICATION_CREDENTIALS` points to valid service account JSON
  - Service account has BigQuery access
  - `BIGQUERY_PROJECT_ID` is correct

### Issue 3: "Port 8000 already in use"

**Solution:**
```bash
# Option 1: Use different port
uvicorn src.api.main:app --port 8001

# Option 2: Find and kill process using port 8000
# On Linux/Mac:
lsof -ti:8000 | xargs kill
# On Windows:
netstat -ano | findstr :8000
taskkill /PID <PID> /F
```

### Issue 4: "Authentication failed"

**Solution:**
- Check `SECRET_KEY` in `.env` is set
- Ensure token is properly formatted: `Bearer <token>`
- For testing, temporarily disable auth (see Option 2 above)

### Issue 5: Import errors with Prophet or arch

**Solution:**
```bash
# These packages may need additional system dependencies
# On Ubuntu/Debian:
sudo apt-get install build-essential

# On Mac:
xcode-select --install

# Then reinstall:
pip install --upgrade prophet arch
```

## Development Mode

For development with auto-reload:

```bash
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

The `--reload` flag automatically restarts the server when code changes.

## Production Deployment

For production, use:

```bash
uvicorn src.api.main:app \
  --host 0.0.0.0 \
  --port 8000 \
  --workers 4 \
  --log-level info
```

Or use a process manager like `gunicorn`:

```bash
gunicorn src.api.main:app \
  -w 4 \
  -k uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000
```

## Next Steps

1. ✅ Verify API is running: http://localhost:8000/health
2. ✅ Explore API docs: http://localhost:8000/docs
3. ✅ Test endpoints with curl or Postman
4. ✅ Configure BigQuery for real data
5. ✅ Set up monitoring and logging

## Need Help?

- Check logs in console output
- Review API documentation at `/docs`
- See `README.md` for full documentation
- Check `QUICKSTART.md` for detailed setup

