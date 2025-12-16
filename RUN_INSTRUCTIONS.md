# Complete Instructions to Run Market Activity Prediction Agent

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Configuration](#configuration)
4. [Running the Application](#running-the-application)
5. [Testing the API](#testing-the-api)
6. [Docker Deployment](#docker-deployment)
7. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Required
- **Python 3.11 or higher** (check with `python3 --version`)
- **pip** (Python package manager)
- **Git** (for cloning repository)

### Optional (for full functionality)
- **Google Cloud Account** with BigQuery access
- **Docker** (for containerized deployment)
- **Redis** (for caching - optional)

---

## Installation

### Step 1: Navigate to Project Directory

```bash
cd /Users/hemambarakotari/Hemambara/GenAI/Hackathon
```

### Step 2: Create Virtual Environment

**On macOS/Linux:**
```bash
python3.12 -m venv venv
source venv/bin/activate
```

**On Windows:**
```bash
python3.12 -m venv venv
venv\Scripts\activate
```

**Verify activation:**
- Your terminal prompt should show `(venv)` at the beginning

### Step 3: Install Dependencies

**Option A: Install Minimal Requirements (Recommended if you encounter errors)**

```bash
# Upgrade pip first
python3.12 -m pip install --upgrade pip

# Install core dependencies (without problematic packages)
python3.12 -m pip install -r requirements-minimal.txt
```

**Option B: Install Full Requirements**

```bash
# Upgrade pip first
python3.12 -m pip install --upgrade pip

# Install all dependencies
python3.12 -m pip install -r requirements.txt
```

**Note:** 
- If you get "Failed building wheel for llvmlite" error, use Option A (minimal requirements)
- The application works fine without `arch` and `prophet` packages
- See `INSTALL_TROUBLESHOOTING.md` for detailed solutions
- If you get "externally-managed-environment" error, make sure you've activated the virtual environment first

**Expected output:**
```
Successfully installed fastapi-0.104.1 uvicorn-0.24.0 ...
```

### Step 4: Verify Installation

```bash
# Check if key packages are installed
python3.12 -c "import fastapi, uvicorn, pandas; print('✓ Dependencies installed successfully')"
```

---

## Configuration

### Step 1: Create Environment File

```bash
# Copy example environment file
cp .env.example .env

# Or create manually
touch .env
```

### Step 2: Configure Environment Variables

Edit `.env` file with your settings:

```bash
# Minimum required configuration
SECRET_KEY=your-secret-key-here-change-in-production
BIGQUERY_PROJECT_ID=your-gcp-project-id
BIGQUERY_DATASET=market_data

# Optional: Google Cloud credentials
GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account-key.json

# Optional: API settings (defaults shown)
API_HOST=0.0.0.0
API_PORT=8000
```

**Generate a secure SECRET_KEY:**
```bash
python3.12 -c "import secrets; print(secrets.token_urlsafe(32))"
```

### Step 3: Configure BigQuery (Optional for Testing)

**Option A: Service Account Key File**
1. Download service account JSON from Google Cloud Console
2. Set path in `.env`: `GOOGLE_APPLICATION_CREDENTIALS=/path/to/key.json`

**Option B: Application Default Credentials**
```bash
gcloud auth application-default login
```

**Note:** The API will run without BigQuery, but prediction endpoints will need data to function properly.

---

## Running the Application

### Method 1: Using the Run Script (Recommended)

```bash
# Make script executable (Linux/Mac)
chmod +x run.py

# Run the application
python3.12 run.py
```

### Method 2: Direct uvicorn Command

```bash
# Development mode (with auto-reload)
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

# Production mode
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### Method 3: Using Python Module

```bash
python3 -m src.api.main
```

### Method 4: Using Shell Script (Linux/Mac)

```bash
chmod +x run.sh
./run.sh
```

### Expected Output

```
INFO:     Started server process [12345]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

---

## Verify Application is Running

### Step 1: Check Health Endpoint

```bash
# Using curl
curl http://localhost:8000/health

# Expected response:
# {"status":"healthy","service":"market-prediction-agent"}
```

### Step 2: Open API Documentation

Open in your browser:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Step 3: Test Root Endpoint

```bash
curl http://localhost:8000/

# Expected response:
# {"message":"Market Activity Prediction Agent API","version":"1.0.0",...}
```

---

## Testing the API

### Step 1: Generate Test Token

```bash
python3 create_test_token.py
```

**Output:**
```
============================================================
Test JWT Token Generated
============================================================

Token: eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...

============================================================
```

### Step 2: Test API Endpoints

**Health Check (No auth required):**
```bash
curl http://localhost:8000/health
```

**Get Prediction (Requires token):**
```bash
# Replace YOUR_TOKEN with token from Step 1
TOKEN="your-token-here"

curl -X GET "http://localhost:8000/api/v1/predictions/SPY?horizon_days=5" \
  -H "Authorization: Bearer $TOKEN"
```

**Run Scenario Simulation:**
```bash
curl -X POST "http://localhost:8000/api/v1/scenarios/" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d '{
    "symbol": "SPY",
    "current_price": 450.0,
    "volatility": 0.15,
    "days": 5
  }'
```

**Get Upcoming Events:**
```bash
curl -X GET "http://localhost:8000/api/v1/events/upcoming?days_ahead=30" \
  -H "Authorization: Bearer $TOKEN"
```

### Step 3: Use Test Script

```bash
chmod +x test_api.sh
./test_api.sh
```

---

## Docker Deployment

### Step 1: Build Docker Image

```bash
docker build -t market-prediction-agent .
```

### Step 2: Run Container

```bash
docker run -p 8000:8000 --env-file .env market-prediction-agent
```

### Step 3: Using Docker Compose

```bash
# Start all services (API + Redis)
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

---

## Running Tests

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/test_data_processor.py

# Run with coverage
pytest --cov=src tests/
```

---

## Running Examples

```bash
# Run basic usage examples
python3 examples/basic_usage.py
```

---

## Troubleshooting

### Issue 1: "pip command not found"

**Solution:**
```bash
# Use python3 -m pip instead
python3 -m pip install -r requirements.txt
```

### Issue 2: "externally-managed-environment" Error

**Solution:**
```bash
# Make sure virtual environment is activated
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Then install
python3 -m pip install -r requirements.txt
```

### Issue 3: "Module not found" Errors

**Solution:**
```bash
# Verify virtual environment is activated
which python3  # Should show venv path

# Reinstall dependencies
python3 -m pip install -r requirements.txt --force-reinstall
```

### Issue 4: "Port 8000 already in use"

**Solution:**
```bash
# Option 1: Use different port
uvicorn src.api.main:app --port 8001

# Option 2: Find and kill process
# On macOS/Linux:
lsof -ti:8000 | xargs kill

# On Windows:
netstat -ano | findstr :8000
taskkill /PID <PID> /F
```

### Issue 5: "BigQuery connection failed"

**Solution:**
- This is OK for testing! API will still run
- Verify credentials:
  ```bash
  # Test BigQuery connection
  python3 -c "from src.data.bigquery_client import BigQueryClient; \
              bq = BigQueryClient(); \
              print('Connected!' if bq.test_connection() else 'Failed')"
  ```

### Issue 6: "Authentication failed" / "401 Unauthorized"

**Solution:**
```bash
# Generate new token
python3 create_test_token.py

# Verify SECRET_KEY in .env matches token generation
```

### Issue 7: Import Errors with Prophet or arch / Failed building wheel for llvmlite

**Solution:**
These packages are optional! The application works without them.

**Quick Fix:**
```bash
# Use minimal requirements instead
python3.12 -m pip install -r requirements-minimal.txt
```

**If you need arch/prophet:**

**On macOS:**
```bash
# Install LLVM
brew install llvm
export LLVM_CONFIG=/opt/homebrew/opt/llvm/bin/llvm-config

# Then install
python3.12 -m pip install arch prophet
```

**On Ubuntu/Debian:**
```bash
sudo apt-get install llvm-dev build-essential
python3.12 -m pip install arch prophet
```

**See `INSTALL_TROUBLESHOOTING.md` for detailed solutions.**

---

## Quick Start Checklist

- [ ] Python 3.11+ installed
- [ ] Virtual environment created and activated
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] `.env` file created and configured
- [ ] `SECRET_KEY` set in `.env`
- [ ] BigQuery configured (optional)
- [ ] Application running (`python3 run.py`)
- [ ] Health check passes (`curl http://localhost:8000/health`)
- [ ] API docs accessible (http://localhost:8000/docs)

---

## Next Steps

1. **Explore API Documentation**: http://localhost:8000/docs
2. **Read Full Documentation**: See `README.md`
3. **Configure BigQuery**: Set up your `market_data` tables
4. **Run Examples**: `python3 examples/basic_usage.py`
5. **Write Tests**: Add tests in `tests/` directory
6. **Deploy**: Use Docker for production deployment

---

## Getting Help

- **API Documentation**: http://localhost:8000/docs (when running)
- **Project README**: `README.md`
- **Quick Start Guide**: `QUICKSTART.md`
- **How to Run**: `HOW_TO_RUN.md`

---

## Production Deployment

For production, consider:

1. **Set secure SECRET_KEY** (use `secrets.token_urlsafe(32)`)
2. **Configure CORS** appropriately in `src/api/main.py`
3. **Set up SSL/TLS** (use reverse proxy like nginx)
4. **Configure logging** (set `LOG_LEVEL=INFO` or `ERROR`)
5. **Use process manager** (systemd, supervisor, or PM2)
6. **Set up monitoring** (Prometheus, Grafana)
7. **Configure rate limiting** (already in config)
8. **Use Redis** for caching (optional but recommended)

---

## Example Production Command

```bash
# Using gunicorn with uvicorn workers
gunicorn src.api.main:app \
  -w 4 \
  -k uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000 \
  --log-level info \
  --access-logfile - \
  --error-logfile -
```

---

**You're all set!** The application should now be running at http://localhost:8000

