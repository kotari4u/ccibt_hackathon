# Quick Reference - Market Activity Prediction Agent

## 🚀 Quick Start (3 Commands)

```bash
# 1. Create and activate virtual environment
python3 -m venv venv && source venv/bin/activate

# 2. Install dependencies
python3 -m pip install -r requirements.txt

# 3. Run the application
python3 run.py
```

## 📋 Essential Commands

### Setup
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
python3 -m pip install -r requirements.txt

# Create .env file
cp .env.example .env
# Edit .env with your settings
```

### Running
```bash
# Method 1: Run script
python3 run.py

# Method 2: Direct uvicorn
uvicorn src.api.main:app --reload

# Method 3: Docker
docker-compose up
```

### Testing
```bash
# Health check
curl http://localhost:8000/health

# Generate test token
python3 create_test_token.py

# Run tests
pytest
```

## 🔧 Configuration

### Minimum .env Settings
```bash
SECRET_KEY=your-secret-key-here
BIGQUERY_PROJECT_ID=your-project-id
BIGQUERY_DATASET=market_data
```

### Generate Secret Key
```bash
python3 -c "import secrets; print(secrets.token_urlsafe(32))"
```

## 🌐 API Endpoints

- **Health**: `GET http://localhost:8000/health`
- **Docs**: `http://localhost:8000/docs`
- **Predictions**: `GET /api/v1/predictions/{symbol}`
- **Scenarios**: `POST /api/v1/scenarios/`
- **Events**: `GET /api/v1/events/upcoming`
- **Patterns**: `GET /api/v1/patterns/detect`

## 🐛 Common Issues

| Issue | Solution |
|-------|----------|
| `pip command not found` | Use `python3 -m pip` |
| `externally-managed-environment` | Activate venv first |
| `Port 8000 in use` | Use `--port 8001` |
| `Module not found` | Reinstall: `pip install -r requirements.txt` |
| `BigQuery failed` | OK for testing, API still runs |

## 📚 Documentation

- **Full Instructions**: `RUN_INSTRUCTIONS.md`
- **Quick Start**: `QUICKSTART.md`
- **How to Run**: `HOW_TO_RUN.md`
- **API Docs**: http://localhost:8000/docs (when running)

## ✅ Verification Checklist

```bash
# 1. Check Python version
python3 --version  # Should be 3.11+

# 2. Verify venv activated
which python3  # Should show venv path

# 3. Test imports
python3 -c "import fastapi, uvicorn; print('OK')"

# 4. Check .env exists
ls -la .env

# 5. Start server
python3 run.py

# 6. Test health
curl http://localhost:8000/health
```

## 🐳 Docker Quick Commands

```bash
# Build
docker build -t market-prediction-agent .

# Run
docker run -p 8000:8000 --env-file .env market-prediction-agent

# Compose
docker-compose up -d
docker-compose logs -f
docker-compose down
```

## 📞 Need Help?

1. Check `RUN_INSTRUCTIONS.md` for detailed steps
2. View API docs at http://localhost:8000/docs
3. Check logs in terminal output
4. Verify `.env` configuration

