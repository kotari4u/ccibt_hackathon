# Quick Fix for Vertex AI Agent Engine Deployment

## The Problem

Build failures are usually caused by:
1. **Problematic packages** that require compilation (arch, prophet, shap, pmdarima)
2. **Version conflicts** between packages
3. **Missing dependencies** or incompatible versions

## Quick Solution

### Step 1: Use Deployment-Ready Requirements

Replace your `requirements.txt` with the deployment-ready version:

```bash
# Backup current requirements
cp requirements.txt requirements-full.txt

# Use deployment-ready requirements
cp requirements-deploy.txt requirements.txt
```

### Step 2: Verify Before Deploying

```bash
# Run verification script
python3.12 verify_deployment.py
```

### Step 3: Test Locally First

```bash
# Install deployment requirements
pip install -r requirements-deploy.txt

# Test that app can start
python3.12 -c "from src.api.main import app; print('✓ Success')"
```

### Step 4: Deploy

Use `requirements.txt` (now pointing to deployment version) in Vertex AI Agent Engine.

## What's Different in requirements-deploy.txt?

**Removed (problematic packages):**
- ❌ `arch` - Requires llvmlite compilation
- ❌ `prophet` - Requires Stan compilation  
- ❌ `pmdarima` - Depends on arch
- ❌ `shap` - Large package, can timeout
- ❌ `redis`, `kafka-python`, `twilio` - Optional, commented out

**Kept (essential packages):**
- ✅ FastAPI, uvicorn
- ✅ scikit-learn, xgboost, statsmodels
- ✅ pandas, numpy, scipy
- ✅ BigQuery client
- ✅ All core functionality

**Note**: The application works fine without the removed packages. They're optional features.

## Alternative: Minimal Requirements

If deployment still fails, try the absolute minimum:

```bash
cp requirements-minimal.txt requirements.txt
```

## Environment Variables

Set these in Vertex AI Agent Engine:

```
BIGQUERY_PROJECT_ID=your-project-id
BIGQUERY_DATASET=market_data
SECRET_KEY=your-secret-key
LOG_LEVEL=INFO
```

## Entry Point

Make sure Vertex AI is configured with:
- **Entry point**: `src.api.main:app`
- **Python version**: 3.12 (or 3.11+)

## Still Failing?

1. **Check build logs** in Vertex AI Console for specific error
2. **Try minimal requirements**: `cp requirements-minimal.txt requirements.txt`
3. **Remove optional packages** from requirements-deploy.txt (redis, kafka, twilio)
4. **Check syntax**: `python3.12 verify_deployment.py`

