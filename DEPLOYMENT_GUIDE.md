# Vertex AI Agent Engine Deployment Guide

## Common Build Failures and Solutions

### Issue 1: Build Failed - Requirements/Dependencies

**Problem**: Some packages in `requirements.txt` may cause build failures in Vertex AI Agent Engine.

**Solution**: Use `requirements-deploy.txt` instead, which excludes problematic packages:

```bash
# Copy deployment-ready requirements
cp requirements-deploy.txt requirements.txt
```

**Problematic packages excluded:**
- `arch` - Requires llvmlite compilation
- `prophet` - Requires Stan compilation
- `pmdarima` - Depends on arch
- `shap` - Large package, can timeout

**Note**: The application works without these packages. They are optional features.

### Issue 2: Import Errors

**Check for syntax errors:**
```bash
python3.12 -m py_compile src/api/main.py
python3.12 -m py_compile src/api/routes/*.py
```

### Issue 3: Missing Entry Point

Vertex AI Agent Engine needs a clear entry point. Ensure:
- `src/api/main.py` has `app = FastAPI(...)`
- Application can be imported: `from src.api.main import app`

### Issue 4: Version Conflicts

**Solution**: Use flexible version constraints in `requirements-deploy.txt`:
- `>=` instead of `==` for better compatibility
- Removed exact version pins that might conflict

## Deployment Steps

### Step 1: Prepare Requirements

```bash
# Use deployment-ready requirements
cp requirements-deploy.txt requirements.txt
```

### Step 2: Verify Code

```bash
# Check for syntax errors
find src -name "*.py" -exec python3.12 -m py_compile {} \;

# Test imports
python3.12 -c "from src.api.main import app; print('✓ Import successful')"
```

### Step 3: Test Locally First

```bash
# Install dependencies
pip install -r requirements-deploy.txt

# Run locally
python3.12 run.py

# Test endpoints
curl http://localhost:8000/health
```

### Step 4: Deploy to Vertex AI

1. **Upload code to Cloud Storage** (if needed)
2. **Use requirements-deploy.txt** in deployment
3. **Set entry point**: `src.api.main:app`
4. **Environment variables**: Set via Vertex AI console or config

## Required Environment Variables

Set these in Vertex AI Agent Engine configuration:

```bash
BIGQUERY_PROJECT_ID=your-project-id
BIGQUERY_DATASET=market_data
SECRET_KEY=your-secret-key
LOG_LEVEL=INFO
```

## Minimal Deployment Requirements

For a minimal deployment that's guaranteed to work:

1. **Use `requirements-deploy.txt`**
2. **Ensure `.env` is not required** (use environment variables instead)
3. **Test imports before deploying:**
   ```bash
   python3.12 -c "
   from src.api.main import app
   from src.utils.config import settings
   print('✓ All imports successful')
   "
   ```

## Troubleshooting

### Check Build Logs

In Vertex AI Console:
1. Go to Agent Engine → Your Agent
2. Check "Build Logs" tab
3. Look for specific error messages

### Common Errors

**Error: "Cannot import module"**
- Check that all imports are correct
- Verify `src/__init__.py` exists
- Ensure relative imports use `src.` prefix

**Error: "Package installation failed"**
- Use `requirements-deploy.txt`
- Remove problematic packages
- Check package versions are compatible

**Error: "Syntax error"**
- Run `python3.12 -m py_compile` on all files
- Check for missing commas, parentheses, etc.

## Testing Deployment Locally

Before deploying, test with the same requirements:

```bash
# Create clean virtual environment
python3.12 -m venv venv_deploy
source venv_deploy/bin/activate

# Install deployment requirements
pip install -r requirements-deploy.txt

# Test application
python3.12 -c "from src.api.main import app; print('Success')"
python3.12 run.py
```

## Alternative: Minimal Requirements

If deployment still fails, try `requirements-minimal.txt`:

```bash
cp requirements-minimal.txt requirements.txt
```

This has the absolute minimum dependencies needed to run.

