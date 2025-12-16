# ADK Deployment Guide for Vertex AI Agent Engine

## Prerequisites

1. **Install ADK** (if not already installed):
   ```bash
   pip install google-cloud-aiplatform[adk]
   ```

2. **Set up Google Cloud credentials**:
   ```bash
   gcloud auth login
   gcloud config set project YOUR_PROJECT_ID
   ```

3. **Set environment variables**:
   ```bash
   export GOOGLE_CLOUD_PROJECT="your-gcp-project-id"
   export GOOGLE_CLOUD_LOCATION="us-central1"  # or your preferred region
   export STAGING_BUCKET="gs://your-staging-bucket-name"
   ```

## Deployment Steps

### Step 1: Prepare Requirements

The deployment-ready requirements are already set:
```bash
# Verify requirements.txt is deployment-ready
cat requirements.txt | head -20
```

### Step 2: Verify Agent Structure

```bash
# Test that agent.py can be imported
python3.12 -c "from agent import app; print('✓ Success')"
```

### Step 3: Deploy to Agent Engine

```bash
# Deploy with display name
adk deploy agent_engine \
  --display_name="mktdata_hackathon" \
  --staging_bucket=$STAGING_BUCKET \
  mktdata_hackathon
```

Or if you have the staging bucket set:
```bash
adk deploy agent_engine \
  --display_name="mktdata_hackathon" \
  mktdata_hackathon
```

## Required Files

The following files are required for ADK deployment:

- ✅ `agent.py` - Agent entry point (exports FastAPI app)
- ✅ `__init__.py` - Package marker
- ✅ `requirements.txt` - Deployment-ready dependencies
- ✅ `src/` - Source code directory
- ✅ `.gcloudignore` - Files to exclude from deployment

## Configuration

### Environment Variables

Set these in Vertex AI Agent Engine console or via ADK:

```bash
BIGQUERY_PROJECT_ID=your-project-id
BIGQUERY_DATASET=market_data
SECRET_KEY=your-secret-key
LOG_LEVEL=INFO
LOG_FORMAT=json
```

### Entry Point

The agent uses:
- **Entry point**: `agent:app` (FastAPI application)
- **Python version**: 3.12 (or 3.11+)

## Troubleshooting

### Error: "adk: command not found"

Install ADK:
```bash
pip install google-cloud-aiplatform[adk]
```

### Error: "Build failed"

1. Use `requirements-deploy.txt`:
   ```bash
   cp requirements-deploy.txt requirements.txt
   ```

2. Verify imports:
   ```bash
   python3.12 verify_deployment.py
   ```

3. Check for syntax errors:
   ```bash
   python3.12 -m py_compile agent.py
   ```

### Error: "Cannot import agent"

Verify the agent structure:
```bash
python3.12 -c "from agent import app; print('Success')"
```

## Post-Deployment

After successful deployment:

1. **Access the agent** via Vertex AI Console
2. **Set environment variables** in the agent configuration
3. **Test endpoints** using the agent's API URL
4. **Monitor logs** in Cloud Logging

## Quick Deploy Command

```bash
# Full command with all options
adk deploy agent_engine \
  --display_name="mktdata_hackathon" \
  --staging_bucket="gs://your-bucket-name" \
  --project="your-project-id" \
  --location="us-central1" \
  mktdata_hackathon
```

