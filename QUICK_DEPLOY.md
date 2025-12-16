# Quick Deploy to Vertex AI Agent Engine

## Prerequisites Check

```bash
# 1. Install ADK (if not installed)
pip install google-cloud-aiplatform[adk]

# 2. Set environment variables
export GOOGLE_CLOUD_PROJECT="ccibt-hack25ww7-736"  # or your project ID
export GOOGLE_CLOUD_LOCATION="us-central1"
export STAGING_BUCKET="gs://your-staging-bucket"  # Create if needed

# 3. Authenticate
gcloud auth login
gcloud config set project $GOOGLE_CLOUD_PROJECT
```

## Quick Deploy

### Option 1: Using the deployment script

```bash
./deploy_agent.sh
```

### Option 2: Direct ADK command

```bash
adk deploy agent_engine \
  --display_name="mktdata_hackathon" \
  mktdata_hackathon
```

### Option 3: With staging bucket

```bash
adk deploy agent_engine \
  --display_name="mktdata_hackathon" \
  --staging_bucket=$STAGING_BUCKET \
  mktdata_hackathon
```

## What Was Prepared

✅ **agent.py** - Agent entry point (exports FastAPI app)
✅ **__init__.py** - Package marker
✅ **requirements.txt** - Deployment-ready (using requirements-deploy.txt)
✅ **All source code** - Ready for deployment

## After Deployment

1. **Set environment variables** in Vertex AI Console:
   - `BIGQUERY_PROJECT_ID=ccibt-hack25ww7-736`
   - `BIGQUERY_DATASET=market_data`
   - `SECRET_KEY=your-secret-key`
   - `LOG_LEVEL=INFO`

2. **Access your agent** via Vertex AI Console

3. **Test the agent** using the provided API endpoints

## Troubleshooting

If deployment fails:
1. Check build logs in Vertex AI Console
2. Verify `requirements.txt` uses deployment-ready version
3. Run `python3.12 verify_deployment.py` to check for issues
4. Ensure all environment variables are set

