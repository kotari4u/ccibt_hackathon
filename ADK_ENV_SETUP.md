# ADK Environment Setup

## Quick Setup

Run the setup script to configure environment variables:

```bash
source setup_adk_env.sh
```

Or set them manually:

```bash
export GOOGLE_CLOUD_PROJECT="ccibt-hack25ww7-736"
export GOOGLE_CLOUD_LOCATION="us-central1"
export STAGING_BUCKET="gs://ccibt-hack25ww7-736-staging"  # Create if needed
```

## Deploy with Environment

### Option 1: Use the deployment script (recommended)

```bash
./deploy_with_env.sh
```

This script:
- Sets up environment variables
- Verifies agent structure
- Deploys with proper configuration

### Option 2: Manual deployment

```bash
# 1. Set environment variables
source setup_adk_env.sh

# 2. Deploy
adk deploy agent_engine \
  --display_name="mktdata_hackathon" \
  --project="$GOOGLE_CLOUD_PROJECT" \
  --location="$GOOGLE_CLOUD_LOCATION" \
  mktdata_hackathon
```

### Option 3: Inline environment variables

```bash
GOOGLE_CLOUD_PROJECT="ccibt-hack25ww7-736" \
GOOGLE_CLOUD_LOCATION="us-central1" \
adk deploy agent_engine \
  --display_name="mktdata_hackathon" \
  --project="ccibt-hack25ww7-736" \
  --location="us-central1" \
  mktdata_hackathon
```

## Create Staging Bucket (if needed)

```bash
gsutil mb -p ccibt-hack25ww7-736 -l us-central1 gs://ccibt-hack25ww7-736-staging
```

## Verify Configuration

```bash
# Check gcloud config
gcloud config get-value project
gcloud config get-value compute/region

# Check environment variables
echo $GOOGLE_CLOUD_PROJECT
echo $GOOGLE_CLOUD_LOCATION
```

## Troubleshooting

**Error: "No project/region or api_key provided"**
- Run `source setup_adk_env.sh` first
- Or use `--project` and `--location` flags explicitly
- Or set `GOOGLE_APPLICATION_CREDENTIALS` to your service account key

**Error: "Permission denied"**
- Ensure you're authenticated: `gcloud auth login`
- Check project permissions
- Verify service account has necessary roles

