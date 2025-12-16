# Correct ADK Deploy Command

## The Issue

ADK doesn't accept `--location` or `--project` as command-line flags. These are set via:
- Environment variables
- gcloud configuration
- API key configuration

## Correct Deployment Command

### Step 1: Set Environment Variables

```bash
export GOOGLE_CLOUD_PROJECT="ccibt-hack25ww7-736"
export GOOGLE_CLOUD_LOCATION="us-central1"
```

### Step 2: Configure gcloud

```bash
gcloud config set project ccibt-hack25ww7-736
```

### Step 3: Deploy (Simple Command)

```bash
adk deploy agent_engine \
  --display_name="mktdata_hackathon" \
  mktdata_hackathon
```

## Full Example

```bash
# Set environment
export GOOGLE_CLOUD_PROJECT="ccibt-hack25ww7-736"
export GOOGLE_CLOUD_LOCATION="us-central1"

# Configure gcloud
gcloud config set project $GOOGLE_CLOUD_PROJECT

# Deploy
adk deploy agent_engine \
  --display_name="mktdata_hackathon" \
  mktdata_hackathon
```

## Using the Script

The updated `deploy_with_env.sh` script now uses the correct syntax:

```bash
./deploy_with_env.sh
```

## Alternative: Set Project in Command

If environment variables aren't working, you can also set the project via gcloud before deploying:

```bash
gcloud config set project ccibt-hack25ww7-736
adk deploy agent_engine --display_name="mktdata_hackathon" mktdata_hackathon
```

## Verify Configuration

Before deploying, verify:

```bash
# Check gcloud project
gcloud config get-value project

# Should output: ccibt-hack25ww7-736

# Check environment
echo $GOOGLE_CLOUD_PROJECT
echo $GOOGLE_CLOUD_LOCATION
```

