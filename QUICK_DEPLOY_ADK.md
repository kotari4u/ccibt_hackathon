# Quick Deploy to Vertex AI Agent Engine

## The Correct Command

ADK accepts `--project` and `--region` flags (note: `--region`, not `--location`).

## Step-by-Step Deployment

### Step 1: Set Environment Variables

```bash
export GOOGLE_CLOUD_PROJECT="ccibt-hack25ww7-736"
export GOOGLE_CLOUD_LOCATION="us-central1"
```

### Step 2: Configure gcloud

```bash
gcloud config set project ccibt-hack25ww7-736
```

### Step 3: Deploy (With Project and Region Flags)

```bash
adk deploy agent_engine \
  --project="ccibt-hack25ww7-736" \
  --region="us-central1" \
  --display_name="mktdata_hackathon" \
  mktdata_hackathon
```

## One-Line Deployment

```bash
export GOOGLE_CLOUD_PROJECT="ccibt-hack25ww7-736" && \
export GOOGLE_CLOUD_LOCATION="us-central1" && \
gcloud config set project $GOOGLE_CLOUD_PROJECT && \
adk deploy agent_engine \
  --project="$GOOGLE_CLOUD_PROJECT" \
  --region="$GOOGLE_CLOUD_LOCATION" \
  --display_name="mktdata_hackathon" \
  mktdata_hackathon
```

## Using the Scripts

### Option 1: Automated Script

```bash
./deploy_with_env.sh
```

### Option 2: Setup Then Deploy

```bash
# Setup environment
source setup_adk_env.sh

# Deploy with flags
adk deploy agent_engine \
  --project="$GOOGLE_CLOUD_PROJECT" \
  --region="$GOOGLE_CLOUD_LOCATION" \
  --display_name="mktdata_hackathon" \
  mktdata_hackathon
```

## Important Notes

- ✅ **Use**: `--project` and `--region` flags (required!)
- ⚠️ **Note**: Use `--region`, NOT `--location`
- ✅ **Also works**: Environment variables (`GOOGLE_CLOUD_PROJECT`, `GOOGLE_CLOUD_LOCATION`) in .env file
- ✅ **Do use**: `gcloud config set project` for authentication
- ✅ **Required**: `--display_name` flag for the display name

## Verify Before Deploying

```bash
# Check environment
echo "Project: $GOOGLE_CLOUD_PROJECT"
echo "Location: $GOOGLE_CLOUD_LOCATION"

# Check gcloud
gcloud config get-value project

# Verify agent directory
ls mktdata_hackathon/
```

