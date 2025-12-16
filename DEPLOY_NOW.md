# Deploy Now - Quick Reference

## The Correct Command

ADK requires `--project` and `--region` flags (note: `--region`, not `--location`).

## Quick Deploy Command

```bash
adk deploy agent_engine \
  --project="ccibt-hack25ww7-736" \
  --region="us-central1" \
  --display_name="mktdata_hackathon" \
  mktdata_hackathon
```

## Or Use the Script

```bash
./deploy_agent.sh
```

The script will:
1. Set environment variables automatically
2. Configure gcloud
3. Deploy with the correct flags

## What Was Wrong

- ❌ `--location` flag doesn't exist (use `--region`)
- ❌ Missing `--project` flag
- ✅ ADK help shows: `--project` and `--region` are valid options

## Verify Before Deploying

```bash
# Check gcloud is authenticated
gcloud auth list

# Check project is set
gcloud config get-value project

# Should show: ccibt-hack25ww7-736
```

## If You Get Authentication Errors

```bash
# Login to gcloud
gcloud auth login

# Set application default credentials
gcloud auth application-default login
```
