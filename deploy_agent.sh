#!/bin/bash
# Deployment script for Vertex AI Agent Engine

set -e

echo "=========================================="
echo "Deploying to Vertex AI Agent Engine"
echo "=========================================="

# Check if adk is installed
if ! command -v adk &> /dev/null; then
    echo "⚠ ADK not found. Installing..."
    pip install google-cloud-aiplatform[adk]
fi

# Check if requirements.txt is deployment-ready
if grep -q "arch==" requirements.txt || grep -q "prophet==" requirements.txt; then
    echo "⚠ Using deployment-ready requirements..."
    cp requirements-deploy.txt requirements.txt
fi

# Set default project if not set
if [ -z "$GOOGLE_CLOUD_PROJECT" ]; then
    export GOOGLE_CLOUD_PROJECT="ccibt-hack25ww7-736"
    echo "✓ Set GOOGLE_CLOUD_PROJECT=$GOOGLE_CLOUD_PROJECT"
fi

if [ -z "$GOOGLE_CLOUD_LOCATION" ]; then
    export GOOGLE_CLOUD_LOCATION="us-central1"
    echo "✓ Set GOOGLE_CLOUD_LOCATION=$GOOGLE_CLOUD_LOCATION"
fi

# Configure gcloud (required for ADK)
gcloud config set project "$GOOGLE_CLOUD_PROJECT" 2>/dev/null || {
    echo "⚠ Warning: gcloud not configured. Please run: gcloud auth login"
}

# Verify agent can be imported
echo "Verifying agent structure..."
python3.12 -c "from agent import app; print('✓ Agent verified')" || {
    echo "✗ Agent verification failed"
    exit 1
}

# Set default project if not set
if [ -z "$GOOGLE_CLOUD_PROJECT" ]; then
    export GOOGLE_CLOUD_PROJECT="ccibt-hack25ww7-736"
    echo "✓ Set GOOGLE_CLOUD_PROJECT=$GOOGLE_CLOUD_PROJECT"
fi

if [ -z "$GOOGLE_CLOUD_LOCATION" ]; then
    export GOOGLE_CLOUD_LOCATION="us-central1"
    echo "✓ Set GOOGLE_CLOUD_LOCATION=$GOOGLE_CLOUD_LOCATION"
fi

# Configure gcloud
gcloud config set project "$GOOGLE_CLOUD_PROJECT" 2>/dev/null || {
    echo "⚠ Warning: gcloud not configured. Please run: gcloud auth login"
}

# Check if agent directory exists
if [ ! -d "mktdata_hackathon" ]; then
    echo "Creating agent directory structure..."
    mkdir -p mktdata_hackathon
    cp agent.py __init__.py requirements.txt mktdata_hackathon/
    cp -r src mktdata_hackathon/
    cp data_set_config.json mktdata_hackathon/ 2>/dev/null || true
    echo "✓ Agent directory created"
fi

# Verify agent in directory
echo "Verifying agent in directory..."
cd mktdata_hackathon
python3.12 -c "from agent import app; print('✓ Agent verified')" || {
    echo "✗ Agent verification failed"
    exit 1
}
cd ..

# Deploy
echo ""
echo "Deploying agent..."
echo "  Project: $GOOGLE_CLOUD_PROJECT"
echo "  Region: $GOOGLE_CLOUD_LOCATION"
echo "  Display name: mktdata_hackathon"
echo "  Agent directory: mktdata_hackathon"
echo ""

adk deploy agent_engine \
  --project="$GOOGLE_CLOUD_PROJECT" \
  --region="$GOOGLE_CLOUD_LOCATION" \
  --display_name="mktdata_hackathon" \
  mktdata_hackathon

echo ""
echo "✓ Deployment initiated!"
echo "Check Vertex AI Console for deployment status."

