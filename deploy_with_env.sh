#!/bin/bash
# Deploy with environment setup

set -e

echo "=========================================="
echo "Deploying to Vertex AI Agent Engine"
echo "=========================================="

# Source environment setup
source setup_adk_env.sh

# Check if adk is installed
if ! command -v adk &> /dev/null; then
    echo "⚠ ADK not found. Installing..."
    pip install google-cloud-aiplatform[adk]
fi

# Verify agent directory exists
if [ ! -d "mktdata_hackathon" ]; then
    echo "Creating agent directory..."
    mkdir -p mktdata_hackathon
    cp agent.py __init__.py requirements-deploy.txt mktdata_hackathon/requirements.txt
    cp -r src mktdata_hackathon/
    cp data_set_config.json mktdata_hackathon/ 2>/dev/null || true
fi

# Verify agent
echo "Verifying agent..."
cd mktdata_hackathon
python3.12 -c "from agent import app; print('✓ Agent verified')" || {
    echo "✗ Agent verification failed"
    exit 1
}
cd ..

# Set project via gcloud config (for authentication)
echo "Configuring gcloud..."
gcloud config set project "$GOOGLE_CLOUD_PROJECT" 2>/dev/null || {
    echo "⚠ Warning: gcloud not configured. Please run: gcloud auth login"
}

# Deploy
echo ""
echo "Deploying agent..."
echo "  Project: $GOOGLE_CLOUD_PROJECT"
echo "  Region: $GOOGLE_CLOUD_LOCATION"
echo "  Display name: mktdata_hackathon"
echo ""

# ADK requires --project and --region flags (not --location!)
adk deploy agent_engine \
  --project="$GOOGLE_CLOUD_PROJECT" \
  --region="$GOOGLE_CLOUD_LOCATION" \
  --display_name="mktdata_hackathon" \
  mktdata_hackathon

echo ""
echo "✓ Deployment initiated!"

