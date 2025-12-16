#!/bin/bash
# Setup environment variables for ADK deployment

echo "Setting up ADK deployment environment..."
echo "=========================================="

# Set project ID
export GOOGLE_CLOUD_PROJECT="ccibt-hack25ww7-736"
echo "✓ GOOGLE_CLOUD_PROJECT=$GOOGLE_CLOUD_PROJECT"

# Set region (default to us-central1)
export GOOGLE_CLOUD_LOCATION="${GOOGLE_CLOUD_LOCATION:-us-central1}"
echo "✓ GOOGLE_CLOUD_LOCATION=$GOOGLE_CLOUD_LOCATION"

# Set staging bucket (create if needed)
# Replace with your actual bucket name
export STAGING_BUCKET="${STAGING_BUCKET:-gs://${GOOGLE_CLOUD_PROJECT}-staging}"
echo "✓ STAGING_BUCKET=$STAGING_BUCKET"

# Configure gcloud
echo ""
echo "Configuring gcloud..."
gcloud config set project $GOOGLE_CLOUD_PROJECT
gcloud config set compute/region $GOOGLE_CLOUD_LOCATION

echo ""
echo "Environment variables set:"
echo "  GOOGLE_CLOUD_PROJECT=$GOOGLE_CLOUD_PROJECT"
echo "  GOOGLE_CLOUD_LOCATION=$GOOGLE_CLOUD_LOCATION"
echo "  STAGING_BUCKET=$STAGING_BUCKET"
echo ""
echo "To persist these, add them to your ~/.bashrc or ~/.zshrc:"
echo "  export GOOGLE_CLOUD_PROJECT=\"$GOOGLE_CLOUD_PROJECT\""
echo "  export GOOGLE_CLOUD_LOCATION=\"$GOOGLE_CLOUD_LOCATION\""
echo "  export STAGING_BUCKET=\"$STAGING_BUCKET\""

