#!/bin/bash

# Test API endpoints script

BASE_URL="http://localhost:8000"

echo "Testing Market Activity Prediction Agent API"
echo "============================================"
echo ""

# Test 1: Health check
echo "1. Testing health endpoint..."
curl -s "$BASE_URL/health" | python3 -m json.tool
echo ""

# Test 2: Root endpoint
echo "2. Testing root endpoint..."
curl -s "$BASE_URL/" | python3 -m json.tool
echo ""

# Test 3: Get token (if create_test_token.py exists)
if [ -f "create_test_token.py" ]; then
    echo "3. Generating test token..."
    TOKEN=$(python3 create_test_token.py 2>/dev/null | grep -A 1 "Token:" | tail -1 | tr -d ' ')
    if [ ! -z "$TOKEN" ]; then
        echo "Token generated: ${TOKEN:0:50}..."
        echo ""
        echo "4. Testing prediction endpoint (requires BigQuery)..."
        curl -s -X GET "$BASE_URL/api/v1/predictions/SPY?horizon_days=5" \
            -H "Authorization: Bearer $TOKEN" | python3 -m json.tool || echo "Note: This requires BigQuery configuration"
    fi
fi

echo ""
echo "============================================"
echo "API Documentation: $BASE_URL/docs"
echo "============================================"

