#!/bin/bash
# Test script to verify both servers start correctly

set -e

echo "=== Testing Environment Server Startup ==="
export DOMAIN=airline
export TAU2_SERVER_PORT=8002

# Start environment server in background
echo "Starting environment server..."
python -m environment.run_server --domain $DOMAIN --host 127.0.0.1 --port $TAU2_SERVER_PORT &
ENV_PID=$!

# Wait for server to start with retry logic
echo "Waiting for server to be ready..."
MAX_RETRIES=30
RETRY_COUNT=0
SERVER_READY=false

while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    if curl -f http://localhost:8002/health 2>/dev/null >/dev/null; then
        SERVER_READY=true
        break
    fi
    RETRY_COUNT=$((RETRY_COUNT + 1))
    sleep 1
done

# Test health endpoint
echo "Testing health endpoint..."
if [ "$SERVER_READY" = true ]; then
    echo "✓ Environment server is healthy"
else
    echo "✗ Environment server health check failed (timeout after ${MAX_RETRIES}s)"
    kill $ENV_PID 2>/dev/null || true
    exit 1
fi

# Test tools endpoint
echo "Testing tools endpoint..."
if curl -f http://localhost:8002/tools 2>/dev/null | grep -q "tools"; then
    echo "✓ Tools endpoint working"
else
    echo "✗ Tools endpoint failed"
    kill $ENV_PID 2>/dev/null || true
    exit 1
fi

# Cleanup
echo "Cleaning up..."
kill $ENV_PID 2>/dev/null || true
wait $ENV_PID 2>/dev/null || true

echo ""
echo "=== All tests passed! ==="
