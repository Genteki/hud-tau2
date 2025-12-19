#!/bin/bash
# Start the tau2-bench environment server with uvicorn

# Get domain from environment variable or use default
DOMAIN="${DOMAIN:-airline}"
HOST="${TAU2_SERVER_HOST:-127.0.0.1}"
PORT="${TAU2_SERVER_PORT:-8002}"

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Activate virtual environment if it exists
if [ -f "$HOME/.venv/tau2/bin/activate" ]; then
    source "$HOME/.venv/tau2/bin/activate"
fi

# Run the server
cd "$PROJECT_ROOT"
python3 -m environment.run_server --domain "$DOMAIN" --host "$HOST" --port "$PORT"
