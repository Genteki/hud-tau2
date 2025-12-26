FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Clone and install tau2-bench
RUN git clone https://github.com/sierra-research/tau2-bench.git
RUN pip install -e tau2-bench

# Install project dependencies
COPY pyproject.toml ./
RUN pip install .

# Copy source code
COPY server/ ./server/
COPY task/ ./task/
COPY environment/ ./environment/

# Environment configuration
ENV TAU2_DATA_DIR=/app/tau2-bench/data
ENV DOMAIN=airline
ENV BACKEND_PORT=8002
ENV TAU2_SERVER_URL=http://localhost:8002

# Start environment server in background, then run MCP server
# CRITICAL: Environment server logs MUST go to stderr, not stdout (MCP uses stdout for JSONRPC)
# Note: MCP server now has retry logic to wait for environment server, so minimal sleep is fine
CMD ["sh", "-c", "python -m environment.run_server --domain ${DOMAIN} --host 0.0.0.0 --port ${BACKEND_PORT} >&2 & sleep 1 && python -m server.main"]
