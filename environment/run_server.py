"""Run the tau2-bench environment server.

This script starts a uvicorn server that exposes domain tools as HTTP endpoints.
All environments run in multi-turn mode with user simulator.
"""

import argparse
import logging
import sys
import os

# Import tau2-bench registry
try:
    from tau2.registry import registry
except ImportError:
    # Try adding tau2-bench to path
    tau2_bench_path = os.path.join(os.path.dirname(__file__), '../../tau2-bench/src')
    if os.path.exists(tau2_bench_path):
        sys.path.insert(0, tau2_bench_path)
    from tau2.registry import registry

# Import our custom EnvironmentServer (not tau2-bench's)
from environment.server import EnvironmentServer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    # Strip quotes from API key if present (common issue with env var configuration)
    if "OPENAI_API_KEY" in os.environ:
        api_key = os.environ["OPENAI_API_KEY"]
        # Remove surrounding quotes (single or double)
        cleaned_key = api_key.strip().strip('"').strip("'")
        if cleaned_key != api_key:
            logger.info("Cleaned quotes from OPENAI_API_KEY")
            os.environ["OPENAI_API_KEY"] = cleaned_key

    parser = argparse.ArgumentParser(description="Start tau2-bench environment server")
    parser.add_argument(
        "--domain",
        type=str,
        default="airline",
        help="Domain to load (airline, telecom, retail, mock)"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host to bind to"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8002,
        help="Port to bind to"
    )

    args = parser.parse_args()

    logger.info(f"Loading domain: {args.domain}")

    # Get environment constructor from registry - always multi-turn mode
    env_constructor = registry.get_env_constructor(args.domain)
    environment = env_constructor(solo_mode=False)

    logger.info(f"Starting environment server for domain '{args.domain}' on {args.host}:{args.port}")
    tool_count = len(environment.tools.tools) if hasattr(environment.tools, 'tools') else 0
    logger.info(f"Tools available: {tool_count}")
    if environment.user_tools:
        user_tool_count = len(environment.user_tools.tools) if hasattr(environment.user_tools, 'tools') else 0
        logger.info(f"User tools available: {user_tool_count}")

    # Create and run server
    server = EnvironmentServer(environment)
    server.run(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
