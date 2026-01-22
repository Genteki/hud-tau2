"""TAU2-bench Environment - Customer service task evaluation.

This environment provides:
- HTTP-based domain tools (airline, retail, telecom)
- @env.scenario() for task evaluation lifecycle
- Multi-turn conversation support via send_message
"""

import logging
import sys

from hud import Environment
from server.tools.message_log import create_record_message_tool

# Configure Python's standard logging to stderr (MCP uses stdout for communication)
# Only configure if not already configured (don't override test script settings)
if not logging.getLogger().hasHandlers():
    logging.basicConfig(
        stream=sys.stderr,
        level=logging.INFO,  # INFO level to show evaluation results
        format="[%(levelname)s] %(asctime)s | %(name)s | %(message)s",
    )

# Suppress verbose loggers
for logger_name in ["httpx", "httpcore", "LiteLLM", "litellm"]:
    logging.getLogger(logger_name).setLevel(logging.WARNING)

# Suppress "Tool already exists" warnings from HUD's tool manager
logging.getLogger("hud.environment.tool_manager").setLevel(logging.ERROR)

# CRITICAL: Configure tau2's loguru logger (tau2 uses loguru, not standard logging!)
# Set to WARNING level to suppress verbose INFO messages from tau2
from loguru import logger as tau2_logger
tau2_logger.remove()  # Remove default handler
tau2_logger.add(sys.stderr, level="WARNING")

logger = logging.getLogger(__name__)

# Create the environment
env = Environment(name="tau2-bench")

# Global conversation tool reference (needed for setup to bootstrap conversations)
_conversation_tool = None


def get_conversation_tool():
    """Get the globally registered conversation tool."""
    return _conversation_tool


@env.initialize
async def init():
    """Initialize tau2-bench environment with HTTP-based tools."""
    global _conversation_tool

    logger.info("Initializing tau2-bench environment")

    # Mount hubs
    from server.setup import setup
    from server.evaluate import evaluate
    env.mount(setup)
    env.mount(evaluate)
    env.add_tool(create_record_message_tool())
    

    # Register scenarios
    from server.scenarios import register_tau2_scenarios
    register_tau2_scenarios(env)

    logger.info("Initialized tau2-bench - domain tools will be loaded per scenario")


if __name__ == "__main__":
    env.run(transport="stdio")
