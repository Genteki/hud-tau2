"""TAU2-bench Environment - Customer service task evaluation.

This environment provides:
- HTTP-based domain tools (airline, retail, telecom)
- @env.scenario() for task evaluation lifecycle
- Multi-turn conversation support via send_message
"""

import logging
import sys

from hud import Environment

# Configure logging to stderr (MCP uses stdout for communication)
logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="[%(levelname)s] %(asctime)s | %(name)s | %(message)s",
    force=True,
)
for logger_name in ["httpx", "httpcore"]:
    logging.getLogger(logger_name).setLevel(logging.WARNING)

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

    # Register scenarios
    from server.scenarios import register_tau2_scenarios
    register_tau2_scenarios(env)

    # Register conversation tool for multi-turn mode
    from server.tools.conversation_new import create_conversation_tool
    _conversation_tool = create_conversation_tool()
    env.add_tool(_conversation_tool)

    # Don't load HTTP tools at initialization - they will be loaded dynamically
    # when a scenario is set up (because the domain changes per scenario)
    logger.info("Initialized tau2-bench - domain tools will be loaded per scenario")


if __name__ == "__main__":
    env.run(transport="stdio")
