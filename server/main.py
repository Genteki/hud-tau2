import sys
import os
import logging
from hud.server import MCPServer

logger = logging.getLogger(__name__)

mcp = MCPServer(name="tau2-bench")
from .setup import setup
from .evaluate import evaluate

# Global conversation tool reference (needed for setup to bootstrap conversations)
_conversation_tool = None

def get_conversation_tool():
    """Get the globally registered conversation tool."""
    return _conversation_tool


@mcp.initialize
async def init():
    """Initialize tau2-bench MCP server with HTTP-based tools."""
    global _conversation_tool

    logger.info("Initializing tau2-bench MCP server")

    # Mount hubs
    mcp.mount(setup)
    mcp.mount(evaluate)

    # Register conversation tool for multi-turn mode
    from .tools.conversation import create_conversation_tool
    _conversation_tool = create_conversation_tool()
    # Register the tool object (callable), not its FunctionTool wrapper.
    mcp.add_tool(_conversation_tool)

    # Load HTTP-based tools from environment server
    from .tools.http_tool import create_http_tools_from_server

    try:
        http_tools = create_http_tools_from_server()

        # Register all HTTP-based domain tools (callables)
        for tool_name, http_tool in http_tools.items():
            mcp.add_tool(http_tool)

        logger.info(f"Initialized with {len(http_tools)} HTTP-based domain tools + send_message")
    except RuntimeError as e:
        logger.error(f"Failed to initialize HTTP tools: {e}")
        logger.warning("MCP server will start without domain tools. Start environment server first.")


if __name__ == "__main__":
    mcp.run(transport="stdio")
