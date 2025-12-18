"""HTTP-based MCP tool wrapper for tau2-bench.

Wraps HTTP calls to the environment server as HUD MCP tools.
"""

import logging
from typing import Any
from hud.tools.base import BaseTool
from mcp.types import TextContent
from server.tools.http_client import get_http_client

logger = logging.getLogger(__name__)


class HTTPTool(BaseTool):
    """
    MCP tool that executes via HTTP calls to the environment server.

    This replaces the direct tool execution with HTTP-based execution,
    allowing tools, databases, and environments to run in a separate uvicorn process.
    """

    def __init__(self, tool_name: str, tool_description: str, tool_schema: dict):
        """
        Initialize HTTP-based tool.

        Args:
            tool_name: Name of the tool
            tool_description: Description for MCP
            tool_schema: Parameter schema from environment server
        """
        super().__init__(
            env=None,
            name=tool_name,
            description=tool_description,
        )
        self.tool_schema = tool_schema
        self._http_client = None

        # Create a fake signature to satisfy FastMCP's **kwargs check
        # The actual parameters come from tool_schema
        import inspect
        self.__signature__ = inspect.Signature([])

    @property
    def http_client(self):
        """Lazy-loaded HTTP client."""
        if self._http_client is None:
            self._http_client = get_http_client()
        return self._http_client

    async def __call__(self, **kwargs) -> list[TextContent]:
        """
        Execute tool via HTTP and log to tau2_task message trajectory.

        Args:
            **kwargs: Tool parameters

        Returns:
            List of TextContent with tool execution result
        """
        try:
            # Get tau2_task for message logging
            from server.state import get_tau2_task
            from tau2.data_model.message import AssistantMessage, ToolMessage
            from uuid import uuid4
            import json
            from datetime import datetime

            tau2_task = get_tau2_task()

            # Create tool call object (following tau2-bench ToolCall structure)
            tool_call_id = str(uuid4())
            tool_call_dict = {
                "id": tool_call_id,
                "name": self.name,
                "arguments": kwargs
            }

            # Log AssistantMessage with tool call (agent calling the tool)
            # Note: In tau2-bench, agent messages with tool_calls have content=None
            assistant_msg = AssistantMessage(
                role="assistant",
                content=None,
                tool_calls=[tool_call_dict],
                cost=0.0,  # TODO: Track actual LLM cost if agent made decision
                timestamp=datetime.now().isoformat()
            )
            tau2_task.add_message(assistant_msg)

            # Execute tool via HTTP
            result = self.http_client.execute_tool(self.name, **kwargs)

            # Format result
            result_str = json.dumps(result, indent=2, ensure_ascii=False)

            # Log ToolMessage (environment response)
            tool_msg = ToolMessage(
                id=tool_call_id,
                role="tool",
                content=result_str,
                timestamp=datetime.now().isoformat()
            )
            tau2_task.add_message(tool_msg)

            return [TextContent(type="text", text=result_str)]

        except Exception as e:
            import traceback
            error_msg = f"HTTP tool execution error for '{self.name}': {str(e)}\n\n{traceback.format_exc()}"
            logger.error(error_msg)
            return [TextContent(type="text", text=error_msg)]

    # Don't override mcp - use BaseTool's default implementation
    # But we need to override parameters after the FunctionTool is created
    @property
    def mcp(self):
        """Get FunctionTool with custom parameter schema from environment server."""
        # Use parent's mcp creation
        mcp_tool = super().mcp
        # Override parameters with schema from environment server
        mcp_tool.parameters = self.tool_schema
        return mcp_tool


class HTTPUserTool(BaseTool):
    """
    MCP tool for user tools that executes via HTTP calls to the environment server.
    """

    def __init__(self, tool_name: str, tool_description: str, tool_schema: dict):
        """
        Initialize HTTP-based user tool.

        Args:
            tool_name: Name of the user tool
            tool_description: Description for MCP
            tool_schema: Parameter schema from environment server
        """
        super().__init__(
            env=None,
            name=tool_name,
            description=tool_description,
        )
        self.tool_schema = tool_schema
        self._http_client = None

        # Create a fake signature to satisfy FastMCP's **kwargs check
        # The actual parameters come from tool_schema
        import inspect
        self.__signature__ = inspect.Signature([])

    @property
    def http_client(self):
        """Lazy-loaded HTTP client."""
        if self._http_client is None:
            self._http_client = get_http_client()
        return self._http_client

    async def __call__(self, **kwargs) -> list[TextContent]:
        """
        Execute user tool via HTTP.

        Args:
            **kwargs: Tool parameters

        Returns:
            List of TextContent with tool execution result
        """
        try:
            # Execute user tool via HTTP
            result = self.http_client.execute_user_tool(self.name, **kwargs)

            # Format result as JSON string
            import json
            result_str = json.dumps(result, indent=2, ensure_ascii=False)

            return [TextContent(type="text", text=result_str)]

        except Exception as e:
            import traceback
            error_msg = f"HTTP user tool execution error for '{self.name}': {str(e)}\n\n{traceback.format_exc()}"
            logger.error(error_msg)
            return [TextContent(type="text", text=error_msg)]

    # Don't override mcp - use BaseTool's default implementation
    # But we need to override parameters after the FunctionTool is created
    @property
    def mcp(self):
        """Get FunctionTool with custom parameter schema from environment server."""
        # Use parent's mcp creation
        mcp_tool = super().mcp
        # Override parameters with schema from environment server
        mcp_tool.parameters = self.tool_schema
        return mcp_tool


def create_http_tools_from_server():
    """
    Query the environment server and create HTTP-based MCP tools.

    Returns:
        Dict mapping tool names to HTTPTool instances
    """
    client = get_http_client()

    # Check server health first
    if not client.health_check():
        raise RuntimeError(
            "Environment server is not reachable. Please start it first with:\n"
            "  ./hud-tau2/scripts/start_environment_server.sh"
        )

    # Get tools list from server
    tools_data = client.list_tools()

    http_tools = {}

    # Create HTTPTool instances for regular tools
    for tool_info in tools_data.get("tools", []):
        tool_name = tool_info["name"]
        tool_desc = tool_info["description"]
        tool_schema = tool_info.get("parameters", {"type": "object", "properties": {}})

        http_tools[tool_name] = HTTPTool(
            tool_name=tool_name,
            tool_description=tool_desc,
            tool_schema=tool_schema
        )

    # Create HTTPUserTool instances for user tools
    for tool_info in tools_data.get("user_tools", []):
        tool_name = tool_info["name"]
        tool_desc = tool_info["description"]
        tool_schema = tool_info.get("parameters", {"type": "object", "properties": {}})

        http_tools[tool_name] = HTTPUserTool(
            tool_name=tool_name,
            tool_description=tool_desc,
            tool_schema=tool_schema
        )

    logger.info(f"Created {len(http_tools)} HTTP-based MCP tools from environment server")

    return http_tools
