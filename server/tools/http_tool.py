"""HTTP-based MCP tool wrapper for tau2-bench.

Wraps HTTP calls to the environment server as HUD MCP tools.
"""

import logging
from typing import Any, Dict
from hud.tools.base import BaseTool
from mcp.types import TextContent
from server.tools.http_client import get_http_client

logger = logging.getLogger(__name__)

# Global registry of HTTP tools (for dynamic reloading per domain)
_http_tool_registry: Dict[str, BaseTool] = {}


def get_http_tool_registry() -> Dict[str, BaseTool]:
    """Get the global HTTP tool registry."""
    return _http_tool_registry


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
            from tau2.environment.environment import Environment
            from uuid import uuid4
            from datetime import datetime

            tau2_task = get_tau2_task()

            # Create tool call object (following tau2-bench ToolCall structure)
            from tau2.data_model.message import ToolCall
            tool_call = ToolCall(
                id=str(uuid4()),
                name=self.name,
                arguments=kwargs,
                requestor="assistant",
            )

            # Log AssistantMessage with tool call (agent calling the tool)
            assistant_msg = AssistantMessage(
                role="assistant",
                tool_calls=[tool_call],
                cost=0.0,
            )
            tau2_task.add_message(assistant_msg)

            # Execute tool via HTTP
            error = False
            try:
                result = self.http_client.execute_tool(self.name, **kwargs)
                # Server already applied Environment.to_json_str, just serialize to JSON string
                import json
                if isinstance(result, str):
                    # String results don't need further serialization
                    result_str = result
                else:
                    # Dict/list results need JSON encoding
                    result_str = json.dumps(result, ensure_ascii=False)

                # Log tool execution result during runtime
                logger.debug(f"[RUNTIME] Tool '{self.name}' with args {kwargs} returned: {result_str[:200]}")
            except Exception as e:
                result_str = f"Error: {e}"
                error = True
                logger.error(f"[RUNTIME] Tool '{self.name}' failed: {e}")

            # Log ToolMessage (environment response) - matches reference implementation
            tool_msg = ToolMessage(
                id=tool_call.id,
                role="tool",
                content=result_str,
                requestor=tool_call.requestor,
                error=error,
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


def create_http_tools_from_server(max_retries=30, retry_delay=1.0):
    """
    Query the environment server and create HTTP-based MCP tools.

    Args:
        max_retries: Maximum number of connection attempts
        retry_delay: Delay between retries in seconds

    Returns:
        Dict mapping tool names to HTTPTool instances
    """
    import time

    client = get_http_client()

    # Check server health with retries (for Docker container startup)
    for attempt in range(max_retries):
        if client.health_check():
            break

        if attempt < max_retries - 1:
            logger.info(f"Environment server not ready, retrying in {retry_delay}s... (attempt {attempt + 1}/{max_retries})")
            time.sleep(retry_delay)
        else:
            raise RuntimeError(
                f"Environment server is not reachable after {max_retries} attempts. "
                "Please ensure the environment server is running."
            )

    # Get tools list from server
    tools_data = client.list_tools()

    http_tools = {}

    # Create HTTPTool instances for regular tools
    # Skip send_message - we use turn-based conversation instead
    for tool_info in tools_data.get("tools", []):
        tool_name = tool_info["name"]

        # Filter out send_message tool
        if tool_name == "send_message":
            logger.info("Skipping send_message tool (using turn-based conversation)")
            continue

        tool_desc = tool_info["description"]
        tool_schema = tool_info.get("parameters", {"type": "object", "properties": {}})

        http_tools[tool_name] = HTTPTool(
            tool_name=tool_name,
            tool_description=tool_desc,
            tool_schema=tool_schema
        )

    # Don't create HTTPUserTool instances - user tools are only for UserSimulator
    # User tools are fetched separately in ConversationTool.initialize_global()
    # and passed to the UserSimulator, not exposed to the agent

    logger.info(f"Created {len(http_tools)} HTTP-based MCP tools from environment server")

    # Update global registry
    global _http_tool_registry
    _http_tool_registry.update(http_tools)

    return http_tools
