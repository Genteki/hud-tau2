"""UserSimulator initialization for tau2-bench multi-turn conversations.

This module initializes the UserSimulator that's used by the conversation loop
in hud.agents.base._run_conversation_loop().
"""

import os
import logging
from typing import Optional
from tau2.user.user_simulator import UserSimulator

logger = logging.getLogger(__name__)


def execute_user_tool_via_http(tool_call):
    """
    Execute a user tool call via HTTP to the environment server.

    Args:
        tool_call: ToolCall object from user simulator

    Returns:
        ToolMessage with the tool execution result
    """
    import json
    from tau2.data_model.message import ToolMessage
    from server.tools.http_client import get_http_client

    http_client = get_http_client()

    try:
        # tau2-bench ToolCall has .name and .arguments (not .function.name)
        tool_name = tool_call.name
        tool_args = tool_call.arguments if isinstance(tool_call.arguments, dict) else json.loads(tool_call.arguments)

        # Execute user tool via HTTP
        result = http_client.execute_user_tool(
            tool_name=tool_name,
            **tool_args
        )

        result_str = json.dumps(result, ensure_ascii=False)

        # Log user tool execution result during runtime
        # logger.debug(f"[RUNTIME-USER] Tool '{tool_name}' with args {tool_args} returned: {result_str[:200]}")

        # Create tau2-bench ToolMessage
        return ToolMessage(
            id=tool_call.id,
            role="tool",
            content=result_str,
            requestor=tool_call.requestor,
            error=False
        )
    except Exception as e:
        logger.error(f"HTTP user tool execution failed: {e}")
        # Return error as tool message
        return ToolMessage(
            id=tool_call.id,
            role="tool",
            content=f"Error: {str(e)}",
            requestor=tool_call.requestor,
            error=True
        )


# Class-level state for UserSimulator (shared across conversation loop calls)
_user_simulator: Optional[UserSimulator] = None
_user_state = None


def initialize_user_simulator(tau2_task):
    """Initialize the UserSimulator with task scenario.

    This is called by scenarios.py during setup to prepare the UserSimulator
    for the conversation loop in hud.agents.base._run_conversation_loop().
    """
    global _user_simulator, _user_state

    task = tau2_task.task
    if not task or not task.user_scenario:
        raise ValueError("Task must have user_scenario for multi-turn conversation mode.")

    # Get user LLM configuration
    # Use gpt-4o for larger context window (128k vs 8k in gpt-4-0613)
    user_llm = os.getenv("USER_LLM", "gpt-4.1")
    user_llm_args = {
        "temperature": float(os.getenv("USER_TEMPERATURE", "0.0")),
        # "max_tokens": int(os.getenv("USER_MAX_TOKENS", "2500")),
    }

    # Get user tools from HTTP environment server
    # Create Tool objects with dummy functions (actual execution happens via HTTP)
    user_tools = None
    try:
        from server.tools.http_client import get_http_client
        from tau2.environment.toolkit import Tool

        http_client = get_http_client()
        tools_data = http_client.list_tools()

        logger.info(f"Tools data from server: agent_tools={len(tools_data.get('tools', []))}, user_tools={len(tools_data.get('user_tools', []))}")

        # Convert user_tools to Tool objects with dummy functions
        if "user_tools" in tools_data and tools_data["user_tools"]:
            import inspect
            from typing import get_type_hints

            user_tools = []
            for tool_info in tools_data["user_tools"]:
                tool_name = tool_info["name"]
                tool_desc = tool_info["description"]
                tool_params = tool_info.get("parameters", {"type": "object", "properties": {}})

                # Create a dummy function with dynamic signature matching the tool's parameters
                # The actual execution happens via HTTP in execute_user_tool_via_http
                def make_dummy_func(name, desc, params_schema):
                    # Extract parameter names from schema
                    properties = params_schema.get("properties", {})
                    required = params_schema.get("required", [])

                    # Build parameter list for function signature
                    param_list = []
                    for param_name, param_info in properties.items():
                        if param_name in required:
                            # Required parameter
                            param_list.append(inspect.Parameter(
                                param_name,
                                inspect.Parameter.POSITIONAL_OR_KEYWORD
                            ))
                        else:
                            # Optional parameter with default None
                            param_list.append(inspect.Parameter(
                                param_name,
                                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                                default=None
                            ))

                    # Create function with dynamic signature
                    def dummy_func(**kwargs):
                        """Dummy function - actual execution via HTTP"""
                        pass

                    # Set function metadata
                    dummy_func.__name__ = name
                    dummy_func.__doc__ = desc
                    dummy_func.__signature__ = inspect.Signature(param_list)

                    return dummy_func

                dummy_func = make_dummy_func(tool_name, tool_desc, tool_params)

                # Create Tool object with the dummy function
                tool = Tool(func=dummy_func)
                user_tools.append(tool)

            logger.info(f"Loaded {len(user_tools)} user tools for UserSimulator")
        else:
            logger.warning("No user_tools found in tools_data!")
    except Exception as e:
        logger.error(f"Failed to load user tools: {e}")
        import traceback
        traceback.print_exc()

    # Create UserSimulator
    _user_simulator = UserSimulator(
        tools=user_tools,
        instructions=task.user_scenario.instructions,
        llm=user_llm,
        llm_args=user_llm_args,
    )
    _user_state = _user_simulator.get_init_state(message_history=[])


def get_user_simulator():
    """Get the initialized UserSimulator instance."""
    return _user_simulator


def get_user_state():
    """Get the current user simulator state."""
    return _user_state


def set_user_state(state):
    """Update the user simulator state."""
    global _user_state
    _user_state = state
