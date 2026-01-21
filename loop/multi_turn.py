"""Multi-turn TAU-Bench agent interaction"""

import asyncio
import json
import logging
from typing import Any

from hud.eval.context import EvalContext
from hud.types import Trace
from hud.agents.base import text_to_blocks

logger = logging.getLogger(__name__)

# Conversation termination signals
STOP_SIGNALS = ["###STOP###", "###TRANSFER###", "###OUT-OF-SCOPE###"]


async def _initialize_agent_with_filters(agent: Any, ctx: EvalContext) -> None:
    """Initialize agent and apply tool filtering if configured."""
    if agent._initialized:
        return

    await agent._initialize_from_ctx(ctx)

    # Apply allowed_tools filter if configured
    # Note: We explicitly check 'is not None' because an empty list [] is a valid
    # filter that should result in 0 tools (for domains with no user tools)
    if hasattr(agent.config, 'allowed_tools') and agent.config.allowed_tools is not None:
        agent._available_tools = [
            t for t in agent._available_tools if t.name in agent.config.allowed_tools
        ]
        agent._tool_map = {t.name: t for t in agent._available_tools}
        tool_names = ', '.join(t.name for t in agent._available_tools) if agent._available_tools else '(none)'
        agent.console.info(
            f"Filtered to {len(agent._available_tools)} tools: {tool_names}"
        )
        # Re-trigger provider-specific tool conversion
        agent._on_tools_ready()


def _uses_role_dict_format(messages: list[Any]) -> bool:
    """Best-effort check for OpenAI-style role dict messages."""
    return bool(messages) and isinstance(messages[0], dict) and "role" in messages[0]


def _tool_calls_to_openai(tool_calls: list[Any]) -> list[dict[str, Any]]:
    """Convert MCPToolCall list to OpenAI tool_calls format."""
    formatted = []
    for call in tool_calls:
        formatted.append(
            {
                "id": call.id,
                "type": "function",
                "function": {
                    "name": call.name,
                    "arguments": json.dumps(call.arguments or {}),
                },
            }
        )
    return formatted


def _ensure_openai_tool_messages(
    tool_calls: list[Any], tool_messages: list[Any]
) -> list[Any]:
    """Ensure every tool_call_id has a corresponding tool message."""
    existing_ids = {
        msg.get("tool_call_id")
        for msg in tool_messages
        if isinstance(msg, dict) and msg.get("role") == "tool"
    }
    for call in tool_calls:
        if call.id not in existing_ids:
            tool_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": "Tool executed successfully",
                }
            )
    return tool_messages


def _repair_openai_tool_history(messages: list[Any]) -> list[Any]:
    """Ensure tool messages immediately follow assistant tool_calls."""
    if not _uses_role_dict_format(messages):
        return messages

    i = 0
    while i < len(messages):
        msg = messages[i]
        if isinstance(msg, dict) and msg.get("role") == "assistant" and msg.get("tool_calls"):
            tool_calls = msg.get("tool_calls") or []
            pending = {tc.get("id") for tc in tool_calls if isinstance(tc, dict) and tc.get("id")}
            j = i + 1
            while j < len(messages):
                next_msg = messages[j]
                if isinstance(next_msg, dict) and next_msg.get("role") == "tool":
                    tool_call_id = next_msg.get("tool_call_id")
                    if tool_call_id in pending:
                        pending.remove(tool_call_id)
                    j += 1
                    continue
                break
            if pending:
                insert_at = j
                for tool_call_id in pending:
                    messages.insert(
                        insert_at,
                        {
                            "role": "tool",
                            "tool_call_id": tool_call_id,
                            "content": "Tool executed successfully",
                        },
                    )
                    insert_at += 1
                i = insert_at
            else:
                i = j
        else:
            i += 1

    return messages


async def multi_turn_run(
    ctx: EvalContext,
    agent: Any,
    simulated_user,
    max_steps: int = 30,
) -> Trace:
    """
    Run agent evaluation with conversation-aware stopping conditions.

    This function implements a turn-based conversation loop:
    1. Agent acts (tool calls or message)
    2. When agent stops calling tools, user agent responds
    3. Both agents share the same conversation history

    The loop terminates when detecting conversation signals from tau2-bench:
    - ###STOP### - Task complete, user is satisfied
    - ###TRANSFER### - User wants to speak to a human
    - ###OUT-OF-SCOPE### - Scenario doesn't provide enough information

    Args:
        ctx: EvalContext from hud.eval()
        agent: MCPAgent instance (assistant agent)
        simulated_user: MCPAgent instance (user agent)
        max_steps: Maximum number of agent steps

    Returns:
        Trace with done, content, isError fields

    Usage:
        ```python
        async with hud.eval(task) as ctx:
            # Create assistant agent with agent tools
            assistant = create_agent(model="gpt-4o", allowed_tools=agent_tool_names)

            # Create user agent with user tools and user scenario prompt
            user = create_agent(model="gpt-4o", system_prompt=user_prompt,
                              allowed_tools=user_tool_names)

            # Run multi-turn conversation
            trace = await multi_turn_run(ctx, assistant, user, max_steps=30)
        ```
    """
    if not isinstance(ctx, EvalContext):
        raise TypeError(f"ctx must be EvalContext, got {type(ctx).__name__}")

    if not ctx.prompt:
        raise ValueError("ctx.prompt is not set - did the scenario setup run?")

    # NOTE: Agents should be pre-configured with system_prompt and allowed_tools
    # before calling this function. See loop/agent_config.py for helper functions.

    # Store context for tool calls in both agents
    agent.ctx = ctx
    simulated_user.ctx = ctx

    # Initialize both agents with context and apply tool filtering
    await _initialize_agent_with_filters(agent, ctx)
    await _initialize_agent_with_filters(simulated_user, ctx)

    try:
        # Run conversation loop
        result = await _run_conversation_loop(
            agent, simulated_user, text_to_blocks(ctx.prompt), max_steps=max_steps
        )

        # Submit final answer to context (only if scenario is running)
        if result.content and ctx.has_scenario:
            await ctx.submit(result.content)

        return result

    except Exception as e:
        logger.exception("Error while running agent:")
        return Trace(
            reward=0.0,
            done=True,
            content=f"Agent failed with error: {e}",
            isError=True,
            info={"error": str(e)},
        )
    finally:
        # Cleanup auto-created resources
        await agent._cleanup()
        await simulated_user._cleanup()


async def _run_conversation_loop(
    agent, simulated_user, context: list[Any], *, max_steps: int = 100
) -> Trace:
    """
    Core conversation loop with HUD agent-based user simulation.

    This implements a turn-based conversation:
    1. Agent acts (tool calls or message)
    2. When agent stops calling tools, user agent responds
    3. Both agents share the same conversation history
    """
    final_response = None
    error = None
    agent_messages: list[Any] = []
    user_messages: list[Any] = []

    def check_for_stop_signal(text: str) -> bool:
        """Check if text contains a conversation stop signal."""
        for signal in STOP_SIGNALS:
            if signal in text:
                logger.info(f"Detected stop signal: {signal}")
                return True
        return False

    async def get_user_response() -> str:
        """Get user agent response using shared conversation history.

        User agent sees the same chat history as the assistant agent (roles flipped),
        EXCEPT the initial greeting instruction (which is only for the assistant).
        User can call tools before responding.
        """
        try:
            # User can call tools and respond (max iterations to prevent infinite loops)
            max_user_iterations = 6
            for _ in range(max_user_iterations):
                _repair_openai_tool_history(user_messages)
                user_response_obj = await simulated_user.get_response(user_messages)

                # If user has tool calls, execute them
                if user_response_obj.tool_calls:
                    if user_response_obj.content:
                        # Some providers include content with tool calls; ignore text and continue.
                        logger.warning(
                            "User returned both text and tool calls; ignoring text content"
                        )

                    logger.info(f"User executing {len(user_response_obj.tool_calls)} tool(s)")

                    if _uses_role_dict_format(user_messages):
                        user_messages.append(
                            {
                                "role": "assistant",
                                "content": "",
                                "tool_calls": _tool_calls_to_openai(user_response_obj.tool_calls),
                            }
                        )

                    user_tool_results = await simulated_user.call_tools(user_response_obj.tool_calls)

                    # Format tool results and add to user's message history
                    tool_messages = await simulated_user.format_tool_results(
                        user_response_obj.tool_calls, user_tool_results
                    )
                    if _uses_role_dict_format(user_messages):
                        tool_messages = _ensure_openai_tool_messages(
                            user_response_obj.tool_calls, tool_messages
                        )
                    user_messages.extend(tool_messages)

                    # Continue to get text response after tools
                    continue
                else:
                    # No tool calls - user has a text response
                    user_text = user_response_obj.content or ""
                    if _uses_role_dict_format(user_messages):
                        user_messages.append({"role": "assistant", "content": user_text})
                    else:
                        user_messages.extend(await simulated_user.format_message(user_text))
                    return user_text

            # Max iterations reached - return last content
            return user_response_obj.content or ""

        except asyncio.TimeoutError:
            logger.error("User response timed out")
            return "Sorry, I took too long to respond."
        except Exception as e:
            logger.error(f"Failed to get user response: {e}")
            import traceback
            traceback.print_exc()
            return f"Error getting user response: {e}"

    try:
        # Start with system messages for assistant and user agents
        agent_messages = await agent.get_system_messages()
        user_messages = await simulated_user.get_system_messages()
        agent_uses_roles = _uses_role_dict_format(agent_messages)
        user_uses_roles = _uses_role_dict_format(user_messages)

        # Add initial context (greeting instruction) - ONLY for assistant agent
        # User agent should not see this instruction
        agent_messages.extend(await agent.format_message(context))
        agent.console.debug(f"Agent messages: {agent_messages}")

        async def append_agent_message(role: str, content: str) -> None:
            if agent_uses_roles:
                agent_messages.append({"role": role, "content": content})
            else:
                agent_messages.extend(await agent.format_message(content))

        async def append_user_message(role: str, content: str) -> None:
            if user_uses_roles:
                user_messages.append({"role": role, "content": content})
            else:
                user_messages.extend(await simulated_user.format_message(content))

        async def append_conversation_message(role: str, content: str) -> None:
            # Agent sees roles as-is; user sees roles flipped (matches tau2-bench)
            await append_agent_message(role, content)
            flipped_role = "user" if role == "assistant" else "assistant"
            await append_user_message(flipped_role, content)

        step_count = 0
        iteration_count = 0  # Track total iterations

        while max_steps == -1 or step_count < max_steps:
            iteration_count += 1
            if max_steps == -1:
                agent.console.debug(f"Iteration {iteration_count}, Step {step_count} (unlimited)")
            else:
                agent.console.debug(f"Iteration {iteration_count}, Step {step_count}/{max_steps}")

            try:
                # 1. Get model response
                if agent_uses_roles:
                    agent_messages = _repair_openai_tool_history(agent_messages)
                response = await agent.get_response(agent_messages)
                agent.console.debug(f"Agent:\n{response}")

                # 2. Check if agent has tool calls
                if response.tool_calls:
                    if response.content:
                        # Some providers include text with tool calls; ignore text and continue.
                        agent.console.warning_log(
                            "Agent returned both text and tool calls; ignoring text content"
                        )

                    if agent_uses_roles:
                        agent_messages.append(
                            {
                                "role": "assistant",
                                "content": "",
                                "tool_calls": _tool_calls_to_openai(response.tool_calls),
                            }
                        )

                    # Execute tools
                    tool_calls = response.tool_calls
                    tool_results = await agent.call_tools(tool_calls)

                    # Format tool results and add to messages
                    tool_messages = await agent.format_tool_results(tool_calls, tool_results)
                    if agent_uses_roles:
                        tool_messages = _ensure_openai_tool_messages(tool_calls, tool_messages)
                    agent_messages.extend(tool_messages)

                    # Show tool calls and results in compact one-line format
                    for call, result in zip(tool_calls, tool_results, strict=False):
                        # One line for tool call
                        agent.console.info(f"{call}")
                        # One line for result
                        agent.console.info(f"{result}")

                else:
                    # No tool calls - agent sent a message to user
                    agent_message = response.content or ""

                    # Check if agent's message signals end
                    if check_for_stop_signal(agent_message):
                        agent.console.info("Conversation ended by agent")
                        final_response = response
                        break

                    agent.console.info(f"[bold cyan]Agent:[/bold cyan] {agent_message}")

                    # Add agent message to shared history
                    await append_conversation_message("assistant", agent_message)

                    # Get user response using shared message history
                    user_response = await get_user_response()

                    # Check if user's response signals end
                    if check_for_stop_signal(user_response):
                        agent.console.info("Conversation ended by user signal")
                        final_response = response
                        break

                    agent.console.info(f"[bold green]User:[/bold green] {user_response}")

                    # Add user response to shared messages
                    await append_conversation_message("user", user_response)

                    # Increment step count after non-tool exchange (matches tau2-bench)
                    step_count += 1

            except Exception as e:
                agent.console.error_log(f"Step failed: {e}")
                error = str(e)
                break

    except KeyboardInterrupt:
        agent.console.warning_log("Agent execution interrupted by user")
        error = "Interrupted by user"
    except asyncio.CancelledError:
        agent.console.warning_log("Agent execution cancelled")
        error = "Cancelled"
    except Exception as e:
        agent.console.error_log(f"Unexpected error: {e}")
        error = str(e)

    # Build result
    if error is not None or (
        final_response and hasattr(final_response, "isError") and final_response.isError
    ):
        is_error = True
    else:
        is_error = False

    # Ensure all parameters are the correct type
    trace_params = {
        "reward": 0.0,
        "done": True,
        "messages": agent_messages,
        "content": final_response.content if final_response else error,
        "isError": is_error,
        "info": {"error": error} if error else {},
    }
    trace_result = Trace(**trace_params)

    return trace_result
    