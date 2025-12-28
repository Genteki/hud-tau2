"""Custom agent loop for tau2-bench multi-turn conversations.

This module provides a custom evaluation loop that extends the standard
agent.run() behavior with conversation-specific stopping conditions.
"""

import logging
from typing import Any

from hud.types import Trace

logger = logging.getLogger(__name__)


async def multi_agent_loop(ctx, agent, max_steps: int = 30) -> Trace:
    """
    Run agent evaluation with conversation-aware stopping conditions.

    This function wraps the standard agent.run() flow but adds additional
    checks for conversation termination signals from tau2-bench:
    - ###STOP### - Task complete, user is satisfied
    - ###TRANSFER### - User wants to speak to a human
    - ###OUT-OF-SCOPE### - Scenario doesn't provide enough information

    Args:
        ctx: EvalContext from hud.eval()
        agent: MCPAgent instance to run
        max_steps: Maximum number of agent steps

    Returns:
        Trace with done, content, isError fields

    Usage:
        ```python
        async with hud.eval(task) as ctx:
            agent = OpenAIChatAgent.create(model="gpt-4o")
            trace = await multi_agent_loop(ctx, agent, max_steps=30)
        ```
    """
    from hud.agents.base import text_to_blocks
    from hud.eval.context import EvalContext

    if not isinstance(ctx, EvalContext):
        raise TypeError(f"ctx must be EvalContext, got {type(ctx).__name__}")

    if not ctx.prompt:
        raise ValueError("ctx.prompt is not set - did the scenario setup run?")

    # Store context for tool calls
    agent.ctx = ctx

    # Initialize tools from context
    if not agent._initialized:
        await agent._initialize_from_ctx(ctx)

    try:
        # Run custom conversation loop
        result = await _run_conversation_loop(
            agent, text_to_blocks(ctx.prompt), max_steps=max_steps
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


async def _run_conversation_loop(
    agent, context: list[Any], *, max_steps: int = 30
) -> Trace:
    """
    Core conversation loop with tau2-bench turn-based interaction.

    This implements a turn-based conversation:
    1. Agent acts (tool calls or message)
    2. When agent stops calling tools, user simulator responds
    3. Agent response is added to messages, loop continues
    """
    import asyncio
    from typing import Literal

    final_response = None
    error = None
    messages: list[Any] = []

    # Conversation termination signals
    STOP_SIGNALS = ["###STOP###", "###TRANSFER###", "###OUT-OF-SCOPE###"]

    def check_for_stop_signal(text: str) -> bool:
        """Check if text contains a conversation stop signal."""
        for signal in STOP_SIGNALS:
            if signal in text:
                logger.info(f"Detected stop signal: {signal}")
                return True
        return False

    async def get_user_response(agent_message: str) -> str:
        """Get user simulator response to agent's message.

        Handles user tool calls by executing them via HTTP and getting next response.
        Uses tau2-bench's UserSimulator directly.
        """
        try:
            from server.state import get_tau2_task
            from server.tools.conversation import ConversationTool, execute_user_tool_via_http
            from datetime import datetime
            from tau2.data_model.message import AssistantMessage, MultiToolMessage

            # Get tau2 task state
            tau2_task = get_tau2_task()

            # Create agent message for tau2
            agent_msg = AssistantMessage(
                role="assistant",
                content=agent_message,
                cost=0.0,
                timestamp=datetime.now().isoformat()
            )
            tau2_task.add_message(agent_msg)

            # Generate initial user response using tau2-bench's UserSimulator
            user_message, new_state = ConversationTool._user_simulator.generate_next_message(
                message=agent_msg, state=ConversationTool._user_state
            )
            ConversationTool._user_state = new_state
            tau2_task.add_message(user_message)

            # Handle user tool calls (user performs actions)
            while user_message.is_tool_call():
                logger.info(f"User executing {len(user_message.tool_calls)} tool(s)")

                # Execute user tools via HTTP
                tool_messages = []
                for tool_call in user_message.tool_calls:
                    tool_msg = execute_user_tool_via_http(tool_call)
                    tool_messages.append(tool_msg)
                    tau2_task.add_message(tool_msg)
                    logger.info(f"  - {tool_call.name}({tool_call.arguments})")

                # Get next user response after tool execution
                multi_tool_msg = MultiToolMessage(role="tool", tool_messages=tool_messages)
                user_message, new_state = ConversationTool._user_simulator.generate_next_message(
                    message=multi_tool_msg, state=ConversationTool._user_state
                )
                ConversationTool._user_state = new_state
                tau2_task.add_message(user_message)

            # Return user's text response
            return user_message.content or ""

        except Exception as e:
            logger.error(f"Failed to get user response: {e}")
            import traceback
            traceback.print_exc()
            return f"Error getting user response: {e}"

    try:
        # Start with system messages
        messages = await agent.get_system_messages()

        # Add initial context
        messages.extend(await agent.format_message(context))
        agent.console.debug(f"Messages: {messages}")

        step_count = 0
        while max_steps == -1 or step_count < max_steps:
            step_count += 1
            if max_steps == -1:
                agent.console.debug(f"Step {step_count} (unlimited)")
            else:
                agent.console.debug(f"Step {step_count}/{max_steps}")

            try:
                # 1. Get model response
                response = await agent.get_response(messages)
                agent.console.debug(f"Agent:\n{response}")

                # 2. Add agent response to messages (before checking tool calls)
                messages.extend(await agent.format_message(response))

                # 3. Check if agent has tool calls
                if response.tool_calls:
                    # Execute tools
                    tool_calls = response.tool_calls
                    tool_results = await agent.call_tools(tool_calls)

                    # Format tool results and add to messages
                    tool_messages = await agent.format_tool_results(tool_calls, tool_results)
                    messages.extend(tool_messages)

                    # Compact step completion display
                    step_info = f"\n[bold]Step {step_count}"
                    if max_steps != -1:
                        step_info += f"/{max_steps}"
                    step_info += "[/bold]"

                    # Show tool calls and results in compact format
                    for call, result in zip(tool_calls, tool_results, strict=False):
                        step_info += f"\n{call}\n{result}"

                    agent.console.info_log(step_info)

                    # Check if agent included text with tool calls (like transfer message)
                    if response.content:
                        agent_message = response.content
                        agent.console.info(f"[bold cyan]Agent (with tools):[/bold cyan] {agent_message}")

                        # Check if agent's message signals end
                        if check_for_stop_signal(agent_message):
                            agent.console.info("Conversation ended by agent signal")
                            final_response = response
                            break

                        # Get user response to agent's message
                        user_response = await get_user_response(agent_message)

                        # Check if user's response signals end
                        if check_for_stop_signal(user_response):
                            agent.console.info("Conversation ended by user signal")
                            final_response = response
                            break

                        agent.console.info(f"[bold green]User:[/bold green] {user_response}")

                        # Add user response to messages
                        messages.extend(await agent.format_message(user_response))

                else:
                    # No tool calls - agent sent a message to user
                    # Get user simulator response
                    agent_message = response.content or ""

                    # Check if agent's message signals end
                    if check_for_stop_signal(agent_message):
                        agent.console.info("Conversation ended by agent")
                        final_response = response
                        break

                    agent.console.info(f"[bold cyan]Agent:[/bold cyan] {agent_message}")

                    # Get user response
                    user_response = await get_user_response(agent_message)

                    # Check if user's response signals end
                    if check_for_stop_signal(user_response):
                        agent.console.info("Conversation ended by user signal")
                        final_response = response
                        break

                    agent.console.info(f"[bold green]User:[/bold green] {user_response}")

                    # Add user response to messages and continue
                    messages.extend(await agent.format_message(user_response))

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
        "messages": messages,
        "content": final_response.content if final_response else error,
        "isError": is_error,
        "info": {"error": error} if error else {},
    }
    trace_result = Trace(**trace_params)

    return trace_result
