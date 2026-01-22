"""Multi-turn TAU-Bench agent interaction"""

import asyncio
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
    if hasattr(agent.config, "allowed_tools") and agent.config.allowed_tools is not None:
        agent._available_tools = [
            t for t in agent._available_tools if t.name in agent.config.allowed_tools
        ]
        agent._tool_map = {t.name: t for t in agent._available_tools}
        tool_names = (
            ", ".join(t.name for t in agent._available_tools)
            if agent._available_tools
            else "(none)"
        )
        if len(tool_names) > 100:
            tool_names = f"{tool_names[:97]}..."
        agent.console.info(
            f"Filtered to {len(agent._available_tools)} tools: {tool_names}"
        )
        # Re-trigger provider-specific tool conversion
        agent._on_tools_ready()


async def get_response(
    agent: Any,
    conversation: list[Any],
) -> tuple[str, int]:
    """Run tool calls until a text response is produced."""
    tool_steps = 0
    while True:
        response = await agent.get_response(conversation)
        if response.tool_calls:
            tool_steps += 1
            tool_results = await agent.call_tools(response.tool_calls)
            tool_messages = await agent.format_tool_results(
                response.tool_calls, tool_results
            )
            conversation.extend(tool_messages)
            if response.content:
                text = response.content
                conversation.extend(await agent.format_message(text))
                return text, tool_steps
            continue

        text = response.content or ""
        conversation.extend(await agent.format_message(text))
        return text, (tool_steps+1)


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

    # Ensure message logging tool is available (for communicate checks)
    await ctx.list_tools()
    tool_names = {t.name for t in ctx.as_tools()}
    has_record_message = "record_message" in tool_names
    if not has_record_message:
        logger.warning("Missing record_message tool; communicate checks may fail")

    # Initialize both agents with context and apply tool filtering
    await _initialize_agent_with_filters(agent, ctx)
    await _initialize_agent_with_filters(simulated_user, ctx)

    try:
        # Run conversation loop
        result = await _run_conversation_loop(
            agent,
            simulated_user,
            text_to_blocks(ctx.prompt),
            ctx=ctx,
            has_record_message=has_record_message,
            max_steps=max_steps,
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
    agent,
    simulated_user,
    context: list[Any],
    *,
    ctx: EvalContext,
    has_record_message: bool,
    max_steps: int = 100,
) -> Trace:
    """
    Core conversation loop with HUD agent-based user simulation.

    This implements a turn-based conversation:
    1. Agent acts (tool calls or message)
    2. When agent stops calling tools, user agent responds
    3. Both agents share the same conversation history
    """
    final_content: str | None = None
    error = None
    agent_messages: list[Any] = []
    user_messages: list[Any] = []
    text_history: list[dict[str, str]] = []

    def _one_line(text: str, limit: int = 60) -> str:
        collapsed = " ".join(text.split())
        if len(collapsed) > limit:
            return f"{collapsed[:limit - 3]}..."
        return collapsed

    def check_for_stop_signal(text: str) -> bool:
        """Check if text contains a conversation stop signal."""
        for signal in STOP_SIGNALS:
            if signal in text:
                logger.info(f"Detected stop signal: {signal}")
                return True
        return False

    try:
        # Start with system messages for assistant and user agents
        agent_messages = await agent.get_system_messages()
        user_messages = await simulated_user.get_system_messages()

        # Add initial context (greeting instruction) - ONLY for assistant agent
        # User agent should not see this instruction
        agent_messages.extend(await agent.format_message(context))
        agent.console.debug(
            f"Agent messages initialized (count={len(agent_messages)})"
        )

        async def append_agent_message(role: str, content: str) -> None:
            agent_messages.extend(await agent.format_message(content))

        async def append_user_message(role: str, content: str) -> None:
            user_messages.extend(await simulated_user.format_message(content))

        async def append_conversation_message(role: str, content: str) -> None:
            # Agent sees roles as-is; user sees roles flipped (matches tau2-bench)
            await append_agent_message(role, content)
            flipped_role = "user" if role == "assistant" else "assistant"
            await append_user_message(flipped_role, content)
            text_history.append({"role": role, "content": content})

        async def get_text_response(agent_obj, messages, label: str) -> str:
            try:
                text, _ = await get_response(agent_obj, messages)
                return text
            except asyncio.TimeoutError:
                logger.error("%s response timed out", label)
                return "Sorry, I took too long to respond."
            except Exception as exc:
                logger.error("Failed to get %s response: %s", label, exc)
                import traceback
                traceback.print_exc()
                return "Error getting response"

        step_count = 0
        iteration_count = 0  # Track total iterations

        while max_steps == -1 or step_count < max_steps:
            iteration_count += 1
            if max_steps == -1:
                agent.console.debug(
                    _one_line(f"Iteration {iteration_count}, Step {step_count} (unlimited)")
                )
            else:
                agent.console.debug(
                    _one_line(f"Iteration {iteration_count}, Step {step_count}/{max_steps}")
                )

            try:
                # 1. Get agent response (tool calls loop handled inside get_response)
                agent_message = await get_text_response(agent, agent_messages, "agent")
                if check_for_stop_signal(agent_message):
                    agent.console.info("Conversation ended by agent")
                    final_content = agent_message
                    break

                agent.console.info(f"Agent: {_one_line(agent_message)}")
                await append_conversation_message("assistant", agent_message)

                user_response = await get_text_response(
                    simulated_user, user_messages, "user"
                )

                if check_for_stop_signal(user_response):
                    agent.console.info("Conversation ended by user signal")
                    final_content = user_response
                    break

                agent.console.info(f"User: {_one_line(user_response)}")
                await append_conversation_message("user", user_response)
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
    is_error = error is not None

    if has_record_message and text_history:
        try:
            await ctx.call_tool("record_message", conversation=text_history)
        except Exception as e:
            logger.error("record_message failed: %s", e)

    trace_params = {
        "reward": 0.0,
        "done": True,
        "messages": text_history,
        "content": final_content if final_content is not None else error,
        "isError": is_error,
        "info": {"error": error} if error else {},
    }
    trace_result = Trace(**trace_params)

    return trace_result
