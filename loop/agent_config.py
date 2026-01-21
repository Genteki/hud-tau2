"""Agent configuration helper for tau2-bench tasks.

This module provides utilities to configure agents with proper system prompts
and tool filters based on tau2 task parameters.
"""

import logging
from typing import Any
from hud.eval.context import EvalContext

logger = logging.getLogger(__name__)


async def configure_agents_for_tau2(
    ctx: EvalContext,
    agent: Any,
    user_agent: Any
) -> None:
    """
    Configure agents with tau2-bench system prompts and tool filters.

    This function works in both local and remote execution:
    - Local: Uses global tau2_task state (populated by scenario setup)
    - Remote: Extracts params from ctx.task and loads from registry

    Args:
        ctx: HUD evaluation context
        agent: Assistant agent to configure
        user_agent: User agent to configure
    """
    from prompts.user_prompts import user_system_prompt

    logger.info("[CONFIG] Configuring agents for tau2 task")

    # Try local execution first (check global state)
    try:
        from server.state import get_tau2_task
        tau2_task_state = get_tau2_task()

        # Check if global state is populated (local execution)
        if (tau2_task_state.domain and
            tau2_task_state.task and
            tau2_task_state.agent_tool_names):

            logger.info("[CONFIG] Using global tau2_task state (local execution)")

            tau2_task = tau2_task_state.task
            agent_tool_names = tau2_task_state.agent_tool_names
            user_tool_names = tau2_task_state.user_tool_names
            agent_system_prompt = tau2_task_state.system_prompt

            logger.info(f"[CONFIG] From state: domain={tau2_task_state.domain}, "
                       f"task_id={tau2_task_state.task_id}, "
                       f"agent_tools={len(agent_tool_names)}, "
                       f"user_tools={len(user_tool_names)}")

            # Configure assistant agent
            if not hasattr(agent.config, 'system_prompt') or not agent.config.system_prompt:
                agent.config.system_prompt = agent_system_prompt
                agent.system_prompt = agent_system_prompt
                logger.info("[CONFIG] Set agent system prompt")

            if not hasattr(agent.config, 'allowed_tools') or agent.config.allowed_tools is None:
                agent.config.allowed_tools = agent_tool_names
                logger.info(f"[CONFIG] Set agent allowed_tools: {len(agent_tool_names)} tools")

            # Configure user agent
            if not hasattr(user_agent.config, 'system_prompt') or not user_agent.config.system_prompt:
                if tau2_task.user_scenario:
                    user_prompt = user_system_prompt(
                        user_scenario=tau2_task.user_scenario,
                        user_tool_names=user_tool_names
                    )
                    user_agent.config.system_prompt = user_prompt
                    user_agent.system_prompt = user_prompt
                    logger.info("[CONFIG] Set user system prompt")
                else:
                    logger.warning("[CONFIG] No user_scenario in task")

            if not hasattr(user_agent.config, 'allowed_tools') or user_agent.config.allowed_tools is None:
                user_agent.config.allowed_tools = user_tool_names
                logger.info(f"[CONFIG] Set user allowed_tools: {len(user_tool_names)} tools")

            logger.info("[CONFIG] Agent configuration complete (local)")
            return  # Success - exit early

    except Exception as e:
        logger.info(f"[CONFIG] Global state not available: {e}")

    # Remote execution - load from ctx.task and registry
    logger.info("[CONFIG] Using ctx.task for configuration (remote execution)")

    if not hasattr(ctx, 'task') or not ctx.task:
        logger.error("[CONFIG] No task found in context or global state")
        return

    task_args = ctx.task.get('args', {})
    domain = task_args.get('domain', 'airline')
    task_id = str(task_args.get('task_id', 0))
    task_split = task_args.get('task_split', 'base')

    logger.info(f"[CONFIG] Task params from ctx: domain={domain}, task_id={task_id}, split={task_split}")

    try:
        from tau2.registry import registry
        from server.tools.http_client import get_http_client

        # Load the tau2 task from registry
        task_loader = registry.get_tasks_loader(domain)
        tasks = task_loader(task_split_name=task_split)

        # Find the specific task
        tau2_task = None
        for t in tasks:
            if str(t.id) == task_id:
                tau2_task = t
                break

        if tau2_task is None:
            logger.error(f"[CONFIG] Task {task_id} not found in {domain}/{task_split}")
            return

        logger.info(f"[CONFIG] Loaded tau2 task: {tau2_task.id}")

        # Get policy from environment (via HTTP)
        http_client = get_http_client()

        try:
            policy_response = http_client.get_policy()
            policy = policy_response.get('policy', 'No policy available')
        except Exception as e:
            logger.warning(f"[CONFIG] Could not get policy: {e}")
            policy = "No policy available"

        # Build agent system prompt
        agent_system_prompt = f"""<instructions>
You are a customer service agent that helps the user according to the <policy> provided below.
In each turn you can either:
- Send a message to the user (by providing text in your response).
- Make a tool call to check or modify data.
You cannot do both at the same time.

Try to be helpful and always follow the policy.
</instructions>

<policy>
{policy}
</policy>"""

        # Get tool names from environment
        try:
            tools_data = http_client.list_tools()
            agent_tool_names = [t["name"] for t in tools_data.get("tools", []) if t["name"] != "send_message"]
            user_tool_names = [t["name"] for t in tools_data.get("user_tools", [])]
        except Exception as e:
            logger.error(f"[CONFIG] Could not get tool names: {e}")
            agent_tool_names = []
            user_tool_names = []

        logger.info(f"[CONFIG] Agent tools: {len(agent_tool_names)}, User tools: {len(user_tool_names)}")

        # Configure assistant agent
        if not hasattr(agent.config, 'system_prompt') or not agent.config.system_prompt:
            agent.config.system_prompt = agent_system_prompt
            agent.system_prompt = agent_system_prompt
            logger.info("[CONFIG] Set agent system prompt")

        if not hasattr(agent.config, 'allowed_tools') or agent.config.allowed_tools is None:
            agent.config.allowed_tools = agent_tool_names
            logger.info(f"[CONFIG] Set agent allowed_tools: {len(agent_tool_names)} tools")

        # Configure user agent
        if not hasattr(user_agent.config, 'system_prompt') or not user_agent.config.system_prompt:
            if tau2_task.user_scenario:
                user_prompt = user_system_prompt(
                    user_scenario=tau2_task.user_scenario,
                    user_tool_names=user_tool_names
                )
                user_agent.config.system_prompt = user_prompt
                user_agent.system_prompt = user_prompt
                logger.info("[CONFIG] Set user system prompt")
            else:
                logger.warning("[CONFIG] No user_scenario in task")

        if not hasattr(user_agent.config, 'allowed_tools') or user_agent.config.allowed_tools is None:
            user_agent.config.allowed_tools = user_tool_names
            logger.info(f"[CONFIG] Set user allowed_tools: {len(user_tool_names)} tools")

        logger.info("[CONFIG] Agent configuration complete (remote)")

    except Exception as e:
        logger.error(f"[CONFIG] Failed to configure agents: {e}")
        import traceback
        traceback.print_exc()
