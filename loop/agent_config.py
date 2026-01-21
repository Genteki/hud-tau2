"""Agent configuration helper for tau2-bench tasks.

This module provides utilities to get agent configuration (system prompts and tools)
based on tau2 task parameters from the evaluation context.
"""

import logging
import re
from pathlib import Path
from typing import Tuple, List, Any
from hud.eval.context import EvalContext

logger = logging.getLogger(__name__)

def _load_domain_policy_from_file(domain: str) -> str | None:
    """Load policy from tau2-bench data files as a fallback."""
    try:
        if domain == "airline":
            from tau2.domains.airline.utils import AIRLINE_POLICY_PATH

            policy_path = AIRLINE_POLICY_PATH
        elif domain == "retail":
            from tau2.domains.retail.utils import RETAIL_POLICY_PATH

            policy_path = RETAIL_POLICY_PATH
        elif domain == "telecom":
            from tau2.domains.telecom.utils import TELECOM_MAIN_POLICY_PATH

            policy_path = TELECOM_MAIN_POLICY_PATH
        else:
            return None

        if isinstance(policy_path, Path) and policy_path.exists():
            return policy_path.read_text(encoding="utf-8")
    except Exception as e:
        logger.critical(f"[CONFIG] Failed to load fallback policy: {e}")

    return None
def _extract_task_params_from_ctx(ctx: EvalContext) -> dict[str, Any]:
    """Extract tau2 task parameters from EvalContext if available.

    Prefer task args from the EvalContext (preserves telecom task IDs with brackets).
    Falls back to an empty dict when not available.
    """
    task = getattr(ctx, "_task", None)
    if task is None:
        return {}

    args = getattr(task, "args", None)
    if isinstance(args, dict):
        return args

    return {}


async def get_tau2_config(
    ctx: EvalContext,
    domain: str | None = None,
    task_id: str | None = None,
    task_split: str | None = None
) -> Tuple[str, str, List[str], List[str]]:
    """
    Get tau2-bench agent configuration from evaluation context.

    Extracts task parameters (either from args or from ctx.name), loads the task from tau2 registry,
    and returns configuration for both assistant and user agents.

    No dependency on global state - everything is loaded fresh from ctx.

    Args:
        ctx: HUD evaluation context
        domain: Domain name (airline, retail, telecom). If None, parsed from ctx.name
        task_id: Task ID. If None, parsed from ctx.name
            Note: For local tests with complex telecom task IDs like
            "[service_issue]airplane_mode_on[PERSONA:None]", it's recommended to pass
            this explicitly as the task_id gets sanitized in ctx.name
        task_split: Task split (base, dev, test, small). If None, parsed from ctx.name

    Returns:
        Tuple of (user_prompt, assistant_prompt, user_tools, assistant_tools)

    Raises:
        ValueError: If task has no user_scenario (should never happen with valid tau2 tasks)
    """
    from prompts.user_prompts import user_system_prompt
    from prompts.assistant_prompts import assistant_system_prompt
    from tau2.registry import registry
    from server.tools.http_client import get_http_client

    logger.critical("[CONFIG] ===== STARTING AGENT CONFIGURATION =====")

    # Get task parameters - prefer ctx task args (keeps telecom task_id intact)
    if domain is None or task_id is None or task_split is None:
        ctx_args = _extract_task_params_from_ctx(ctx)
        if ctx_args:
            domain = domain or ctx_args.get("domain")
            task_id = task_id or ctx_args.get("task_id")
            task_split = task_split or ctx_args.get("task_split")

        # Fall back to parsing from ctx.name if still missing
        if domain is None or task_id is None or task_split is None:
            # Parse from ctx.name
            # Format examples:
            #   Remote: "tau2-benchtau2-with-airline-10-base"
            #   Local: "tau2-with-telecom-service-issueairpla-small"
            ctx_name = ctx.name if hasattr(ctx, "name") else str(ctx)
            logger.critical(f"[CONFIG] ctx.name: {ctx_name}")

            # Pattern: tau2-with-{domain}-{task_id}-{split}
            # where split is one of: base, dev, test, small
            # and task_id can contain hyphens
            match = re.search(r"tau2-with-([^-]+)-(.+)-(base|dev|test|small)$", ctx_name)
            if not match:
                logger.critical(
                    "[CONFIG] ERROR: Could not parse task params from ctx.name: %s",
                    ctx_name,
                )
                raise ValueError(f"Could not parse task params from ctx.name: {ctx_name}")

            domain = domain or match.group(1)
            task_id = task_id or match.group(2)
            task_split = task_split or match.group(3)

    if domain is None or task_id is None or task_split is None:
        raise ValueError(
            f"Missing task params after parsing: domain={domain}, task_id={task_id}, split={task_split}"
        )

    logger.critical(f"[CONFIG] Using params: domain={domain}, task_id={task_id}, split={task_split}")

    try:
        # Load task directly from tau2 registry
        logger.critical("[CONFIG] Loading task from tau2 registry")
        task_loader = registry.get_tasks_loader(domain)
        tasks = task_loader(task_split)
        logger.critical(f"[CONFIG] Loaded {len(tasks)} tasks")

        # Find the specific task
        tau2_task = None
        for t in tasks:
            if str(t.id) == task_id:
                tau2_task = t
                break

        if tau2_task is None:
            logger.critical(f"[CONFIG] ERROR: Task {task_id} not found")
            logger.critical(f"[CONFIG] Available task IDs: {[str(t.id) for t in tasks[:10]]}")
            raise ValueError(f"Task {task_id} not found in domain {domain}")

        logger.critical(f"[CONFIG] Found task: {tau2_task.id}")
        logger.critical(f"[CONFIG] Task has user_scenario: {tau2_task.user_scenario is not None}")

        # Get policy and tools from environment server
        http_client = get_http_client()

        policy_source = "server"
        try:
            policy = http_client.get_policy()
            logger.critical(f"[CONFIG] Got policy from server (length: {len(policy)})")
        except Exception as e:
            logger.critical(f"[CONFIG] Could not get policy: {e}")
            policy = ""

        if not policy or len(policy.strip()) < 50 or "No policy" in policy:
            fallback_policy = _load_domain_policy_from_file(domain)
            if fallback_policy:
                policy = fallback_policy
                policy_source = "fallback_file"
                logger.critical(
                    "[CONFIG] Using fallback policy file for domain=%s (length: %d)",
                    domain,
                    len(policy),
                )
            else:
                policy = "No policy available"
                policy_source = "missing"

        logger.critical(
            "[CONFIG] Policy source=%s preview=%r",
            policy_source,
            policy[:200],
        )

        # Get tool names from environment
        try:
            # Clear any task-level tool filters (these can accidentally whitelist only transfer tools)
            if getattr(ctx, "_agent_include", None) is not None or getattr(ctx, "_agent_exclude", None) is not None:
                logger.critical(
                    "[CONFIG] Clearing ctx agent filters: include=%s exclude=%s",
                    getattr(ctx, "_agent_include", None),
                    getattr(ctx, "_agent_exclude", None),
                )
                ctx._agent_include = None
                ctx._agent_exclude = None

            # Clear connector include/exclude filters to fetch full tool list
            for connector in getattr(ctx, "_connections", {}).values():
                if getattr(connector, "config", None) is not None:
                    connector.config.include = None
                    connector.config.exclude = None

            tools_data = http_client.list_tools()
            agent_tool_names = [
                t["name"] for t in tools_data.get("tools", []) if t["name"] != "send_message"
            ]
            user_tool_names = [t["name"] for t in tools_data.get("user_tools", [])]

            # transfer_to_human_agents should never be a user tool
            if "transfer_to_human_agents" in user_tool_names:
                user_tool_names = [
                    name for name in user_tool_names if name != "transfer_to_human_agents"
                ]

            # If user tools are missing, fall back to scenario state (telecom has user tools)
            if not user_tool_names:
                try:
                    from server.state import get_tau2_task

                    tau2_task = get_tau2_task()
                    state_user_tools = getattr(tau2_task, "user_tool_names", None)
                    if state_user_tools:
                        user_tool_names = list(state_user_tools)
                        if "transfer_to_human_agents" in user_tool_names:
                            user_tool_names = [
                                name
                                for name in user_tool_names
                                if name != "transfer_to_human_agents"
                            ]
                        logger.critical(
                            "[CONFIG] Using user tools from scenario state (len=%d)",
                            len(user_tool_names),
                        )
                except Exception as e:
                    logger.critical(f"[CONFIG] Could not load user tools from state: {e}")

            # Prefer ctx tools (more reliable in HUD eval) and subtract user tools
            await ctx.list_tools()
            ctx_tool_names = [t.name for t in ctx.as_tools() if t.name != "send_message"]
            agent_tool_names_from_ctx = [
                name for name in ctx_tool_names if name not in user_tool_names
            ]
            if agent_tool_names_from_ctx:
                agent_tool_names = agent_tool_names_from_ctx
            else:
                logger.critical(
                    "[CONFIG] ctx tools empty after filtering; falling back to server tools."
                )
            logger.critical(f"[CONFIG] Got tools: agent={len(agent_tool_names)}, user={len(user_tool_names)}")
        except Exception as e:
            logger.critical(f"[CONFIG] Could not get tools: {e}")
            agent_tool_names = []
            user_tool_names = []

        # Build agent system prompt (match tau2-bench)
        assistant_prompt = assistant_system_prompt(policy)
        logger.critical(
            "[CONFIG] Assistant prompt length=%d preview=%r",
            len(assistant_prompt),
            assistant_prompt[:200],
        )

        # Build user system prompt
        user_scenario = getattr(tau2_task, "user_scenario", None)
        if not user_scenario:
            logger.critical("[CONFIG] ERROR: No user_scenario in task")
            raise ValueError(f"Task {task_id} has no user_scenario - this should never happen")

        logger.critical("[CONFIG] Building user system prompt")
        user_prompt = user_system_prompt(
            user_scenario=user_scenario,
            user_tool_names=user_tool_names,
        )
        logger.critical(f"[CONFIG] User system prompt length: {len(user_prompt)}")

        logger.critical("[CONFIG] ===== AGENT CONFIGURATION COMPLETE =====")
        logger.critical(f"[CONFIG] Returning: assistant_tools={len(agent_tool_names)}, user_tools={len(user_tool_names)}")

        # Return (user_prompt, assistant_prompt, user_tools, assistant_tools)
        return user_prompt, assistant_prompt, user_tool_names, agent_tool_names

    except Exception as e:
        logger.critical(f"[CONFIG] ===== EXCEPTION: {e} =====")
        import traceback
        logger.critical(f"[CONFIG] Traceback:\n{traceback.format_exc()}")
        raise
