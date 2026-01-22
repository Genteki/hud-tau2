"""Agent configuration helper for tau2-bench tasks."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, List, Tuple

from hud.eval.context import EvalContext

logger = logging.getLogger(__name__)


def _extract_task_params(ctx: EvalContext) -> dict[str, Any]:
    task = getattr(ctx, "_task", None)
    return getattr(task, "args", {}) if isinstance(getattr(task, "args", None), dict) else {}


def _parse_ctx_name(ctx: EvalContext) -> dict[str, str]:
    ctx_name = ctx.name if hasattr(ctx, "name") else str(ctx)
    match = re.search(r"tau2-with-([^-]+)-(.+)-(base|dev|test|small)$", ctx_name)
    if not match:
        raise ValueError(f"Could not parse task params from ctx.name: {ctx_name}")
    return {"domain": match.group(1), "task_id": match.group(2), "task_split": match.group(3)}


def _load_policy_from_file(domain: str) -> str | None:
    try:
        if domain == "airline":
            from tau2.domains.airline.utils import AIRLINE_POLICY_PATH as path
        elif domain == "retail":
            from tau2.domains.retail.utils import RETAIL_POLICY_PATH as path
        elif domain == "telecom":
            from tau2.domains.telecom.utils import TELECOM_MAIN_POLICY_PATH as path
        else:
            return None
        return path.read_text(encoding="utf-8") if isinstance(path, Path) and path.exists() else None
    except Exception:
        return None


def _policy_matches_domain(policy: str, domain: str) -> bool:
    policy = policy.lower()
    return f"{domain} agent policy" in policy


async def get_tau2_config(
    ctx: EvalContext,
    domain: str | None = None,
    task_id: str | None = None,
    task_split: str | None = None,
) -> Tuple[str, str, List[str], List[str]]:
    from prompts.assistant_prompts import assistant_system_prompt
    from prompts.user_prompts import user_system_prompt
    from server.tools.http_client import get_http_client
    from tau2.registry import registry

    params = {**_extract_task_params(ctx)}
    if domain or task_id or task_split:
        params.update({k: v for k, v in {"domain": domain, "task_id": task_id, "task_split": task_split}.items() if v})
    if not all(params.get(k) for k in ("domain", "task_id", "task_split")):
        params.update(_parse_ctx_name(ctx))

    domain = params.get("domain")
    task_id = params.get("task_id")
    task_split = params.get("task_split")
    if not domain or not task_id or not task_split:
        raise ValueError(
            f"Missing task params after parsing: domain={domain}, task_id={task_id}, split={task_split}"
        )

    task_loader = registry.get_tasks_loader(domain)
    tasks = task_loader(task_split)
    tau2_task = next((t for t in tasks if str(t.id) == str(task_id)), None)
    if tau2_task is None:
        raise ValueError(f"Task {task_id} not found in domain {domain}")

    http_client = get_http_client()
    try:
        policy = http_client.get_policy()
    except Exception:
        policy = ""
    if not policy or len(policy.strip()) < 50 or "no policy" in policy.lower() or not _policy_matches_domain(policy, domain):
        policy = _load_policy_from_file(domain) or "No policy available"

    # Clear agent filters that can collapse tools
    if getattr(ctx, "_agent_include", None) is not None or getattr(ctx, "_agent_exclude", None) is not None:
        ctx._agent_include = None
        ctx._agent_exclude = None
    for connector in getattr(ctx, "_connections", {}).values():
        if getattr(connector, "config", None) is not None:
            connector.config.include = None
            connector.config.exclude = None

    tools_data = http_client.list_tools()
    agent_tools = [t["name"] for t in tools_data.get("tools", []) if t["name"] != "send_message"]
    user_tools = [t["name"] for t in tools_data.get("user_tools", [])]
    user_tools = [t for t in user_tools if t != "transfer_to_human_agents"]

    if not user_tools:
        try:
            from server.state import get_tau2_task

            state_user_tools = getattr(get_tau2_task(), "user_tool_names", None)
            if state_user_tools:
                user_tools = [t for t in state_user_tools if t != "transfer_to_human_agents"]
        except Exception:
            pass

    await ctx.list_tools()
    ctx_tool_names = [t.name for t in ctx.as_tools() if t.name != "send_message"]
    agent_tools = [t for t in ctx_tool_names if t not in user_tools] or agent_tools

    if not tau2_task.user_scenario:
        raise ValueError(f"Task {task_id} has no user_scenario")

    assistant_prompt = assistant_system_prompt(policy)
    user_prompt = user_system_prompt(tau2_task.user_scenario, user_tools)

    return user_prompt, assistant_prompt, user_tools, agent_tools
