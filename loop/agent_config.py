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


STATIC_USER_TOOLS_BY_DOMAIN = {
    "telecom": [
        "check_status_bar",
        "check_network_status",
        "check_network_mode_preference",
        "set_network_mode_preference",
        "run_speed_test",
        "toggle_airplane_mode",
        "check_sim_status",
        "reseat_sim_card",
        "toggle_data",
        "toggle_roaming",
        "check_data_restriction_status",
        "toggle_data_saver_mode",
        "check_apn_settings",
        "set_apn_settings",
        "reset_apn_settings",
        "check_wifi_status",
        "toggle_wifi",
        "check_wifi_calling_status",
        "toggle_wifi_calling",
        "check_vpn_status",
        "connect_vpn",
        "disconnect_vpn",
        "check_installed_apps",
        "check_app_status",
        "check_app_permissions",
        "grant_app_permission",
        "can_send_mms",
        "reboot_device",
        "check_payment_request",
        "make_payment",
    ],
    "telecom-workflow": [
        "check_status_bar",
        "check_network_status",
        "check_network_mode_preference",
        "set_network_mode_preference",
        "run_speed_test",
        "toggle_airplane_mode",
        "check_sim_status",
        "reseat_sim_card",
        "toggle_data",
        "toggle_roaming",
        "check_data_restriction_status",
        "toggle_data_saver_mode",
        "check_apn_settings",
        "set_apn_settings",
        "reset_apn_settings",
        "check_wifi_status",
        "toggle_wifi",
        "check_wifi_calling_status",
        "toggle_wifi_calling",
        "check_vpn_status",
        "connect_vpn",
        "disconnect_vpn",
        "check_installed_apps",
        "check_app_status",
        "check_app_permissions",
        "grant_app_permission",
        "can_send_mms",
        "reboot_device",
        "check_payment_request",
        "make_payment",
    ],
}

STATIC_AGENT_TOOLS_BY_DOMAIN = {
    "telecom": [
        "get_customer_by_phone",
        "get_customer_by_id",
        "get_customer_by_name",
        "get_details_by_id",
        "suspend_line",
        "resume_line",
        "get_bills_for_customer",
        "send_payment_request",
        "get_data_usage",
        "enable_roaming",
        "disable_roaming",
        "refuel_data",
        "transfer_to_human_agents",
    ],
    "telecom-workflow": [
        "get_customer_by_phone",
        "get_customer_by_id",
        "get_customer_by_name",
        "get_details_by_id",
        "suspend_line",
        "resume_line",
        "get_bills_for_customer",
        "send_payment_request",
        "get_data_usage",
        "enable_roaming",
        "disable_roaming",
        "refuel_data",
        "transfer_to_human_agents",
    ],
}


async def get_tau2_config(
    ctx: EvalContext,
    domain: str | None = None,
    task_id: str | None = None,
    task_split: str | None = None,
) -> Tuple[str, str, List[str], List[str]]:
    from prompts.assistant_prompts import assistant_system_prompt
    from prompts.user_prompts import user_system_prompt
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

    policy = _load_policy_from_file(domain) or "No policy available"

    # Clear agent filters that can collapse tools
    if getattr(ctx, "_agent_include", None) is not None or getattr(ctx, "_agent_exclude", None) is not None:
        ctx._agent_include = None
        ctx._agent_exclude = None
    for connector in getattr(ctx, "_connections", {}).values():
        if getattr(connector, "config", None) is not None:
            connector.config.include = None
            connector.config.exclude = None

    await ctx.list_tools()
    ctx_tool_names = [t.name for t in ctx.as_tools() if t.name != "send_message"]

    user_tools = getattr(tau2_task, "user_tool_names", None) or []
    if not user_tools:
        user_tools = STATIC_USER_TOOLS_BY_DOMAIN.get(domain, [])
    user_tools = [t for t in user_tools if t in ctx_tool_names and t != "transfer_to_human_agents"]

    agent_tools = STATIC_AGENT_TOOLS_BY_DOMAIN.get(domain, [])
    agent_tools = [t for t in agent_tools if t in ctx_tool_names]
    if not agent_tools:
        agent_tools = [t for t in ctx_tool_names if t not in user_tools]

    if not tau2_task.user_scenario:
        raise ValueError(f"Task {task_id} has no user_scenario")

    assistant_prompt = assistant_system_prompt(policy)
    user_prompt = user_system_prompt(tau2_task.user_scenario, user_tools)

    return user_prompt, assistant_prompt, user_tools, agent_tools
