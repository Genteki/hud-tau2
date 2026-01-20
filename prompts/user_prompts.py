"""User agent system prompt generation.

This module creates system prompts for the user agent that are identical
to tau2-bench's UserSimulator system prompts.
"""

from tau2.user.user_simulator import UserSimulator


# System prompt template (matches tau2-bench's SYSTEM_PROMPT exactly)
USER_SYSTEM_PROMPT = """
{global_user_sim_guidelines}

<scenario>
{instructions}
</scenario>
""".strip()


def user_system_prompt(user_scenario, user_tool_names=None) -> str:
    """
    Build user agent system prompt identical to tau2-bench's UserSimulator.

    Args:
        user_scenario: UserScenario object from tau2_task.task.user_scenario
        user_tool_names: List of user tool names (if None, automatically determined)

    Returns:
        System prompt string matching tau2-bench's UserSimulator
    """
    # Determine if tools are available
    # If user_tool_names is provided, check if non-empty
    # Otherwise, assume no tools
    if user_tool_names is None:
        from server.state import get_tau2_task
        tau2_task = get_tau2_task()
        user_tool_names = tau2_task.user_tool_names if hasattr(tau2_task, 'user_tool_names') else []

    has_tools = bool(user_tool_names and len(user_tool_names) > 0)

    # Create a temporary UserSimulator to get the guidelines
    # (guidelines differ based on whether tools are available)
    class TempSimulator(UserSimulator):
        def __init__(self, tools):
            self.tools = tools

    # Pass dummy tools list to indicate tools are available
    sim = TempSimulator(tools=["dummy"] if has_tools else None)
    global_guidelines = sim.global_simulation_guidelines

    # Format instructions from user scenario
    # Use str(user_scenario) to get the formatted string with BOTH persona and instructions
    # This matches exactly how tau2-bench's UserSimulator formats the system prompt
    instructions = str(user_scenario)

    # Build the system prompt using tau2's template
    system_prompt = USER_SYSTEM_PROMPT.format(
        global_user_sim_guidelines=global_guidelines,
        instructions=instructions,
    )

    return system_prompt
