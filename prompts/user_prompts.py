"""User agent system prompt generation.

This module creates system prompts for the user agent that are identical
to tau2-bench's UserSimulator system prompts.
"""

from tau2.user.user_simulator import UserSimulator
import tau2.user.user_simulator as us_module

# System prompt template (matches tau2-bench's SYSTEM_PROMPT exactly)
USER_SYSTEM_PROMPT = """
{global_user_sim_guidelines}

<scenario>
{instructions}
</scenario>
""".strip()


def user_system_prompt(user_scenario, has_tools: bool = True) -> str:
    """
    Build user agent system prompt identical to tau2-bench's UserSimulator.

    Args:
        user_scenario: UserScenario object from tau2_task.task.user_scenario
        has_tools: Whether the user has tools available (affects guidelines)

    Returns:
        System prompt string matching tau2-bench's UserSimulator
    """
    # Create a temporary UserSimulator to get the guidelines
    # (guidelines differ based on whether tools are available)
    class TempSimulator(UserSimulator):
        def __init__(self, tools):
            self.tools = tools

    # Pass dummy tools list to indicate tools are available
    sim = TempSimulator(tools=["dummy"] if has_tools else None)
    global_guidelines = sim.global_simulation_guidelines

    # Format instructions from user scenario
    # This matches how tau2-bench's UserSimulator formats instructions
    instructions = user_scenario.instructions

    # Build the system prompt using tau2's template
    system_prompt = USER_SYSTEM_PROMPT.format(
        global_user_sim_guidelines=global_guidelines,
        instructions=instructions,
    )

    return system_prompt