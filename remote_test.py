"""Remote test for tau2-bench with HUD platform.

This script runs on HUD's remote platform using agents with proper tool filtering.
"""

import asyncio
import hud
from hud.agents import create_agent
from hud.datasets import load_tasks
from loop.multi_turn import multi_turn_run
from server.state import get_tau2_task
from prompts.user_prompts import user_system_prompt


async def main():
    ds = "TAU2-Test"
    model = "claude-haiku-4-5"
    tasks = load_tasks(ds)

    async with hud.eval(tasks, max_concurrent=10) as ctx:
        # Get agent configurations from tau2_task (populated during scenario setup)
        tau2_task = get_tau2_task()

        # Create assistant agent with agent tools and policy
        assistant_agent = create_agent(
            model=model,
            system_prompt=tau2_task.system_prompt,  # Includes policy
            allowed_tools=tau2_task.agent_tool_names  # Only agent tools
        )

        # Create user agent with user tools and scenario instructions
        user_agent = create_agent(
            model=model,
            system_prompt=user_system_prompt(
                user_scenario=tau2_task.user_scenario,
                user_tool_names=tau2_task.user_tool_names  # Auto-determines has_tools
            ),  # tau2 user simulation guidelines + scenario
            allowed_tools=tau2_task.user_tool_names  # Only user tools
        )

        # Run multi-turn conversation
        await multi_turn_run(
            ctx=ctx,
            agent=assistant_agent,
            simulated_user=user_agent,
            max_steps=30
        )


if __name__ == "__main__":
    asyncio.run(main())

