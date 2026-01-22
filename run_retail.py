"""Remote test for tau2-bench with HUD platform.

This script runs on HUD's remote platform using agents with proper tool filtering.
"""

import asyncio
import hud
from hud.agents import create_agent
from hud.datasets import load_tasks
from loop.multi_turn import multi_turn_run
from loop.agent_config import get_tau2_config

ds = "TAU2-Retail"
assistant_model = "gpt-5"
user_model = "gpt-4o"
max_concurrent = 30
max_steps = 200


async def main():

    tasks = load_tasks(ds)

    async with hud.eval(tasks, max_concurrent=max_concurrent) as ctx:
        # Get tau2 configuration
        user_prompt, assistant_prompt, user_tools, assistant_tools = (
            await get_tau2_config(ctx)
        )

        # Create agents with proper configuration
        assistant_agent = create_agent(
            model=assistant_model,
            system_prompt=assistant_prompt,
            allowed_tools=assistant_tools,
        )
        user_agent = create_agent(
            model=user_model,
            system_prompt=user_prompt,
            allowed_tools=user_tools,
        )
        setattr(assistant_agent, "temperature", 0.0)
        setattr(user_agent, "temperature", 0.0)
        # Run multi-turn conversation
        await multi_turn_run(
            ctx=ctx, agent=assistant_agent, simulated_user=user_agent, max_steps=max_steps
        )


if __name__ == "__main__":
    asyncio.run(main())
