"""Remote test for tau2-bench with HUD platform.

This script runs on HUD's remote platform using agents with proper tool filtering.
"""

import asyncio
import hud
from hud.agents import create_agent
from hud.datasets import load_tasks
from loop.multi_turn import multi_turn_run


async def main():
    ds = "TAU2-Test"
    model = "claude-haiku-4-5"
    tasks = load_tasks(ds)

    async with hud.eval(tasks, max_concurrent=10) as ctx:
        # Create agents with model only - tool filtering happens in multi_turn_run
        # after scenario setup populates tau2_task
        assistant_agent = create_agent(model=model)
        user_agent = create_agent(model=model)

        # Run multi-turn conversation
        # This will configure agents with proper system prompts and tools
        # after scenario setup runs
        await multi_turn_run(
            ctx=ctx,
            agent=assistant_agent,
            simulated_user=user_agent,
            max_steps=30
        )


if __name__ == "__main__":
    asyncio.run(main())

