"""Remote test for tau2-bench with HUD platform.

This script runs on HUD's remote platform using agents with proper tool filtering.
"""

import asyncio
import hud
from hud.agents import create_agent
from hud.datasets import load_tasks
from loop.multi_turn import multi_turn_run
from loop.agent_config import configure_agents_for_tau2


async def main():
    ds = "TAU2-Test"
    model = "claude-haiku-4-5"
    tasks = load_tasks(ds)

    async with hud.eval(tasks[2:3], max_concurrent=1) as ctx:
        # Create agents with model only
        assistant_agent = create_agent(model=model)
        user_agent = create_agent(model=model)

        # Configure agents with tau2-specific system prompts and tool filters
        await configure_agents_for_tau2(ctx, assistant_agent, user_agent)

        # Run multi-turn conversation
        await multi_turn_run(
            ctx=ctx,
            agent=assistant_agent,
            simulated_user=user_agent,
            max_steps=30
        )


if __name__ == "__main__":
    asyncio.run(main())

