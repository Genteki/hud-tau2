"""Remote test for tau2-bench with HUD platform.

This script runs on HUD's remote platform using agents with proper tool filtering.
"""

import asyncio
import logging
import hud
from hud.agents import create_agent
from hud.datasets import load_tasks
from loop.multi_turn import multi_turn_run
from loop.agent_config import get_tau2_config

# Use CRITICAL level to ensure it appears in remote logs
logger = logging.getLogger(__name__)
logger.critical("[REMOTE_TEST] ===== REMOTE_TEST.PY STARTING =====")

assistant_model = "gpt-5"
user_model = "gpt-5"

async def main():
    logger.critical("[REMOTE_TEST] main() function called")
    ds = "TAU2-Test"
    tasks = load_tasks(ds)

    async with hud.eval(tasks, max_concurrent=30) as ctx:
        # Get tau2 configuration
        user_prompt, assistant_prompt, user_tools, assistant_tools = await get_tau2_config(ctx)
        logger.critical("[REMOTE_TEST] Assistant prompt length=%d", len(assistant_prompt))
        logger.critical("[REMOTE_TEST] Assistant prompt preview=%r", assistant_prompt[:400])
        # logger.critical("[REMOTE_TEST] Assistant prompt full=%r", assistant_prompt)
        logger.critical("[REMOTE_TEST] User prompt length=%d", len(user_prompt))
        logger.critical("[REMOTE_TEST] User prompt preview=%r", user_prompt[:400])
        # logger.critical("[REMOTE_TEST] User prompt full=%r", user_prompt)
        
        # Create agents
        assistant_agent = create_agent(
            model=assistant_model,
            system_prompt=assistant_prompt,
            allowed_tools=assistant_tools
        )
        user_agent = create_agent(
            model=user_model,
            system_prompt=user_prompt,
            allowed_tools=user_tools,
        )

        # Run multi-turn conversation
        await multi_turn_run(
            ctx=ctx,
            agent=assistant_agent,
            simulated_user=user_agent,
            max_steps=100
        )


if __name__ == "__main__":
    asyncio.run(main())
