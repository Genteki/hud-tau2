"""Local test script for the tau2-bench environment.

Development workflow:
1. Start environment server: ./hud-tau2/scripts/start_environment_server.sh
2. Run tests: python local_test.py

The environment server provides domain tools via HTTP.
This script uses the local env module for scenarios/tools.
"""

import os
import asyncio
import logging
import hud
from env import env, init
from openai import AsyncOpenAI
from hud.agents import OpenAIChatAgent, create_agent
from loop.multi_turn import multi_turn_run
from loop.agent_config import configure_agents_for_tau2
from server.tools.conversation import get_user_simulator

# Use HUD inference gateway
api_key = os.getenv("HUD_API_KEY") or os.getenv("OPENAI_API_KEY")
if not api_key:
    from hud.settings import settings
    api_key = settings.api_key

client = AsyncOpenAI(base_url="https://inference.hud.ai", api_key=api_key) if api_key else None


async def test_tools_standalone():
    """Test environment tools directly."""
    print("=== Test 1: Standalone Tools ===")

    # Initialize environment
    await init()
    tools_dict = await env.get_tools()
    tool_names = list(tools_dict.keys())
    print(f"Tools: {tool_names}")
    print(f"Total tools: {len(tools_dict)}")

async def test_telecom():
    print("\n=== Test: Multi-turn conversation with agent and user ===")
    # Initialize the environment first
    await init()

    task = env(
        "tau2",
        domain="telecom",
        task_id="[service_issue]airplane_mode_on[PERSONA:None]",
        task_split="small"  # Use tasks_small.json which is valid
    )
    bound_tasks = [task]

    async with hud.eval(bound_tasks, max_concurrent=1) as ctx:
        assistant_agent = OpenAIChatAgent.create(model="gpt-4o")
        user_agent = OpenAIChatAgent.create(model="gpt-4o")
        await configure_agents_for_tau2(ctx, assistant_agent, user_agent)
        await multi_turn_run(ctx, assistant_agent, user_agent, max_steps=30)

async def test_airline():
    print("\n=== Test: Multi-turn conversation with agent and user ===")
    # Initialize the environment first
    await init()

    task = env(
        "tau2",
        domain="airline",
        task_id="1",
        task_split="base"  # Use tasks_small.json which is valid
    )
    bound_tasks = [task]

    async with hud.eval(bound_tasks, max_concurrent=1) as ctx:
        assistant_agent = OpenAIChatAgent.create(model="gpt-4o")
        user_agent = OpenAIChatAgent.create(model="gpt-4o")
        await configure_agents_for_tau2(ctx, assistant_agent, user_agent)
        await multi_turn_run(ctx, assistant_agent, user_agent, max_steps=30)


async def main():
    # Set up file logging for debugging (all output goes to log file)
    log_file = "log"
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s | %(levelname)-7s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))

    # Add file handler to root logger to capture everything
    logging.getLogger().addHandler(file_handler)

    # Also add to loguru (tau2's logger) for file output
    from loguru import logger as tau2_logger
    tau2_logger.add(log_file, level="DEBUG", format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}")

    # await test_tools_standalone()
    # await test_telecom()
    await test_airline()

if __name__ == "__main__":
    asyncio.run(main())
