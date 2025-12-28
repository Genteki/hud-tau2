"""Remote test script for tau2-bench environment.

This script demonstrates:
1. Loading tasks from remote_tasks.json
2. Loading tasks from the HUD platform by slug

Usage:
    python remote_test.py
"""

import asyncio

import hud
from hud.agents import OpenAIChatAgent
from hud.datasets import load_tasks

from env import env
from server.agent_loop import multi_agent_loop


async def test_from_json():
    """Load tasks from remote_tasks.json and run locally.

    This binds remote task definitions to the local environment
    for testing before deploying to the platform.
    """
    print("=== Test 1: Load from JSON (Local Binding) ===")

    # Initialize the environment first
    from env import init
    await init()

    tasks = load_tasks("remote_tasks.json")

    # Bind to local environment (ignores the "env" field in JSON)
    bound_tasks = [env(t.scenario, **t.args) for t in tasks]

    async with hud.eval(bound_tasks) as ctx:
        agent = OpenAIChatAgent.create(model="gpt-4o")
        await multi_agent_loop(ctx, agent, max_steps=30)


async def test_from_platform(slug: str = "Genteki/tau2-bench-tiny"):
    """Load and run tasks from the HUD platform locally.

    This binds platform tasks to the local environment for testing.
    """
    print(f"=== Test 2: Load from Platform: {slug} ===")

    # Initialize the environment first
    from env import init
    await init()

    tasks = load_tasks(slug)

    # Bind to local environment (ignores the "env" field in tasks)
    bound_tasks = [env(t.scenario, **t.args) for t in tasks]

    async with hud.eval(bound_tasks) as ctx:
        agent = OpenAIChatAgent.create(model="gpt-4o")
        await multi_agent_loop(ctx, agent, max_steps=30)


async def main():
    await test_from_json()
    # await test_from_platform(slug="Genteki/tau2-bench-tiny")


if __name__ == "__main__":
    asyncio.run(main())
