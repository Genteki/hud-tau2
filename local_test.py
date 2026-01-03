"""Local test script for the tau2-bench environment.

Development workflow:
1. Start environment server: ./hud-tau2/scripts/start_environment_server.sh
2. Run tests: python local_test.py

The environment server provides domain tools via HTTP.
This script uses the local env module for scenarios/tools.
"""
import asyncio
import os
import logging

import hud
from hud.agents import OpenAIChatAgent
from hud.datasets import load_tasks
from openai import AsyncOpenAI

from env import env
from env import init

# Use HUD inference gateway - see all models at https://hud.ai/models
# Get API key from environment or HUD settings
api_key = os.getenv("HUD_API_KEY") or os.getenv("OPENAI_API_KEY")
if not api_key:
    from hud.settings import settings
    api_key = settings.api_key

client = AsyncOpenAI(base_url="https://inference.hud.ai", api_key=api_key) if api_key else None


async def test_tools_standalone():
    """Test environment tools directly."""
    print("=== Test 1: Standalone Tools ===")

    # Initialize environment
    from env import init
    await init()

    # Get tools (async method)
    tools_dict = await env.get_tools()
    tool_names = list(tools_dict.keys())
    print(f"Tools: {tool_names}")
    print(f"Total tools: {len(tools_dict)}")


async def test_airline_scenario():
    """Test airline scenario with manual OpenAI calls."""
    print("\n=== Test 2: Airline Scenario (Manual Agent Loop) ===")

    if not client:
        print("Skipping: API key not set. Set HUD_API_KEY or OPENAI_API_KEY environment variable.")
        return
    await init()
    tasks = load_tasks("/home/genteki/gentekis_document/hud/tai2-bench/make_data/datasets/airline/test10.json")
    # Bind to local environment and run
    bound_tasks = [env(t.scenario, **t.args) for t in tasks]
    async with hud.eval(bound_tasks,max_concurrent=1) as ctx:
        agent = OpenAIChatAgent.create(model="gpt-5")
        await agent.run(ctx, max_steps=100)  # Match tau2-bench default


async def test_retail_scenario():
    """Test retail scenario with OpenAIChatAgent."""
    print("\n=== Test 2: Retail Scenario 10 tasks ===")
    await init()
    tasks = load_tasks("/home/genteki/gentekis_document/hud/tai2-bench/make_data/datasets/retail/test10.json")
    # Bind to local environment and run
    bound_tasks = [env(t.scenario, **t.args) for t in tasks[9:]]
    async with hud.eval(bound_tasks,max_concurrent=1) as ctx:
        agent = OpenAIChatAgent.create(model="gpt-5")
        await agent.run(ctx, max_steps=100)  # Match tau2-bench default

async def test_telecom_scenario():
    """Test telecom scenario with OpenAIChatAgent."""
    print("\n=== Test 4: Telecom Scenario 10 tasks (OpenAIChatAgent) ===")
    await init()
    tasks = load_tasks("/home/genteki/gentekis_document/hud/tai2-bench/make_data/datasets/telecom/test10.json")
    # Bind to local environment and run
    bound_tasks = [env(t.scenario, **t.args) for t in tasks]
    async with hud.eval(bound_tasks,max_concurrent=1) as ctx:
        agent = OpenAIChatAgent.create(model="gpt-5")
        await agent.run(ctx, max_steps=100)  # Match tau2-bench default

async def test_from_json():
    """Load tasks from JSON and run locally."""
    print("\n=== Test: Load from JSON ===")


    # Initialize the environment first
    await init()

    tasks = load_tasks("local_tasks.json")

    # Bind to local environment and run
    bound_tasks = [env(t.scenario, **t.args) for t in tasks]

    async with hud.eval(bound_tasks) as ctx:
        from hud.agents.claude import ClaudeAgent
        agent = ClaudeAgent.create(
            model="claude-haiku-4-5",
        )
        await multi_agent_loop(ctx, agent, max_steps=30)

async def test_hud_run():
    print("\n=== Test: Load from JSON ===")
    # Initialize the environment first
    await init()

    tasks = load_tasks("local_tasks.json")

    # Bind to local environment and run
    bound_tasks = [env(t.scenario, **t.args) for t in tasks]

    async with hud.eval(bound_tasks,max_concurrent=1) as ctx:
        agent = OpenAIChatAgent.create(model="gpt-5")
        # from hud.agents.claude import ClaudeAgent
        # agent = ClaudeAgent.create(
        #     model="claude-haiku-4-5",
        # )
        await agent.run(ctx, max_steps=100)  # Match tau2-bench default


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

    print("TAU2-Bench Environment - Local Test")
    print("=" * 50)
    print("Make sure the environment server is running:")
    print("  ./hud-tau2/scripts/start_environment_server.sh")
    print("=" * 50)
    print(f"Logging to: {log_file}")
    print("=" * 50)
    print()

    # Test loading from JSON (recommended)

    # Or test individual scenarios:
    # await test_tools_standalone()
    # await test_airline_scenario()
    # await test_retail_scenario()
    # await test_telecom_scenario()
    await test_hud_run()

if __name__ == "__main__":
    asyncio.run(main())
