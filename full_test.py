"""Local test script for the tau2-bench.
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

MAX_STEPS = 100  # Match tau2-bench default

# Use HUD inference gateway - see all models at https://hud.ai/models
# Get API key from environment or HUD settings
api_key = os.getenv("HUD_API_KEY") or os.getenv("OPENAI_API_KEY")
if not api_key:
    from hud.settings import settings
    api_key = settings.api_key

client = AsyncOpenAI(base_url="https://inference.hud.ai", api_key=api_key) if api_key else None


async def test_airline_scenario():
    """Test airline scenario with manual OpenAI calls."""
    print("\n=== Test 1: Airline Scenario ===")

    if not client:
        print("Skipping: API key not set. Set HUD_API_KEY or OPENAI_API_KEY environment variable.")
        return
    await init()
    tasks = load_tasks("/home/genteki/gentekis_document/hud/tai2-bench/make_data/datasets/airline/test_full.json")
    # Bind to local environment and run
    bound_tasks = [env(t.scenario, **t.args) for t in tasks]
    async with hud.eval(bound_tasks,max_concurrent=1) as ctx:
        agent = OpenAIChatAgent.create(model="gpt-5")
        await agent.run(ctx, max_steps=MAX_STEPS)


async def test_retail_scenario():
    """Test retail scenario with OpenAIChatAgent."""
    print("\n=== Test 2: Retail Scenario  ===")
    await init()
    tasks = load_tasks("/home/genteki/gentekis_document/hud/tai2-bench/make_data/datasets/retail/test_full.json")
    # Bind to local environment and run
    bound_tasks = [env(t.scenario, **t.args) for t in tasks[9:]]
    async with hud.eval(bound_tasks,max_concurrent=1) as ctx:
        agent = OpenAIChatAgent.create(model="gpt-5")
        await agent.run(ctx, max_steps=MAX_STEPS)

async def test_telecom_scenario():
    """Test telecom scenario with OpenAIChatAgent."""
    print("\n=== Test 3: Telecom Scenario ===")
    await init()
    tasks = load_tasks("/home/genteki/gentekis_document/hud/tai2-bench/make_data/datasets/telecom/test_full.json")
    # Bind to local environment and run
    bound_tasks = [env(t.scenario, **t.args) for t in tasks]
    async with hud.eval(bound_tasks,max_concurrent=1) as ctx:
        agent = OpenAIChatAgent.create(model="gpt-5")
        await agent.run(ctx, max_steps=MAX_STEPS)


async def main():
    # Read domain from environment variable
    domain = os.getenv("DOMAIN", "airline").lower()

    # Validate domain
    valid_domains = ["airline", "retail", "telecom"]
    if domain not in valid_domains:
        print(f"Error: Invalid domain '{domain}'. Must be one of: {', '.join(valid_domains)}")
        return

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
    print(f"Domain: {domain}")
    print(f"Server URL: {os.getenv('TAU2_SERVER_URL', 'http://127.0.0.1:8002')}")
    print("=" * 50)
    print("Make sure the environment server is running:")
    print(f"  python environment/run_server.py --domain {domain} --port <PORT>")
    print("=" * 50)
    print(f"Logging to: {log_file}")
    print("=" * 50)
    print()

    # Run only the selected domain test
    if domain == "airline":
        await test_airline_scenario()
    elif domain == "retail":
        await test_retail_scenario()
    elif domain == "telecom":
        await test_telecom_scenario()

if __name__ == "__main__":
    asyncio.run(main())
