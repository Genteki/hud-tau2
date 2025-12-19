"""Local test script for the tau2-bench environment.

Development workflow:
1. Start environment server: ./hud-tau2/scripts/start_environment_server.sh
2. Run tests: python local_test.py

The environment server provides domain tools via HTTP.
This script uses the local env module for scenarios/tools.
"""
import asyncio
import os

import hud
from hud.agents import OpenAIChatAgent
from openai import AsyncOpenAI

from env import env

# Use HUD inference gateway - see all models at https://hud.ai/models
# Get API key from environment or HUD settings
api_key = os.getenv("HUD_API_KEY") or os.getenv("OPENAI_API_KEY")
if not api_key:
    try:
        from hud.settings import settings
        api_key = settings.api_key
    except:
        pass

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

    task = env("tau2",
        domain="airline",
        task_id=0,
        task_split="base"
    )

    async with hud.eval(task) as ctx:
        messages = [{"role": "user", "content": ctx.prompt}]

        while True:
            response = await client.chat.completions.create(
                model="gpt-4o",  # https://hud.ai/models
                messages=messages,
                tools=ctx.as_openai_chat_tools(),
            )
            msg = response.choices[0].message

            if not msg.tool_calls:
                break

            messages.append(msg)
            for tc in msg.tool_calls:
                result = await ctx.call_tool(tc)
                messages.append(result)


async def test_retail_scenario():
    """Test retail scenario with OpenAIChatAgent."""
    print("\n=== Test 3: Retail Scenario (OpenAIChatAgent) ===")

    task = env("tau2",
        domain="retail",
        task_id=0,
        task_split="base"
    )

    async with hud.eval(task) as ctx:
        agent = OpenAIChatAgent.create(model="gpt-4o")  # https://hud.ai/models
        await agent.run(ctx)


async def test_telecom_scenario():
    """Test telecom scenario with OpenAIChatAgent."""
    print("\n=== Test 4: Telecom Scenario (OpenAIChatAgent) ===")

    task = env("tau2",
        domain="telecom",
        task_id=0,
        task_split="base"
    )

    async with hud.eval(task) as ctx:
        agent = OpenAIChatAgent.create(model="gpt-4o")  # https://hud.ai/models
        await agent.run(ctx)


async def main():
    print("TAU2-Bench Environment - Local Test")
    print("=" * 50)
    print("Make sure the environment server is running:")
    print("  ./hud-tau2/scripts/start_environment_server.sh")
    print("=" * 50)
    print()

    await test_tools_standalone()
    # Uncomment to run scenarios:
    await test_airline_scenario()
    # await test_retail_scenario()
    # await test_telecom_scenario()


if __name__ == "__main__":
    asyncio.run(main())
