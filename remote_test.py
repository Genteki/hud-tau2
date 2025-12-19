"""Remote test script for tau2-bench environment via HUD Gateway.

This script connects to the hud-tau2 MCP server remotely and runs evaluations.

Workflow:
1. Import the local env module (which connects to MCP server)
2. Create tasks using env()
3. Run evaluations with hud.eval()

Usage:
    python remote_test.py

Requirements:
    - MCP server running (via Docker or local)
    - Environment server running on port 8002
    - HUD API key set in environment
"""
import asyncio
import os

import hud
from hud.agents import OpenAIChatAgent

# Import local environment (this will connect to MCP server)
from env import env


async def test_airline_remote():
    """Test airline scenario remotely."""
    print("=== Remote Test 1: Airline Scenario ===")

    # Create task using the env that's already connected to MCP
    task = env("tau2", domain="airline", task_id=0, task_split="base")

    async with hud.eval(task) as ctx:
        if ctx.prompt:
            print(f"Prompt: {ctx.prompt[:100]}...")
        else:
            print("Prompt: (scenario will provide prompt)")

        # Use an agent to run the task
        agent = OpenAIChatAgent.create(model="claude-sonnet-4-5")
        await agent.run(ctx, max_steps=30)


async def test_telecom_remote():
    """Test telecom scenario remotely."""
    print("\n=== Remote Test 2: Telecom Scenario ===")

    # Create task using the env that's already connected to MCP
    task = env(
        "tau2",
        domain="telecom",
        task_id="[mobile_data_issue]airplane_mode_on|user_abroad_roaming_enabled_off[PERSONA:None]",
        task_split="base"
    )

    async with hud.eval(task) as ctx:
        if ctx.prompt:
            print(f"Prompt: {ctx.prompt[:100]}...")
        else:
            print("Prompt: (scenario will provide prompt)")

        # Use an agent to run the task
        agent = OpenAIChatAgent.create(model="claude-sonnet-4-5")
        await agent.run(ctx, max_steps=30)


async def test_retail_remote():
    """Test retail scenario remotely."""
    print("\n=== Remote Test 3: Retail Scenario ===")

    # Create task using the env that's already connected to MCP
    task = env("tau2", domain="retail", task_id=0, task_split="base")

    async with hud.eval(task) as ctx:
        if ctx.prompt:
            print(f"Prompt: {ctx.prompt[:100]}...")
        else:
            print("Prompt: (scenario will provide prompt)")

        # Use an agent to run the task
        agent = OpenAIChatAgent.create(model="gpt-4o")
        await agent.run(ctx, max_steps=40)


async def test_manual_tool_calls():
    """Test manual tool calls without an agent."""
    print("\n=== Remote Test 4: Manual Tool Calls ===")

    # Create task using the env that's already connected to MCP
    task = env(
        "tau2",
        domain="telecom",
        task_id="[mobile_data_issue]airplane_mode_on|user_abroad_roaming_enabled_off[PERSONA:None]",
        task_split="base"
    )

    async with hud.eval(task) as ctx:
        if ctx.prompt:
            print(f"Scenario prompt: {ctx.prompt[:100]}...")
        else:
            print("Scenario prompt: (scenario will provide prompt)")

        # Get available tools
        tools = await ctx.get_tools()
        print(f"\nAvailable tools: {list(tools.keys())[:5]}...")  # Show first 5 tools

        # Example: Send a message to the user
        try:
            response = await ctx.call_tool("send_message", message="Hello! How can I help you today?")
            print(f"\nUser response: {response}")
        except Exception as e:
            print(f"Error calling tool: {e}")

        # Submit empty answer to end evaluation
        await ctx.submit("")


async def main():
    print("TAU2-Bench Remote Test via HUD Gateway")
    print("=" * 60)
    print("Make sure:")
    print("  1. MCP server is running (docker or local)")
    print("  2. Environment server is running on port 8002")
    print("  3. HUD_API_KEY or OPENAI_API_KEY is set")
    print("=" * 60)
    print()

    # Initialize the environment
    from env import init
    await init()
    print("Environment initialized\n")

    # Check for API key
    api_key = os.getenv("HUD_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("WARNING: No API key found. Set HUD_API_KEY or OPENAI_API_KEY.")
        print()

    # Run tests
    # Uncomment the tests you want to run:

    await test_airline_remote()
    await test_retail_remote()
    # await test_telecom_remote()
    # await test_manual_tool_calls()

    print("\n" + "=" * 60)
    print("Remote tests completed!")


if __name__ == "__main__":
    asyncio.run(main())
