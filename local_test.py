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
from hud.agents import OpenAIAgent, create_agent
from hud.agents.gateway import build_gateway_client
from loop.multi_turn import multi_turn_run
from typing import Awaitable, Callable, cast

logger = logging.getLogger(__name__)

# Use HUD inference gateway
api_key = os.getenv("HUD_API_KEY") or os.getenv("OPENAI_API_KEY")
if not api_key:
    from hud.settings import settings
    api_key = settings.api_key

client = AsyncOpenAI(base_url="https://inference.hud.ai", api_key=api_key) if api_key else None
gateway_openai = build_gateway_client("openai")

init_env = cast(Callable[[], Awaitable[None]], init)

assistant_model = "gpt-5"
user_model = "gpt-5"

async def test_tools_standalone():
    """Test environment tools directly."""
    print("=== Test 1: Standalone Tools ===")

    # Initialize environment
    await init_env()
    tools_dict = await env.get_tools()
    tool_names = list(tools_dict.keys())
    print(f"Tools: {tool_names}")
    print(f"Total tools: {len(tools_dict)}")

async def test_telecom():
    print("\n=== Test: Multi-turn conversation with agent and user ===")
    # Initialize the environment first
    await init_env()

    domain = "telecom"
    task_id = "[mms_issue]airplane_mode_on|bad_network_preference|bad_wifi_calling|break_apn_mms_setting|break_app_sms_permission|data_mode_off|data_usage_exceeded|unseat_sim_card|user_abroad_roaming_disabled_on[PERSONA:None]"
    task_split = "base"

    task = env(
        "tau2",
        domain=domain,
        task_id=task_id,
        task_split=task_split
    )
    bound_tasks = [task]

    async with hud.eval(bound_tasks, max_concurrent=1) as ctx:
        # Get tau2 configuration - pass params explicitly to avoid parsing issues
        from loop.agent_config import get_tau2_config
        user_prompt, assistant_prompt, user_tools, assistant_tools = await get_tau2_config(
            ctx, domain=domain, task_id=task_id, task_split=task_split
        )
        print("\n=== SYSTEM PROMPTS (telecom) ===")
        print("\n[USER PROMPT]\n")
        print(user_prompt)
        logger.critical("[REMOTE_TEST] Assistant prompt length=%d", len(assistant_prompt))
        # logger.critical("[REMOTE_TEST] Assistant prompt preview=%r", assistant_prompt[:400])
        logger.critical("[REMOTE_TEST] Assistant prompt full=%r", assistant_prompt)
        # logger.critical("[REMOTE_TEST] User prompt length=%d", len(user_prompt))
        # logger.critical("[REMOTE_TEST] User prompt preview=%r", user_prompt[:400])
        # logger.critical("[REMOTE_TEST] User prompt full=%r", user_prompt)
        # Create agents with proper configuration
        assistant_agent = OpenAIAgent.create(
            model=assistant_model,
            system_prompt=assistant_prompt,
            allowed_tools=assistant_tools,
            model_client=gateway_openai,
            validate_api_key=False,
        )
        user_agent = OpenAIAgent.create(
            model=user_model,
            system_prompt=user_prompt,
            allowed_tools=user_tools,
            model_client=gateway_openai,
            validate_api_key=False,
        )


        await multi_turn_run(ctx, assistant_agent, user_agent, max_steps=60)

async def test_airline():
    print("\n=== Test: Multi-turn conversation with agent and user ===")
    # Initialize the environment first
    await init_env()

    domain = "airline"
    task_id = "8"
    task_split = "base"

    task = env(
        "tau2",
        domain=domain,
        task_id=task_id,
        task_split=task_split,
    )
    bound_tasks = [task]

    async with hud.eval(bound_tasks, max_concurrent=1) as ctx:
        # Get tau2 configuration - pass params explicitly to avoid parsing issues
        from loop.agent_config import get_tau2_config
        user_prompt, assistant_prompt, user_tools, assistant_tools = await get_tau2_config(
            ctx, domain=domain, task_id=task_id, task_split=task_split
        )
        print("\n=== SYSTEM PROMPTS (airline) ===")
        print("\n[ASSISTANT PROMPT]\n")
        print(assistant_prompt)
        print("\n[USER PROMPT]\n")
        print(user_prompt)

        # Create agents with proper configuration
        assistant_agent = create_agent(
            model=assistant_model,
            system_prompt=assistant_prompt,
            allowed_tools=assistant_tools,
        )
        user_agent = OpenAIAgent.create(
            model=user_model,
            system_prompt=user_prompt,
            allowed_tools=user_tools,
            model_client=gateway_openai,
            validate_api_key=False,
        )
        setattr(assistant_agent, "temperature", 0.0)
        setattr(user_agent, "temperature", 0.0)
        logger.info(
            "Airline temps: assistant=%s user=%s",
            getattr(assistant_agent, "temperature", None),
            getattr(user_agent, "temperature", None),
        )

        await multi_turn_run(ctx, assistant_agent, user_agent, max_steps=30)

async def test_retail():
    print("\n=== Test: Multi-turn conversation with agent and user ===")
    # Initialize the environment first
    await init_env()

    domain = "retail"
    task_id = "67"
    task_split = "base"

    task = env("tau2", domain=domain, task_id=task_id, task_split=task_split)
    bound_tasks = [task]

    async with hud.eval(bound_tasks, max_concurrent=1) as ctx:
        # Get tau2 configuration - pass params explicitly to avoid parsing issues
        from loop.agent_config import get_tau2_config

        user_prompt, assistant_prompt, user_tools, assistant_tools = (
            await get_tau2_config(
                ctx, domain=domain, task_id=task_id, task_split=task_split
            )
        )
        
        # print("\n=== SYSTEM PROMPTS (retail) ===")
        # print("\n[ASSISTANT PROMPT]\n")
        # print(assistant_prompt)
        # print("\n[USER PROMPT]\n")
        # print(user_prompt)

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
        logger.info(
            "Retail temps: assistant=%s user=%s",
            getattr(assistant_agent, "temperature", None),
            getattr(user_agent, "temperature", None),
        )

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
    # await test_airline()
    await test_retail()

if __name__ == "__main__":
    asyncio.run(main())
