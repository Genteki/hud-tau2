"""TAU2-bench scenarios - customer service task evaluation."""
import logging
from typing import Any

logger = logging.getLogger(__name__)


def register_tau2_scenarios(env):
    """Register all tau2-bench scenarios with the environment."""

    @env.scenario("tau2")
    async def tau2_scenario(
        domain: str = "airline",
        task_id: int = 0,
        task_split: str = "dev"
    ) -> Any:
        """
        Run a TAU2-bench customer service task.

        Args:
            domain: Domain to test (airline, retail, telecom)
            task_id: Task ID within the domain
            task_split: Task split (dev, test)

        Returns:
            Task evaluation result
        """
        # ===== SETUP SECTION =====
        # Load the task and initialize environment
        logger.info(f"Setting up tau2 scenario: domain={domain}, task_id={task_id}, split={task_split}")

        setup_result = await env.call_tool(
            "setup__load",
            domain=domain,
            task_id=task_id,
            task_split=task_split,
            solo_mode=False,
            start_conversation=False
        )

        if "error" in setup_result:
            logger.error(f"Setup failed: {setup_result['error']}")
            yield f"Setup failed: {setup_result['error']}"
            yield 0.0
            return

        # Get the initial greeting for the agent
        initial_greeting = setup_result.get("initial_greeting", "")
        system_message = setup_result.get("system_message", "")

        # ===== PROMPT (first yield) =====
        # Provide the task prompt to the agent
        prompt = f"""{system_message}

Initial customer message:
{initial_greeting}

Your task is to help this customer by using the available tools and communicating via the send_message tool.
Continue the conversation until the customer's issue is resolved or they end the conversation."""

        # Yield the prompt and let the agent interact
        # The answer is not used since tau2 evaluates the full conversation trajectory
        _ = yield prompt

        # ===== EVALUATE SECTION =====
        # Evaluate the conversation using TAU2-bench's evaluation
        logger.info("Evaluating tau2 task completion")

        eval_result = await env.call_tool(
            "evaluate__evaluate_task",
            domain=domain,
            task_id=task_id,
            task_split=task_split
        )

        # Extract reward from evaluation result
        reward = eval_result.get("reward", 0.0)
        success = eval_result.get("success", False)

        logger.info(f"Evaluation complete: reward={reward}, success={success}")

        # ===== REWARD (second yield) =====
        yield reward