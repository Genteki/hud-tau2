"""TAU2-bench scenarios - customer service task evaluation."""
import logging
from typing import Any

logger = logging.getLogger(__name__)


def register_tau2_scenarios(env):
    """Register all tau2-bench scenarios with the environment."""

    @env.scenario("tau2")
    async def tau2_scenario(
        domain: str = "airline",
        task_id: int | str = 0,
        task_split: str = "base"
    ) -> Any:
        """
        Run a TAU2-bench customer service task.

        Args:
            domain: Domain to test (airline, retail, telecom)
            task_id: Task ID within the domain (int for airline/retail, str for telecom)
            task_split: Task split (base, dev, test)

        Returns:
            Task evaluation result
        """
        # ===== SETUP SECTION =====
        # Load the task and initialize environment directly (not via tool call)
        logger.info(f"Setting up tau2 scenario: domain={domain}, task_id={task_id}, split={task_split}")

        from server.tools.http_client import get_http_client
        from server.state import get_tau2_task
        from tau2.registry import registry

        try:
            # Initialize scenario via HTTP
            http_client = get_http_client()
            result = http_client.initialize_scenario(
                domain=domain,
                task_id=str(task_id),
                task_split=task_split
            )

            if "error" in result:
                logger.error(f"Setup failed: {result['error']}")
                yield f"Setup failed: {result['error']}"
                yield 0.0
                return

            initial_greeting = result.get("initial_greeting", "Hi! How can I help you today?")

            # Also update global tau2_task state (for message tracking and evaluation)
            tau2_task = get_tau2_task()
            task_loader = registry.get_tasks_loader(domain)
            tasks = task_loader(task_split_name=task_split)
            tau2_task.domain = domain
            tau2_task.tasks = tasks
            tau2_task.set_task(str(task_id))
            tau2_task.solo_mode = False

            logger.info(f"Scenario initialized: domain={domain}, task_id={task_id}, split={task_split}")

            # Dynamically load tools for this domain from environment server
            from server.tools.http_tool import create_http_tools_from_server, get_http_tool_registry

            # Clear old domain tools from registry
            tool_registry = get_http_tool_registry()
            tool_registry.clear()

            # Load new tools for current domain
            http_tools = create_http_tools_from_server()

            # Add tools to environment (this registers them with the MCP server)
            # Use the env parameter from register_tau2_scenarios closure
            for tool_name, http_tool in http_tools.items():
                env.add_tool(http_tool)

            logger.info(f"Loaded {len(http_tools)} tools for domain '{domain}'")

        except Exception as e:
            logger.error(f"Setup failed: {e}")
            import traceback
            traceback.print_exc()
            yield f"Setup failed: {e}"
            yield 0.0
            return

        # ===== PROMPT (first yield) =====
        # Provide the task prompt to the agent with policy (like original tau2-bench)
        # Get policy from environment server
        try:
            policy = http_client.get_policy()
        except Exception as e:
            logger.warning(f"Could not get policy: {e}")
            policy = "No specific policy available."

        prompt = f"""You are a customer service agent for {domain}.

<instructions>
You are a customer service agent that helps the user according to the <policy> provided below.
In each turn you can either:
- Send a message to the user using the send_message tool.
- Make a tool call to check or modify data.
You cannot do both at the same time.

Try to be helpful and always follow the policy.
</instructions>

<policy>
{policy}
</policy>

The customer has sent you this message:
{initial_greeting}

Use the send_message tool to respond to the customer.
"""

        # Yield the prompt and let the agent interact
        # The answer is not used since tau2 evaluates the full conversation trajectory
        _ = yield prompt

        # ===== EVALUATE SECTION =====
        # Evaluate the conversation using TAU2-bench's evaluation (directly, not via tool call)
        logger.info("Evaluating tau2 task completion")

        try:
            tau2_task = get_tau2_task()

            # Run tau2-bench evaluation (inline, same as evaluate/eval.py)
            from tau2.evaluator.evaluator import evaluate_simulation, EvaluationType
            from tau2.data_model.simulation import SimulationRun, TerminationReason
            from tau2.utils.utils import get_now
            import uuid

            # Validate task state
            assert tau2_task.task is not None, "Task not loaded"
            assert tau2_task.domain is not None, "Domain not set"

            # Create SimulationRun from current state
            current_time = get_now()
            simulation = SimulationRun(
                id=str(uuid.uuid4()),
                task_id=tau2_task.task.id,
                start_time=current_time,
                end_time=current_time,
                duration=0.0,
                messages=tau2_task.messages,
                termination_reason=TerminationReason.AGENT_STOP,
            )

            # Run evaluation
            reward_info = evaluate_simulation(
                simulation=simulation,
                task=tau2_task.task,
                evaluation_type=EvaluationType.ALL,
                solo_mode=tau2_task.solo_mode,
                domain=tau2_task.domain,
            )

            reward = float(reward_info.reward)

            # Log detailed evaluation summary
            logger.info("=" * 60)
            logger.info("EVALUATION SUMMARY")
            logger.info("=" * 60)
            logger.info(f"Final Reward: {reward}")

            if reward_info.reward_breakdown:
                logger.info("\nReward Breakdown:")
                for reward_type, value in reward_info.reward_breakdown.items():
                    logger.info(f"  {reward_type}: {value}")

            if reward_info.db_check:
                logger.info(f"\nDatabase Check: match={reward_info.db_check.db_match}, reward={reward_info.db_check.db_reward}")

            if reward_info.env_assertions:
                logger.info(f"\nEnvironment Assertions: {len(reward_info.env_assertions)} checks")
                for i, check in enumerate(reward_info.env_assertions):
                    logger.info(f"  [{i+1}] {check.env_assertion}: met={check.met}, reward={check.reward}")

            if reward_info.action_checks:
                logger.info(f"\nAction Checks: {len(reward_info.action_checks)} checks")
                for i, check in enumerate(reward_info.action_checks):
                    logger.info(f"  [{i+1}] {check.action}: match={check.action_match}, reward={check.action_reward}")

            if reward_info.nl_assertions:
                logger.info(f"\nNL Assertions: {len(reward_info.nl_assertions)} checks")
                for i, check in enumerate(reward_info.nl_assertions):
                    logger.info(f"  [{i+1}] {check.nl_assertion}: met={check.met}")
                    if check.justification:
                        logger.info(f"      Justification: {check.justification}")

            if reward_info.communicate_checks:
                logger.info(f"\nCommunication Checks: {len(reward_info.communicate_checks)} checks")
                for i, check in enumerate(reward_info.communicate_checks):
                    logger.info(f"  [{i+1}] {check.info}: met={check.met}")
                    if check.justification:
                        logger.info(f"      Justification: {check.justification}")

            if reward_info.info:
                logger.info(f"\nAdditional Info: {reward_info.info}")

            logger.info("=" * 60)

        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            import traceback
            traceback.print_exc()
            reward = 0.0

        # ===== REWARD (second yield) =====
        yield reward