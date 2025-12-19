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
        task_split: str = "base"
    ) -> Any:
        """
        Run a TAU2-bench customer service task.

        Args:
            domain: Domain to test (airline, retail, telecom)
            task_id: Task ID within the domain
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

        except Exception as e:
            logger.error(f"Setup failed: {e}")
            import traceback
            traceback.print_exc()
            yield f"Setup failed: {e}"
            yield 0.0
            return

        # ===== PROMPT (first yield) =====
        # Provide the task prompt to the agent
        prompt = f"""
Greet to customer with tool `send_message` tool, with greeting intro:
{initial_greeting}
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
            logger.info(f"Evaluation complete: reward={reward}")

        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            import traceback
            traceback.print_exc()
            reward = 0.0

        # ===== REWARD (second yield) =====
        yield reward