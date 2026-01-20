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

            # Clear previous task state to avoid contamination
            prev_msg_count = len(tau2_task.messages)
            tau2_task.clear_messages()
            tau2_task.reset_tokens()
            if prev_msg_count > 0:
                logger.info(f"Cleared {prev_msg_count} messages from previous task")

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

            # Store tool names for agent filtering
            # This allows runner files to filter tools per agent (assistant vs user)
            from server.tools.http_client import get_http_client
            http_client = get_http_client()
            tools_data = http_client.list_tools()

            # Store tool names in tau2_task for access by agents
            agent_tool_names = [t["name"] for t in tools_data.get("tools", []) if t["name"] != "send_message"]
            user_tool_names = [t["name"] for t in tools_data.get("user_tools", [])]

            tau2_task.agent_tool_names = agent_tool_names
            tau2_task.user_tool_names = user_tool_names

            # Also store user scenario for creating user agent
            if tau2_task.task and tau2_task.task.user_scenario:
                tau2_task.user_scenario = tau2_task.task.user_scenario
                logger.info("User scenario stored for user agent creation")
            else:
                tau2_task.user_scenario = None
                logger.warning("No user scenario found in task")

            logger.info(f"Tool names stored - agent tools: {len(agent_tool_names)}, user tools: {len(user_tool_names)}")

        except Exception as e:
            logger.error(f"Setup failed: {e}")
            import traceback
            traceback.print_exc()
            yield f"Setup failed: {e}"
            yield 0.0
            return

        # ===== PROMPT (first yield) =====
        # Provide the task prompt to the agent with policy (matching tau2-bench structure)
        # Get policy from environment server
        try:
            policy = http_client.get_policy()
        except Exception as e:
            logger.warning(f"Could not get policy: {e}")
            policy = "No specific policy available."

        # System prompt with policy (matching tau2-bench's SYSTEM_PROMPT)
        # NOTE: System prompts are now set in create_agent() via local_test.py/remote_test.py
        # Keeping this here for reference and storing in tau2_task for agent creation
        system_prompt = f"""<instructions>
You are a customer service agent that helps the user according to the <policy> provided below.
In each turn you can either:
- Send a message to the user (by providing text in your response).
- Make a tool call to check or modify data.
You cannot do both at the same time.

Try to be helpful and always follow the policy.
</instructions>

<policy>
{policy}
</policy>"""

        tau2_task.system_prompt = system_prompt
        logger.info("Stored system prompt with policy in tau2_task (used by create_agent)")

        # Initial user message (just the greeting, no policy)
        prompt = f"""Greet the customer with message: {initial_greeting}"""

        # Yield the prompt and let the agent interact
        # The conversation loop will set agent.system_prompt from tau2_task.system_prompt
        _ = yield prompt

        # ===== EVALUATE SECTION =====
        # Evaluate the conversation using TAU2-bench's evaluation (directly, not via tool call)
        logger.info("Evaluating tau2 task completion")

        try:
            tau2_task = get_tau2_task()

            # Log all messages in the trajectory for debugging
            logger.info(f"[EVAL] Starting evaluation with {len(tau2_task.messages)} messages in trajectory")
            # for i, msg in enumerate(tau2_task.messages):
            #     msg_type = type(msg).__name__
            #     if hasattr(msg, 'tool_calls') and msg.tool_calls:
            #         logger.debug(f"[EVAL] Message {i}: {msg_type} with {len(msg.tool_calls)} tool calls")
            #         for tc in msg.tool_calls:
            #             logger.debug(f"[EVAL]   - Tool: {tc.name}, Requestor: {tc.requestor}, Args: {tc.arguments}")
            #     elif hasattr(msg, 'content'):
            #         content_preview = msg.content[:100] if msg.content else "None"
            #         logger.debug(f"[EVAL] Message {i}: {msg_type}, Content: {content_preview}")

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

                # Show expected vs actual database state if mismatch
                if not reward_info.db_check.db_match and tau2_task.environment:
                    try:
                        # Get actual database state
                        actual_agent_db_hash = tau2_task.environment.get_db_hash()
                        actual_user_db_hash = tau2_task.environment.get_user_db_hash()

                        # Compute expected state by replaying golden actions
                        from tau2.environment.environment import Environment
                        expected_env = Environment(domain=tau2_task.domain)
                        if tau2_task.task.initial_state:
                            expected_env.set_state(
                                initialization_data=tau2_task.task.initial_state.initialization_data,
                                initialization_actions=tau2_task.task.initial_state.initialization_actions,
                                message_history=[]
                            )

                        # Run golden actions
                        if tau2_task.task.evaluation_criteria and tau2_task.task.evaluation_criteria.actions:
                            for action in tau2_task.task.evaluation_criteria.actions:
                                try:
                                    expected_env.make_tool_call(
                                        tool_name=action.name,
                                        requestor=action.requestor,
                                        **action.arguments,
                                    )
                                except Exception:
                                    pass

                        expected_agent_db_hash = expected_env.get_db_hash()
                        expected_user_db_hash = expected_env.get_user_db_hash()

                        logger.info(f"  Agent DB - Expected hash: {expected_agent_db_hash}, Actual hash: {actual_agent_db_hash}")
                        logger.info(f"  User DB  - Expected hash: {expected_user_db_hash}, Actual hash: {actual_user_db_hash}")

                        # Show actual database content if available
                        if tau2_task.environment.tools:
                            agent_db = tau2_task.environment.tools.get_db()
                            if agent_db:
                                logger.info(f"  Actual Agent DB: {agent_db.model_dump()}")
                        if tau2_task.environment.user_tools:
                            user_db = tau2_task.environment.user_tools.get_db()
                            if user_db:
                                logger.info(f"  Actual User DB: {user_db.model_dump()}")

                    except Exception as e:
                        logger.debug(f"Could not show database details: {e}")

            if reward_info.env_assertions:
                logger.info(f"\nEnvironment Assertions: {len(reward_info.env_assertions)} checks")
                for i, check in enumerate(reward_info.env_assertions):
                    # Show assertion function and expected value
                    env_assert = check.env_assertion
                    func_info = f"{env_assert.env_type}.{env_assert.func_name}"
                    if env_assert.arguments:
                        args_str = ", ".join(f'{k}={v}' for k, v in env_assert.arguments.items())
                        func_info += f"({args_str})"
                    logger.info(f"  [{i+1}] {func_info}: met={check.met}, reward={check.reward}")

            if reward_info.action_checks:
                logger.info(f"\nAction Checks: {len(reward_info.action_checks)} checks")
                for i, check in enumerate(reward_info.action_checks):
                    # Compact format: Action 1: requestor.action_name(args) -- reward: X.X
                    action = check.action
                    args_str = ", ".join(f'{k}="{v}"' if isinstance(v, str) else f'{k}={v}' for k, v in action.arguments.items())
                    logger.info(f"  Action {i+1}: {action.requestor}.{action.name}({args_str}) -- reward: {check.action_reward}")

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

            # Log token usage summary
            logger.info(f"\nToken Usage:")
            logger.info(f"  Input tokens: {tau2_task.total_input_tokens:,}")
            logger.info(f"  Output tokens: {tau2_task.total_output_tokens:,}")
            if tau2_task.total_cache_creation_tokens > 0:
                logger.info(f"  Cache creation tokens: {tau2_task.total_cache_creation_tokens:,}")
            if tau2_task.total_cache_read_tokens > 0:
                logger.info(f"  Cache read tokens: {tau2_task.total_cache_read_tokens:,}")
            total_tokens = tau2_task.total_input_tokens + tau2_task.total_output_tokens
            logger.info(f"  Total tokens: {total_tokens:,}")

            logger.info("=" * 60)

        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            import traceback
            traceback.print_exc()
            reward = 0.0

        # ===== REWARD (second yield) =====
        yield reward