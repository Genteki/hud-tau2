"""Task loading and setup tools for tau2-bench."""

from typing import Dict, Any, Optional, List
from tau2.data_model.tasks import Task
from tau2.registry import registry
from task import _format_system_prompt
from server.setup import setup
from server.tools.http_client import get_http_client
from server.state import get_tau2_task


@setup.tool("load")
async def load(
    domain: str,
    task_id: str,
    task_split: Optional[str] = None,
    solo_mode: bool = False,
    start_conversation: bool = False,
    initial_greeting: str = "Hi! How can I help you today?"
) -> Dict[str, Any]:
    """
    Complete setup: load domain, set task, and initialize environment.

    Args:
        domain: The domain to use (airline, retail, telecom, or mock)
        task_id: The task ID to load
        task_split: Optional task split name (e.g., "base", "train", "test")
        solo_mode: Whether to run in solo mode (default: False)
        start_conversation: If True and solo_mode=False, send initial greeting (default: False)
        initial_greeting: The greeting to send if start_conversation=True

    Returns:
        Setup status with task and environment information, plus user's first response if start_conversation=True
    """
    try:
        tau2_task = get_tau2_task()

        # 1. Load tasks for the domain
        task_loader = registry.get_tasks_loader(domain)
        tasks: List[Task] = task_loader(task_split_name=task_split)
        tau2_task.domain = domain
        tau2_task.tasks = tasks

        # 2. Set the specific task
        success = tau2_task.set_task(task_id)
        if not success:
            return {
                "error": f"Task {task_id} not found",
                "available_tasks": [t.id for t in tasks],
            }

        # 3. Initialize HTTP client and verify server is running
        http_client = get_http_client()
        if not http_client.health_check():
            return {
                "error": "Environment server is not running",
                "hint": "Start the environment server first: ./hud-tau2/scripts/start_environment_server.sh"
            }

        tau2_task.solo_mode = solo_mode

        # 4. Reset environment and apply initial state via HTTP
        try:
            # Reset environment to clean state
            http_client.reset_environment()

            # Apply task initial state
            if tau2_task.task.initial_state is not None:
                initialization_data = tau2_task.task.initial_state.initialization_data
                initialization_actions = tau2_task.task.initial_state.initialization_actions

                # Prepare initialization data for HTTP
                init_data_dict = None
                if initialization_data is not None:
                    init_data_dict = {}
                    if initialization_data.agent_data is not None:
                        # Convert Pydantic model to dict
                        init_data_dict["agent_data"] = initialization_data.agent_data.model_dump()
                    if initialization_data.user_data is not None:
                        init_data_dict["user_data"] = initialization_data.user_data.model_dump()

                # Prepare initialization actions for HTTP
                init_actions_list = None
                if initialization_actions is not None:
                    # Convert actions to serializable format
                    init_actions_list = [action.model_dump() for action in initialization_actions]

                # Send initialization to environment server
                http_client.initialize_task(
                    initialization_data=init_data_dict,
                    initialization_actions=init_actions_list
                )
        except Exception as e:
            return {
                "error": f"Failed to initialize environment via HTTP: {str(e)}"
            }

        # 5. Get policy from HTTP server + format HUD system prompt
        try:
            policy = http_client.get_policy()
            system_prompt = _format_system_prompt(policy, solo_mode)
        except Exception as e:
            return {
                "error": f"Failed to get policy via HTTP: {str(e)}"
            }

        result = {
            "status": "ready",
            "domain": domain,
            "initial_greeting": initial_greeting,
            "system_message": "**IMPORTANT**: YOU MUST USE THE `send_message` TOOL FOR ALL COMMUNICATION WITH THE USER. NEVER RESPOND WITH PLAIN TEXT!"
        }

        # # 6. Initialize conversation tool for multi-turn mode
        # if not solo_mode:
        #     from server.tools.conversation import ConversationTool

        #     ConversationTool.initialize_global(tau2_task)
        #     # result["conversation_initialized"] = True

        #     if start_conversation:
        #         from server.main import get_conversation_tool

        #         conversation_tool = get_conversation_tool()
        #         if conversation_tool:
        #             response = await conversation_tool(initial_greeting)
        #             # result["conversation_started"] = True
        #             result["initial_response"] = response[0].text if response else "No response"

        return result

    except Exception as e:
        import traceback

        return {"error": f"Setup failed: {str(e)}", "traceback": traceback.format_exc()}


