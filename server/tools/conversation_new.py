"""Simplified conversation tool that wraps HTTP calls to environment server."""

import logging
from datetime import datetime
from hud.tools.base import BaseTool
from mcp.types import TextContent
from server.state import get_tau2_task
from server.tools.http_client import get_http_client
from tau2.data_model.message import AssistantMessage, UserMessage
from tau2.user.base import STOP, TRANSFER, OUT_OF_SCOPE

logger = logging.getLogger(__name__)


class ConversationTool(BaseTool):
    """Send a message to the simulated user via HTTP and receive their response."""

    def __init__(self):
        super().__init__(
            env=None,
            name="send_message",
            description=(
                "Send a message to the customer and receive their response. "
                "Use this to communicate with the customer during the conversation."
            ),
        )

    async def __call__(self, message: str) -> list[TextContent]:
        """Send a message to the user simulator via HTTP and get their response."""
        tau2_task = get_tau2_task()

        # Validate task is loaded
        if tau2_task.task is None:
            return [TextContent(type="text", text="Error: Task not initialized.")]

        try:
            # Log agent message locally
            agent_message = AssistantMessage(
                role="assistant",
                content=message,
                cost=0.0,
                timestamp=datetime.now().isoformat()
            )
            tau2_task.add_message(agent_message)

            # Send message to environment server via HTTP
            http_client = get_http_client()
            response = http_client.send_message(message)

            if "error" in response:
                error_msg = response.get("error", "Unknown error")
                return [TextContent(type="text", text=f"Error: {error_msg}")]

            user_content = response.get("user_message", "")

            # Log user message locally
            user_message = UserMessage(
                role="user",
                content=user_content,
                timestamp=datetime.now().isoformat()
            )
            tau2_task.add_message(user_message)

            # Check for conversation signals and add appropriate notes
            if STOP in user_content:
                return [TextContent(type="text", text=f"{user_content}\n\n[User has ended the conversation.]")]

            if TRANSFER in user_content:
                return [TextContent(type="text", text=f"{user_content}\n\n[User requested transfer to human agent.]")]

            if OUT_OF_SCOPE in user_content:
                return [TextContent(type="text", text=f"{user_content}\n\n[User scenario doesn't provide enough information.]")]

            return [TextContent(type="text", text=user_content)]

        except Exception as e:
            import traceback
            return [
                TextContent(
                    type="text", text=f"Error in conversation: {str(e)}\n\n{traceback.format_exc()}"
                )
            ]


def create_conversation_tool() -> ConversationTool:
    """Create and return a ConversationTool instance."""
    return ConversationTool()
