"""Server-side tool to record assistant/user text in tau2_task."""

from datetime import datetime
import logging
from hud.tools.base import BaseTool
from mcp.types import TextContent
from server.state import get_tau2_task
from tau2.data_model.message import AssistantMessage, UserMessage

logger = logging.getLogger(__name__)


class RecordMessageTool(BaseTool):
    """Append a text message to tau2_task for evaluation."""

    def __init__(self) -> None:
        super().__init__(
            env=None,
            name="__record_message",
            description=(
                "Record a user/assistant text message in the tau2 task history."
            ),
        )

    async def __call__(self, role: str, content: str) -> list[TextContent]:
        try:
            tau2_task = get_tau2_task()
            timestamp = datetime.now().isoformat()
            if role == "assistant":
                msg = AssistantMessage(role="assistant", content=content, timestamp=timestamp)
            elif role == "user":
                msg = UserMessage(role="user", content=content, timestamp=timestamp)
            else:
                return [TextContent(type="text", text="Error: role must be 'assistant' or 'user'")]
            tau2_task.add_message(msg)
            return [TextContent(type="text", text="ok")]
        except Exception as e:
            logger.error("record_message failed: %s", e)
            return [TextContent(type="text", text=f"Error: {e}")]


def create_record_message_tool() -> RecordMessageTool:
    return RecordMessageTool()
