"""Server-side tool to record assistant/user text in tau2_task."""

from datetime import datetime
import logging
from hud.tools.base import BaseTool
from mcp.types import TextContent
from server.state import get_tau2_task
from tau2.data_model.message import AssistantMessage, UserMessage

logger = logging.getLogger(__name__)


class RecordMessageTool(BaseTool):
    """Append conversation messages to tau2_task for evaluation."""

    def __init__(self) -> None:
        super().__init__(
            env=None,
            name="record_message",
            description=(
                "Record a user/assistant text message in the tau2 task history."
            ),
        )

    async def __call__(self, conversation) -> list[TextContent]:
        try:
            tau2_task = get_tau2_task()
            if isinstance(conversation, str):
                import json

                conversation = json.loads(conversation)
            if not isinstance(conversation, list):
                return [TextContent(type="text", text="Error: conversation must be a list")]
            for item in conversation:
                if not isinstance(item, dict):
                    continue
                role = item.get("role")
                content = item.get("content")
                if not role or content is None:
                    continue
                timestamp = datetime.now().isoformat()
                if role == "assistant":
                    msg = AssistantMessage(
                        role="assistant", content=content, timestamp=timestamp
                    )
                elif role == "user":
                    msg = UserMessage(role="user", content=content, timestamp=timestamp)
                else:
                    continue
                tau2_task.add_message(msg)
            return [TextContent(type="text", text="ok")]
        except Exception as e:
            logger.error("record_message failed: %s", e)
            return [TextContent(type="text", text=f"Error: {e}")]


def create_record_message_tool() -> RecordMessageTool:
    return RecordMessageTool()
