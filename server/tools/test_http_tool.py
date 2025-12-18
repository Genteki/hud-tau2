"""Unit tests for HTTPTool and HTTPUserTool classes."""

import pytest
import json
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from mcp.types import TextContent

from server.tools.http_tool import HTTPTool, HTTPUserTool, create_http_tools_from_server

# Configure pytest to use anyio for async tests
pytestmark = pytest.mark.anyio


class TestHTTPTool:
    """Tests for HTTPTool class."""

    @pytest.fixture
    def mock_http_client(self):
        """Create a mock HTTP client."""
        client = Mock()
        client.execute_tool = Mock(return_value={"result": "success", "data": "test_data"})
        return client

    @pytest.fixture
    def http_tool(self):
        """Create an HTTPTool instance for testing."""
        schema = {
            "type": "object",
            "properties": {
                "param1": {"type": "string"},
                "param2": {"type": "integer"}
            },
            "required": ["param1"]
        }
        return HTTPTool(
            tool_name="test_tool",
            tool_description="A test tool",
            tool_schema=schema
        )

    def test_http_tool_initialization(self, http_tool):
        """Test HTTPTool initialization."""
        assert http_tool.name == "test_tool"
        assert http_tool.description == "A test tool"
        assert http_tool.tool_schema["type"] == "object"
        assert "param1" in http_tool.tool_schema["properties"]

    def test_http_client_lazy_loading(self, http_tool, mock_http_client):
        """Test that HTTP client is lazily loaded."""
        assert http_tool._http_client is None

        with patch('server.tools.http_tool.get_http_client', return_value=mock_http_client):
            client = http_tool.http_client
            assert client is not None
            assert client == mock_http_client

    async def test_call_success(self, http_tool, mock_http_client):
        """Test successful tool execution."""
        with patch('server.tools.http_tool.get_http_client', return_value=mock_http_client), \
             patch('server.state.get_tau2_task') as mock_get_task:

            # Mock tau2_task
            mock_task = Mock()
            mock_task.add_message = Mock()
            mock_get_task.return_value = mock_task

            # Execute tool
            result = await http_tool(param1="test", param2=42)

            # Verify result
            assert len(result) == 1
            assert isinstance(result[0], TextContent)
            assert "success" in result[0].text
            assert "test_data" in result[0].text

            # Verify HTTP client was called correctly
            mock_http_client.execute_tool.assert_called_once_with(
                "test_tool",
                param1="test",
                param2=42
            )

            # Verify messages were added to task
            assert mock_task.add_message.call_count == 2  # AssistantMessage + ToolMessage

    async def test_call_with_error(self, http_tool, mock_http_client):
        """Test tool execution with error."""
        mock_http_client.execute_tool.side_effect = Exception("Connection failed")

        with patch('server.tools.http_tool.get_http_client', return_value=mock_http_client), \
             patch('server.state.get_tau2_task') as mock_get_task:

            mock_task = Mock()
            mock_task.add_message = Mock()
            mock_get_task.return_value = mock_task

            result = await http_tool(param1="test")

            # Verify error message is returned
            assert len(result) == 1
            assert isinstance(result[0], TextContent)
            assert "HTTP tool execution error" in result[0].text
            assert "Connection failed" in result[0].text

    async def test_call_logs_tool_messages(self, http_tool, mock_http_client):
        """Test that tool calls are logged to tau2_task."""
        with patch('server.tools.http_tool.get_http_client', return_value=mock_http_client), \
             patch('server.state.get_tau2_task') as mock_get_task:

            mock_task = Mock()
            captured_messages = []
            mock_task.add_message = Mock(side_effect=lambda msg: captured_messages.append(msg))
            mock_get_task.return_value = mock_task

            await http_tool(param1="value1", param2=123)

            # Verify two messages were logged
            assert len(captured_messages) == 2

            # Check AssistantMessage
            assistant_msg = captured_messages[0]
            assert assistant_msg.role == "assistant"
            assert assistant_msg.content is None  # Tool calls have no content
            assert len(assistant_msg.tool_calls) == 1
            tool_call = assistant_msg.tool_calls[0]
            assert tool_call.name == "test_tool"
            assert tool_call.arguments == {"param1": "value1", "param2": 123}

            # Check ToolMessage
            tool_msg = captured_messages[1]
            assert tool_msg.role == "tool"
            assert tool_msg.id == tool_call.id
            assert "success" in tool_msg.content

    def test_mcp_property(self, http_tool):
        """Test MCP tool property."""
        with patch('fastmcp.tools.Tool') as mock_tool_class:
            mock_mcp_tool = Mock()
            mock_mcp_tool.name = "test_tool"
            mock_mcp_tool.description = "A test tool"
            mock_mcp_tool.parameters = http_tool.tool_schema
            mock_tool_class.return_value = mock_mcp_tool

            mcp_tool = http_tool.mcp

            # Verify Tool was created with correct parameters
            mock_tool_class.assert_called_once_with(
                name="test_tool",
                description="A test tool",
                parameters=http_tool.tool_schema,
                fn=http_tool.__call__
            )

            assert mcp_tool.name == "test_tool"
            assert mcp_tool.description == "A test tool"
            assert mcp_tool.parameters == http_tool.tool_schema


class TestHTTPUserTool:
    """Tests for HTTPUserTool class."""

    @pytest.fixture
    def mock_http_client(self):
        """Create a mock HTTP client."""
        client = Mock()
        client.execute_user_tool = Mock(return_value={"status": "completed", "user_response": "Yes"})
        return client

    @pytest.fixture
    def http_user_tool(self):
        """Create an HTTPUserTool instance for testing."""
        schema = {
            "type": "object",
            "properties": {
                "question": {"type": "string"}
            },
            "required": ["question"]
        }
        return HTTPUserTool(
            tool_name="ask_user",
            tool_description="Ask user a question",
            tool_schema=schema
        )

    def test_http_user_tool_initialization(self, http_user_tool):
        """Test HTTPUserTool initialization."""
        assert http_user_tool.name == "ask_user"
        assert http_user_tool.description == "Ask user a question"
        assert "question" in http_user_tool.tool_schema["properties"]

    async def test_user_tool_call_success(self, http_user_tool, mock_http_client):
        """Test successful user tool execution."""
        with patch('server.tools.http_tool.get_http_client', return_value=mock_http_client):
            result = await http_user_tool(question="Are you ready?")

            # Verify result
            assert len(result) == 1
            assert isinstance(result[0], TextContent)
            assert "completed" in result[0].text
            assert "Yes" in result[0].text

            # Verify HTTP client was called
            mock_http_client.execute_user_tool.assert_called_once_with(
                "ask_user",
                question="Are you ready?"
            )

    async def test_user_tool_call_with_error(self, http_user_tool, mock_http_client):
        """Test user tool execution with error."""
        mock_http_client.execute_user_tool.side_effect = Exception("User timeout")

        with patch('server.tools.http_tool.get_http_client', return_value=mock_http_client):
            result = await http_user_tool(question="Are you there?")

            # Verify error message
            assert len(result) == 1
            assert "HTTP user tool execution error" in result[0].text
            assert "User timeout" in result[0].text


class TestCreateHttpToolsFromServer:
    """Tests for create_http_tools_from_server function."""

    @pytest.fixture
    def mock_client(self):
        """Create a mock HTTP client with tools."""
        client = Mock()
        client.health_check = Mock(return_value=True)
        client.list_tools = Mock(return_value={
            "tools": [
                {
                    "name": "search_database",
                    "description": "Search the database",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"}
                        }
                    }
                },
                {
                    "name": "update_record",
                    "description": "Update a record",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            "data": {"type": "object"}
                        }
                    }
                }
            ],
            "user_tools": [
                {
                    "name": "confirm_action",
                    "description": "Ask user to confirm",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "message": {"type": "string"}
                        }
                    }
                }
            ]
        })
        return client

    def test_create_tools_success(self, mock_client):
        """Test successful tool creation from server."""
        with patch('server.tools.http_tool.get_http_client', return_value=mock_client):
            tools = create_http_tools_from_server()

            # Verify correct number of tools
            assert len(tools) == 3  # 2 regular + 1 user tool

            # Verify regular tools
            assert "search_database" in tools
            assert isinstance(tools["search_database"], HTTPTool)
            assert tools["search_database"].name == "search_database"

            assert "update_record" in tools
            assert isinstance(tools["update_record"], HTTPTool)

            # Verify user tool
            assert "confirm_action" in tools
            assert isinstance(tools["confirm_action"], HTTPUserTool)

    def test_create_tools_server_unhealthy(self, mock_client):
        """Test tool creation fails when server is unhealthy."""
        mock_client.health_check.return_value = False

        with patch('server.tools.http_tool.get_http_client', return_value=mock_client):
            with pytest.raises(RuntimeError, match="Environment server is not reachable"):
                create_http_tools_from_server()

    def test_create_tools_empty_response(self, mock_client):
        """Test tool creation with empty tools list."""
        mock_client.list_tools.return_value = {"tools": [], "user_tools": []}

        with patch('server.tools.http_tool.get_http_client', return_value=mock_client):
            tools = create_http_tools_from_server()

            assert len(tools) == 0

    def test_create_tools_with_defaults(self, mock_client):
        """Test tool creation handles missing parameters field."""
        mock_client.list_tools.return_value = {
            "tools": [
                {
                    "name": "simple_tool",
                    "description": "A simple tool"
                    # No parameters field
                }
            ],
            "user_tools": []
        }

        with patch('server.tools.http_tool.get_http_client', return_value=mock_client):
            tools = create_http_tools_from_server()

            assert "simple_tool" in tools
            assert tools["simple_tool"].tool_schema == {"type": "object", "properties": {}}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
