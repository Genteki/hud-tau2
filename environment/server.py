"""FastAPI server for tau2-bench domain tools.

Exposes domain-specific tools as HTTP endpoints following the EnvironmentServer pattern.
"""

import json
import logging
from typing import Any, Optional
from fastapi import FastAPI, HTTPException
from pydantic import create_model
from typing_extensions import Annotated

logger = logging.getLogger(__name__)


class EnvironmentServer:
    """
    A FastAPI server that exposes tau2-bench domain tools as HTTP endpoints.

    This follows the same pattern as tau2-bench's EnvironmentServer, but adapted
    for the MCP tool integration layer.
    """

    def __init__(self, environment):
        """
        Initialize the server with a tau2-bench environment.

        Args:
            environment: The tau2-bench Environment instance
        """
        self.environment = environment
        self.app = FastAPI(
            title=f"TAU2-Bench Environment: {environment.get_domain_name()}",
            description=self._format_description(environment.get_policy()),
            version="1.0.0",
            openapi_tags=[
                {"name": "Tools", "description": "Available tools in this environment"},
                {
                    "name": "User Tools",
                    "description": "User-defined tools in this environment",
                },
            ],
            openapi_url="/api/openapi.json",
            docs_url="/docs",
            redoc_url="/redoc",
        )
        self._setup_routes()
        self._add_management_routes()

    def _format_description(self, policy: str) -> str:
        """Format the API description with markdown for better ReDoc rendering"""
        import re

        # Look for sections using regex
        sections = {}
        for section_name in ["main_policy", "tech_support_policy"]:
            pattern = f"<{section_name}>(.*?)</{section_name}>"
            match = re.search(pattern, policy, re.DOTALL)
            if match:
                sections[section_name] = match.group(1).strip()

        # If no sections found, return original format
        if not sections:
            return f"""
{policy}

## Tools

This environment provides several tools that can be used via API endpoints. Each tool is exposed as a POST endpoint under `/tools/`.

### Authentication

No authentication is required for this API.

### Response Format

All successful responses will return the tool's output directly. Errors will return a 400 status code with an error message.
"""

        # Format the description with sections
        description = []

        # Add main policy if it exists
        if "main_policy" in sections:
            description.append(sections["main_policy"])

        # Add other sections as subsections
        for section_name, content in sections.items():
            if section_name != "main_policy":
                description.append(f"\n## {section_name.replace('_', ' ').title()}\n")
                description.append(content)

        # Add the tools section
        description.append("""

## Tools

This environment provides several tools that can be used via API endpoints. Each tool is exposed as a POST endpoint under `/tools/`.

### Authentication

No authentication is required for this API.

### Response Format

All successful responses will return the tool's output directly. Errors will return a 400 status code with an error message.
""")

        return "\n".join(description)

    def _setup_routes(self):
        """Set up routes for each tool in the environment"""
        from tau2.environment.toolkit import get_tool_signatures

        # Set up regular tools
        tool_signatures = get_tool_signatures(self.environment.tools)
        self._setup_tool_routes(tool_signatures, "tools")

        # Set up user tools if they exist
        if self.environment.user_tools is not None:
            user_tool_signatures = get_tool_signatures(self.environment.user_tools)
            self._setup_tool_routes(user_tool_signatures, "user_tools")

    def _setup_tool_routes(self, tool_signatures: dict, route_prefix: str):
        """Helper method to set up routes for a set of tools"""
        for name, signature in tool_signatures.items():
            # Create a Pydantic model for the tool's parameters
            if signature.params:
                fields = {}
                for param_name, param_schema in signature.params["properties"].items():
                    # Convert JSON schema types to Python types
                    python_type = str  # default type
                    if param_schema.get("type") == "number":
                        python_type = float
                    elif param_schema.get("type") == "integer":
                        python_type = int
                    elif param_schema.get("type") == "boolean":
                        python_type = bool

                    fields[param_name] = (Annotated[python_type, None], ...)

                RequestModel = create_model(
                    f"{name.title()}Request",
                    **fields,
                    __doc__=f"Request model for the {name} tool",
                )
            else:
                RequestModel = create_model(
                    f"{name.title()}Request",
                    __doc__=f"Request model for the {name} tool",
                )

            # Create the route with enhanced documentation
            summary = f"{name.replace('_', ' ').title()}"

            @self.app.post(
                f"/{route_prefix}/{name}",
                response_model=Any,
                description=self._format_tool_description(
                    signature.doc, signature.returns, route_prefix == "user_tools"
                ),
                name=name,
                tags=["User Tools" if route_prefix == "user_tools" else "Tools"],
                summary=summary,
            )
            async def tool_endpoint(
                request: RequestModel,  # type: ignore
                tool_name: str = name,
            ) -> Any:
                try:
                    if route_prefix == "user_tools":
                        result = self.environment.use_user_tool(
                            tool_name=tool_name, **request.model_dump()
                        )
                    else:
                        result = self.environment.use_tool(
                            tool_name=tool_name, **request.model_dump()
                        )
                    return result
                except Exception as e:
                    raise HTTPException(status_code=400, detail=str(e))

    def _format_tool_description(
        self, doc: str, returns: Optional[dict] = None, is_user_tool: bool = False
    ) -> str:
        """Format tool documentation for better ReDoc rendering"""
        import re

        # Extract content between triple quotes using regex
        match = re.search(r'"""(.*?)"""', doc, re.DOTALL)
        if match:
            doc = match.group(1).strip()

        description = f"""
{"(User Tool) " if is_user_tool else ""}{doc}

### Response Format
The response will be the direct output of the tool execution.
"""

        if returns and "properties" in returns:
            # Get the first (and usually only) property's info
            return_info = next(iter(returns["properties"].values()))

            description += "\n<details><summary>Response Schema</summary>\n\n```json\n"
            description += json.dumps(return_info, indent=2)
            description += "\n```\n</details>\n"

        description += """
### Errors
- Returns 400 if the tool execution fails
- Returns 422 if the request parameters are invalid
"""
        return description

    def get_app(self) -> FastAPI:
        """
        Get the FastAPI application.

        Returns:
            The FastAPI application
        """
        return self.app

    def _add_management_routes(self):
        """Add management routes for health checks, tool listing, and initialization."""

        @self.app.get("/health")
        async def health_check():
            """Health check endpoint."""
            return {"status": "healthy", "domain": self.environment.get_domain_name()}

        @self.app.get("/tools")
        async def list_tools():
            """List all available tools."""
            from tau2.environment.toolkit import get_tool_signatures

            tool_signatures = get_tool_signatures(self.environment.tools)
            tools_list = []
            for name, signature in tool_signatures.items():
                tools_list.append({
                    "name": name,
                    "description": signature.doc,
                    "parameters": signature.params
                })

            return {"tools": tools_list}

        @self.app.post("/execute_tool")
        async def execute_tool(request: dict):
            """
            Generic tool execution endpoint that works across domain changes.

            Args:
                request: Dict with 'tool_name' and tool arguments

            Returns:
                Tool execution result
            """
            tool_name = request.get("tool_name")
            if not tool_name:
                raise HTTPException(status_code=400, detail="Missing 'tool_name' in request")

            # Get tool arguments (everything except tool_name)
            tool_args = {k: v for k, v in request.items() if k != "tool_name"}

            try:
                # Try regular tools first
                if self.environment.tools.has_tool(tool_name):
                    result = self.environment.use_tool(tool_name=tool_name, **tool_args)
                    return {"result": result}
                # Try user tools
                elif self.environment.user_tools and self.environment.user_tools.has_tool(tool_name):
                    result = self.environment.use_user_tool(tool_name=tool_name, **tool_args)
                    return {"result": result}
                else:
                    raise HTTPException(
                        status_code=404,
                        detail=f"Tool '{tool_name}' not found in current domain '{self.environment.get_domain_name()}'"
                    )
            except Exception as e:
                import traceback
                logger.error(f"Tool execution error for '{tool_name}': {e}\n{traceback.format_exc()}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/policy")
        async def get_policy():
            """Get the domain policy."""
            return {"policy": self.environment.get_policy()}

        @self.app.post("/initialize_task")
        async def initialize_task(initialization_data: dict = None, initialization_actions: list = None):
            """
            Initialize environment with task-specific state.

            Args:
                initialization_data: Dict with 'agent_data' and 'user_data' keys
                initialization_actions: List of actions to execute

            Returns:
                Status dict
            """
            try:
                if initialization_data:
                    # Apply agent data
                    if "agent_data" in initialization_data and initialization_data["agent_data"]:
                        self.environment.tools.update_db(initialization_data["agent_data"])

                    # Apply user data
                    if "user_data" in initialization_data and initialization_data["user_data"]:
                        if self.environment.user_tools:
                            self.environment.user_tools.update_db(initialization_data["user_data"])

                # Execute initialization actions
                if initialization_actions:
                    for action in initialization_actions:
                        self.environment.run_env_function_call(action)

                return {"status": "initialized"}
            except Exception as e:
                logger.error(f"Task initialization failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/reset")
        async def reset_environment():
            """Reset the environment to initial state."""
            try:
                # Re-create environment (this will reset the DB)
                from tau2.registry import registry
                domain = self.environment.get_domain_name()
                env_constructor = registry.get_env_constructor(domain)
                self.environment = env_constructor(solo_mode=False)

                return {"status": "reset"}
            except Exception as e:
                logger.error(f"Environment reset failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/initialize_scenario")
        async def initialize_scenario(request: dict):
            """
            Initialize a scenario: load task, reset environment, apply initial state.

            Args:
                request: Dict with domain, task_id, task_split

            Returns:
                Initial greeting and scenario info
            """
            from tau2.registry import registry

            domain = request.get("domain")
            task_id = str(request.get("task_id"))
            task_split = request.get("task_split", "base")

            try:
                # 1. Load tasks for the domain
                task_loader = registry.get_tasks_loader(domain)
                tasks = task_loader(task_split_name=task_split)

                # 2. Find the specific task
                task = None
                for t in tasks:
                    if str(t.id) == task_id:
                        task = t
                        break

                if task is None:
                    return {
                        "error": f"Task {task_id} not found",
                        "available_tasks": [str(t.id) for t in tasks]
                    }

                # 3. Reset environment
                env_constructor = registry.get_env_constructor(domain)
                self.environment = env_constructor(solo_mode=False)

                # Store task for user simulator
                self._current_task = task
                self._user_simulator = None  # Will be initialized on first send_message
                self._user_state = None

                # 4. Apply task initial state
                if task.initial_state is not None:
                    initialization_data = task.initial_state.initialization_data
                    initialization_actions = task.initial_state.initialization_actions

                    # Apply initialization data
                    if initialization_data is not None:
                        if initialization_data.agent_data is not None:
                            self.environment.tools.update_db(initialization_data.agent_data.model_dump())
                        if initialization_data.user_data is not None and self.environment.user_tools:
                            self.environment.user_tools.update_db(initialization_data.user_data.model_dump())

                    # Execute initialization actions
                    if initialization_actions is not None:
                        for action in initialization_actions:
                            if action.func_name:
                                # Execute the action on the appropriate toolkit
                                if action.env_type == "agent":
                                    getattr(self.environment.tools, action.func_name)(**action.arguments)
                                elif action.env_type == "user":
                                    getattr(self.environment.user_tools, action.func_name)(**action.arguments)

                # 5. Get user's initial greeting
                if self.environment.user_tools and self.environment.user_tools.db:
                    user_data = self.environment.user_tools.db.model_dump()
                    initial_greeting = user_data.get("greeting", "Hi! How can I help you today?")
                else:
                    initial_greeting = "Hi! How can I help you today?"

                return {
                    "status": "ready",
                    "domain": domain,
                    "task_id": task_id,
                    "initial_greeting": initial_greeting
                }

            except Exception as e:
                import traceback
                logger.error(f"Scenario initialization failed: {e}")
                return {
                    "error": f"Scenario initialization failed: {str(e)}",
                    "traceback": traceback.format_exc()
                }

        @self.app.post("/send_message")
        async def send_message(request: dict):
            """
            Send a message to the simulated user and get their response.

            Args:
                request: Dict with 'message' from agent

            Returns:
                User's response message
            """
            from tau2.user.user_simulator import UserSimulator
            from tau2.data_model.message import AssistantMessage
            from tau2.utils.utils import get_now
            import os
            import json

            message = request.get("message", "")

            try:
                # Initialize UserSimulator if not already done
                if not hasattr(self, '_user_simulator') or self._user_simulator is None:
                    # Get user scenario from current task (stored during initialize_scenario)
                    if not hasattr(self, '_current_task') or self._current_task is None:
                        return {"error": "No task loaded. Call /initialize_scenario first."}

                    user_llm = os.getenv("USER_LLM", "gpt-4-0613")
                    user_llm_args = {
                        "temperature": float(os.getenv("USER_TEMPERATURE", "0.7")),
                        "max_tokens": int(os.getenv("USER_MAX_TOKENS", "2500")),
                    }

                    # Get user tools if available
                    user_tools = None
                    if self.environment.user_tools:
                        # Convert ToolKit to list of Tool objects
                        user_tools = list(self.environment.user_tools.get_tools().values())

                    self._user_simulator = UserSimulator(
                        tools=user_tools,
                        instructions=self._current_task.user_scenario.instructions,
                        llm=user_llm,
                        llm_args=user_llm_args,
                    )
                    self._user_state = self._user_simulator.get_init_state(message_history=[])

                # Create agent message
                agent_message = AssistantMessage(
                    role="assistant",
                    content=message,
                    cost=0.0,
                    timestamp=get_now()
                )

                # Generate user response
                user_message, new_state = self._user_simulator.generate_next_message(
                    message=agent_message, state=self._user_state
                )
                self._user_state = new_state

                # Handle user tool calls if any
                while user_message.is_tool_call():
                    tool_messages = []
                    for tool_call in user_message.tool_calls:
                        # Execute user tool
                        tool_name = tool_call.name
                        tool_args = tool_call.arguments

                        result = getattr(self.environment.user_tools, tool_name)(**tool_args)

                        from tau2.data_model.message import ToolMessage
                        tool_msg = ToolMessage(
                            id=tool_call.id,
                            role="tool",
                            content=json.dumps(result, ensure_ascii=False),
                            requestor=tool_call.requestor,
                            timestamp=get_now()
                        )
                        tool_messages.append(tool_msg)

                    # Get next user response after tool execution
                    if len(tool_messages) > 1:
                        from tau2.data_model.message import MultiToolMessage
                        multi_tool_msg = MultiToolMessage(role="tool", tool_messages=tool_messages)
                        user_message, new_state = self._user_simulator.generate_next_message(
                            message=multi_tool_msg, state=self._user_state
                        )
                    else:
                        user_message, new_state = self._user_simulator.generate_next_message(
                            message=tool_messages[0], state=self._user_state
                        )
                    self._user_state = new_state

                # Return user response
                return {
                    "user_message": user_message.content or "",
                    "role": user_message.role
                }

            except Exception as e:
                import traceback
                logger.error(f"send_message failed: {e}")
                return {
                    "error": f"send_message failed: {str(e)}",
                    "traceback": traceback.format_exc()
                }

    def run(self, host: str = "127.0.0.1", port: int = 8002):
        """
        Run the FastAPI server.

        Args:
            host: The host to bind to
            port: The port to bind to
        """
        import uvicorn

        uvicorn.run(self.app, host=host, port=port)
