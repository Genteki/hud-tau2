"""HTTP client for tau2-bench environment server.

Provides HTTP-based tool execution by calling the uvicorn environment server.
"""

import os
import logging
import requests
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class EnvironmentHTTPClient:
    """HTTP client for communicating with tau2-bench environment server."""

    def __init__(self, base_url: Optional[str] = None):
        """
        Initialize the HTTP client.

        Args:
            base_url: Base URL of the environment server (e.g., "http://127.0.0.1:8002")
                     If not provided, reads from TAU2_SERVER_URL environment variable
        """
        self.base_url = base_url or os.getenv(
            "TAU2_SERVER_URL", "http://127.0.0.1:8002"
        )
        self.session = requests.Session()
        logger.info(f"HTTP client initialized for {self.base_url}")

    def execute_tool(self, tool_name: str, **kwargs) -> Any:
        """
        Execute a domain tool via HTTP.

        Args:
            tool_name: Name of the tool to execute
            **kwargs: Tool parameters

        Returns:
            Tool execution result

        Raises:
            requests.HTTPError: If the tool execution fails
        """
        url = f"{self.base_url}/tools/{tool_name}"

        try:
            response = self.session.post(url, json=kwargs)
            response.raise_for_status()
            return response.json()
        except requests.HTTPError as e:
            logger.error(f"Tool execution failed for '{tool_name}': {e}")
            if e.response is not None:
                logger.error(f"Response: {e.response.text}")
            raise

    def execute_user_tool(self, tool_name: str, **kwargs) -> Any:
        """
        Execute a user tool via HTTP.

        Args:
            tool_name: Name of the user tool to execute
            **kwargs: Tool parameters

        Returns:
            Tool execution result

        Raises:
            requests.HTTPError: If the tool execution fails
        """
        url = f"{self.base_url}/user_tools/{tool_name}"

        try:
            response = self.session.post(url, json=kwargs)
            response.raise_for_status()
            return response.json()
        except requests.HTTPError as e:
            logger.error(f"User tool execution failed for '{tool_name}': {e}")
            if e.response is not None:
                logger.error(f"Response: {e.response.text}")
            raise

    def get_policy(self) -> str:
        """
        Get the domain policy from the server.

        Returns:
            Policy text

        Raises:
            requests.HTTPError: If the request fails
        """
        url = f"{self.base_url}/policy"

        try:
            response = self.session.get(url)
            response.raise_for_status()
            return response.json()["policy"]
        except requests.HTTPError as e:
            logger.error(f"Failed to get policy: {e}")
            raise

    def list_tools(self) -> Dict[str, Any]:
        """
        List all available tools from the server.

        Returns:
            Dictionary with 'tools' and 'user_tools' keys

        Raises:
            requests.HTTPError: If the request fails
        """
        url = f"{self.base_url}/tools"

        try:
            response = self.session.get(url)
            response.raise_for_status()
            return response.json()
        except requests.HTTPError as e:
            logger.error(f"Failed to list tools: {e}")
            raise

    def health_check(self) -> bool:
        """
        Check if the server is healthy and reachable.

        Returns:
            True if server is healthy, False otherwise
        """
        url = f"{self.base_url}/health"

        try:
            response = self.session.get(url, timeout=5)
            response.raise_for_status()
            return response.json().get("status") == "healthy"
        except Exception as e:
            logger.warning(f"Health check failed: {e}")
            return False

    def initialize_task(self, initialization_data: Optional[dict] = None, initialization_actions: Optional[list] = None) -> dict:
        """
        Initialize the environment with task-specific state.

        Args:
            initialization_data: Dict with 'agent_data' and 'user_data' keys
            initialization_actions: List of actions to execute

        Returns:
            Status dict

        Raises:
            requests.HTTPError: If the request fails
        """
        url = f"{self.base_url}/initialize_task"

        payload = {}
        if initialization_data:
            payload["initialization_data"] = initialization_data
        if initialization_actions:
            payload["initialization_actions"] = initialization_actions

        try:
            response = self.session.post(url, json=payload)
            response.raise_for_status()
            return response.json()
        except requests.HTTPError as e:
            logger.error(f"Task initialization failed: {e}")
            if e.response is not None:
                logger.error(f"Response: {e.response.text}")
            raise

    def reset_environment(self) -> dict:
        """
        Reset the environment to initial state.

        Returns:
            Status dict

        Raises:
            requests.HTTPError: If the request fails
        """
        url = f"{self.base_url}/reset"

        try:
            response = self.session.post(url)
            response.raise_for_status()
            return response.json()
        except requests.HTTPError as e:
            logger.error(f"Environment reset failed: {e}")
            raise

    def initialize_scenario(self, domain: str, task_id: str, task_split: str = "base") -> dict:
        """
        Initialize a complete scenario: load task, reset environment, apply initial state.

        Args:
            domain: Domain name (airline, retail, telecom)
            task_id: Task ID to load
            task_split: Task split (base, dev, test)

        Returns:
            Dict with status, domain, task_id, and initial_greeting

        Raises:
            requests.HTTPError: If the request fails
        """
        url = f"{self.base_url}/initialize_scenario"

        payload = {
            "domain": domain,
            "task_id": task_id,
            "task_split": task_split
        }

        try:
            response = self.session.post(url, json=payload)
            response.raise_for_status()
            return response.json()
        except requests.HTTPError as e:
            logger.error(f"Scenario initialization failed: {e}")
            if e.response is not None:
                logger.error(f"Response: {e.response.text}")
            raise

    def send_message(self, message: str) -> dict:
        """
        Send a message to the simulated user and get their response.

        Args:
            message: Agent's message to send

        Returns:
            Dict with user_message and role

        Raises:
            requests.HTTPError: If the request fails
        """
        url = f"{self.base_url}/send_message"

        payload = {"message": message}

        try:
            response = self.session.post(url, json=payload)
            response.raise_for_status()
            return response.json()
        except requests.HTTPError as e:
            logger.error(f"send_message failed: {e}")
            if e.response is not None:
                logger.error(f"Response: {e.response.text}")
            raise


# Global HTTP client instance
_http_client: Optional[EnvironmentHTTPClient] = None


def get_http_client(base_url: Optional[str] = None) -> EnvironmentHTTPClient:
    """
    Get or create the global HTTP client instance.

    Args:
        base_url: Optional base URL for the environment server

    Returns:
        EnvironmentHTTPClient instance
    """
    global _http_client

    if _http_client is None or (base_url and _http_client.base_url != base_url):
        _http_client = EnvironmentHTTPClient(base_url=base_url)

    return _http_client
