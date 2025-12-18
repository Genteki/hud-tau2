"""Global state management for tau2-bench MCP server.

This module holds the global tau2_task instance to avoid circular imports.
"""

from task import Tau2Task

# Global task state
tau2_task = Tau2Task()


def get_tau2_task() -> Tau2Task:
    """Get the global Tau2Task instance."""
    return tau2_task
