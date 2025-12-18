"""Toolkit utilities for tau2-bench environment server.

Provides helper functions to extract tool signatures from tau2-bench tools.
"""

from typing import Dict, Any
from dataclasses import dataclass


@dataclass
class ToolSignature:
    """Tool signature metadata."""
    name: str
    doc: str
    params: Dict[str, Any]
    returns: Dict[str, Any]


def get_tool_signatures(tools: list) -> Dict[str, ToolSignature]:
    """
    Extract tool signatures from a list of tau2 Tool objects.

    Args:
        tools: List of Tool objects from tau2-bench

    Returns:
        Dictionary mapping tool names to ToolSignature objects
    """
    signatures = {}

    for tool in tools:
        # Extract parameters schema
        params_schema = {}
        if hasattr(tool, 'params') and hasattr(tool.params, 'model_json_schema'):
            params_schema = tool.params.model_json_schema()

        # Extract returns schema
        returns_schema = {}
        if hasattr(tool, 'returns') and hasattr(tool.returns, 'model_json_schema'):
            returns_schema = tool.returns.model_json_schema()

        signatures[tool.name] = ToolSignature(
            name=tool.name,
            doc=tool.description or tool.__doc__ or "",
            params=params_schema,
            returns=returns_schema,
        )

    return signatures
