"""Sanitization utilities for identifiers and text."""

import re


def sanitize_id(node_id: str) -> str:
    """Convert a call graph node ID to a valid mermaid identifier.

    Replaces non-alphanumeric characters (except underscores) with underscores,
    collapses consecutive underscores, and strips leading/trailing underscores.

    Args:
        node_id: The original node ID string.

    Returns:
        A sanitized identifier safe for mermaid diagrams.
        Returns "node" if the result would be empty.
    """
    s = re.sub(r"[^a-zA-Z0-9_]", "_", node_id)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "node"
