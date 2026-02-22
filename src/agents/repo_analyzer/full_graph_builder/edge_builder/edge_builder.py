"""Edge builder agent — describes edges from a single caller to all its callees."""

import json
import re
import time
from collections.abc import AsyncIterator
from pathlib import Path

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ResultMessage,
    TextBlock,
    ToolUseBlock,
    UserMessage,
    query,
)

AGENT_DIR = Path(__file__).parent
AGENT_NAME = "edge_builder"


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def _build_prompt(
    caller_id: str,
    caller_source: str,
    callees: list[dict],
) -> str:
    """Build the prompt for a batched edge builder agent.

    Args:
        caller_id: The full node ID of the caller.
        caller_source: Source code of the caller function.
        callees: List of dicts with keys: edge_id (int), callee_id (str), callee_source (str).
    """
    caller_name = caller_id.split("::")[-1]
    caller_file = caller_id.split("::")[0]

    parts: list[str] = [
        f"Analyze these function calls from the caller and output a JSON array.\n",
        f"Caller: `{caller_name}` (in `{caller_file}`)\n```\n{caller_source}\n```\n",
        f"Callees:\n",
    ]

    for c in callees:
        callee_name = c["callee_id"].split("::")[-1]
        callee_file = c["callee_id"].split("::")[0]
        parts.append(
            f"### Edge {c['edge_id']}: `{callee_name}` (in `{callee_file}`)\n"
            f"```\n{c['callee_source']}\n```\n"
        )

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Result parser
# ---------------------------------------------------------------------------

def _extract_json_array(text: str) -> list[dict] | None:
    """Extract a JSON array from agent text output."""
    match = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    return None


def _apply_results(
    edges: list[dict],
    result_json: list[dict],
) -> tuple[int, list[str]]:
    """Apply parsed JSON results to edge dicts by id. Returns (completed, errors)."""
    result_by_id = {entry["id"]: entry for entry in result_json if "id" in entry}

    completed = 0
    errors = []

    for i, edge in enumerate(edges, 1):
        entry = result_by_id.get(i)
        if entry is None:
            errors.append(f"Missing result for edge {i}: {edge['from']} -> {edge['to']}")
            continue
        edge["label"] = entry.get("label", "")
        edge["description"] = entry.get("description", "")
        edge["args"] = entry.get("args", "")
        # Only accept upgrading is_returned from False→True, never downgrade
        agent_is_returned = entry.get("is_returned", False)
        edge["is_returned"] = edge.get("is_returned", True) or agent_is_returned
        edge["condition"] = entry.get("condition")
        edge["index"] = entry.get("index", 0)
        completed += 1

    return completed, errors


# ---------------------------------------------------------------------------
# Agent runner
# ---------------------------------------------------------------------------

def _load_md(filename: str) -> str:
    return (AGENT_DIR / filename).read_text()


async def _as_stream(prompt_text: str) -> AsyncIterator[dict]:
    yield {
        "type": "user",
        "session_id": "",
        "message": {"role": "user", "content": prompt_text},
        "parent_tool_use_id": None,
    }


async def run_edge_builder_for_caller(
    caller_id: str,
    caller_source: str,
    edges: list[dict],
    node_index: dict[str, dict],
) -> dict:
    """Run the edge_builder agent for all edges from a single caller.

    Args:
        caller_id: The full node ID of the caller.
        caller_source: Source code of the caller function.
        edges: List of edge dicts (each has "from" and "to" keys).
        node_index: Mapping of node ID to node dict (to look up callee source).

    Applies results (label, description, args, is_returned, condition, index)
    directly to each edge dict.

    Returns a dict with log lines and stats for consolidated logging.
    """
    # Build callee info list
    callees: list[dict] = []
    for i, edge in enumerate(edges, 1):
        callee = node_index.get(edge["to"])
        callee_source = callee.get("source", "") if callee else ""
        callees.append({
            "edge_id": i,
            "callee_id": edge["to"],
            "callee_source": callee_source,
        })

    system_prompt = _load_md("system_prompt.md")
    prompt = _build_prompt(caller_id, caller_source, callees)

    options = ClaudeAgentOptions(
        model="haiku",
        system_prompt=system_prompt,
        allowed_tools=[],
        permission_mode="bypassPermissions",
        max_turns=1,
    )

    log_lines: list[str] = []
    all_text: list[str] = []
    start_time = time.time()
    errors: list[str] = []
    stats = {"duration_ms": 0, "cost_usd": 0.0, "num_turns": 0}

    try:
        async for message in query(prompt=_as_stream(prompt), options=options):
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        log_lines.append(block.text)
                        all_text.append(block.text)

            elif isinstance(message, ResultMessage):
                stats["duration_ms"] = message.duration_ms or 0
                stats["cost_usd"] = message.total_cost_usd or 0.0
                stats["num_turns"] = message.num_turns or 0

                if message.is_error:
                    errors.append(f"Agent error: {message.result}")
                else:
                    full_text = "\n".join(all_text)
                    result_json = _extract_json_array(full_text)
                    if result_json:
                        completed, apply_errors = _apply_results(edges, result_json)
                        errors.extend(apply_errors)
                        log_lines.append(f"Applied {completed}/{len(edges)} edges")
                    else:
                        errors.append("Could not parse JSON from agent output")
    except Exception as e:
        errors.append(f"Exception: {e}")

    wall_ms = int((time.time() - start_time) * 1000)
    if stats["duration_ms"] == 0:
        stats["duration_ms"] = wall_ms

    success = len(errors) == 0
    return {
        "edge_key": caller_id,
        "success": success,
        "log_lines": log_lines,
        "stats": stats,
        "errors": errors,
    }
