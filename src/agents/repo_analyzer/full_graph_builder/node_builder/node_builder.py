"""Node builder agent — describes nodes in a call graph with human-readable names, descriptions, and tags.

One agent per source file, processing all functions in that file at once.
Source code is embedded in the prompt (no Read tool needed).
"""

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
AGENT_NAME = "node_builder"


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def _build_prompt(file_path: str, nodes: list[dict]) -> str:
    """Build the prompt for a single-file node_builder agent with inline source."""
    parts: list[str] = [
        f"Analyze these functions from `{file_path}` and output a JSON array.\n",
    ]
    for i, node in enumerate(nodes, 1):
        func_name = node["id"].split("::")[-1]
        source = node.get("source", "")
        node_type = node.get("type", "process")
        parts.append(f"### {i}. `{func_name}` (type: {node_type})\n```\n{source}\n```\n")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Result parser
# ---------------------------------------------------------------------------

def _apply_results(nodes: list[dict], result_json: list[dict]) -> tuple[int, list[str]]:
    """Apply parsed JSON results to node dicts by index. Returns (completed, errors)."""
    result_by_id = {entry["id"]: entry for entry in result_json if "id" in entry}

    completed = 0
    errors = []

    for i, node in enumerate(nodes, 1):
        entry = result_by_id.get(i)
        if entry is None:
            errors.append(f"Missing result for function {i}: {node['id']}")
            continue
        node["name"] = entry.get("name", "")
        node["description"] = entry.get("description", "")
        node["tag"] = entry.get("tag", "source")
        completed += 1

    return completed, errors


def _extract_json(text: str) -> list[dict] | None:
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


async def run_node_builder_for_file(
    file_path: str,
    nodes: list[dict],
) -> dict:
    """Run the node_builder agent for all nodes in a single source file.

    Source code is embedded inline in the prompt — no Read tool needed.
    Applies results (name, description, tag) directly to each node dict.

    Returns a dict with log lines and stats for consolidated logging.
    """
    system_prompt = _load_md("system_prompt.md")
    prompt = _build_prompt(file_path, nodes)

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
                    result_json = _extract_json(full_text)
                    if result_json:
                        completed, apply_errors = _apply_results(nodes, result_json)
                        errors.extend(apply_errors)
                        log_lines.append(f"Applied {completed}/{len(nodes)} nodes")
                    else:
                        errors.append("Could not parse JSON from agent output")
    except Exception as e:
        errors.append(f"Exception: {e}")

    wall_ms = int((time.time() - start_time) * 1000)
    if stats["duration_ms"] == 0:
        stats["duration_ms"] = wall_ms

    success = len(errors) == 0
    return {
        "node_id": file_path,
        "success": success,
        "log_lines": log_lines,
        "stats": stats,
        "errors": errors,
    }
