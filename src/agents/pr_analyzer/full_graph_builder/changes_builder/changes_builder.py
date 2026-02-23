"""Changes builder — describes what changed in updated functions by comparing old and new source.

One API call per source file, processing all updated functions in that file at once.
Source code is embedded in the prompt (no Read tool needed).
"""

import json
import re
import time
from pathlib import Path

from llm import get_provider

AGENT_DIR = Path(__file__).parent
AGENT_NAME = "changes_builder"

_provider = None


def _get_provider():
    """Lazily create a shared LLM provider."""
    global _provider
    if _provider is None:
        _provider = get_provider()
    return _provider


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def _build_prompt(file_path: str, nodes: list[dict]) -> str:
    """Build the prompt for a single-file changes_builder with inline source."""
    parts: list[str] = [
        f"Compare the old and new versions of these functions from `{file_path}` and describe what changed.\n",
    ]
    for i, node in enumerate(nodes, 1):
        func_name = node["id"].split("::")[-1]
        old_source = node.get("base_source", "")
        new_source = node.get("source", "")
        parts.append(
            f"### {i}. `{func_name}`\n"
            f"**Old version:**\n```\n{old_source}\n```\n"
            f"**New version:**\n```\n{new_source}\n```\n"
        )
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
        node["update"] = entry.get("update", "")
        completed += 1

    return completed, errors


def _extract_json(text: str) -> list[dict] | None:
    """Extract a JSON array from text output."""
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
# API caller
# ---------------------------------------------------------------------------

def _load_md(filename: str) -> str:
    return (AGENT_DIR / filename).read_text()


async def run_changes_builder_for_file(
    file_path: str,
    nodes: list[dict],
) -> dict:
    """Run the changes_builder for all updated nodes in a single source file.

    Each node must have both "source" (new) and "base_source" (old) fields.
    Applies the "update" field directly to each node dict.

    Returns a dict with log lines and stats for consolidated logging.
    """
    system_prompt = _load_md("system_prompt.md")
    prompt = _build_prompt(file_path, nodes)

    start_time = time.time()
    errors: list[str] = []
    log_lines: list[str] = []
    stats = {"duration_ms": 0, "cost_usd": 0.0, "num_turns": 1}

    try:
        provider = _get_provider()
        response = await provider.complete(
            system_prompt=system_prompt,
            user_prompt=prompt,
            max_tokens=4096,
        )

        log_lines.append(response.text)
        stats["cost_usd"] = response.cost_usd

        result_json = _extract_json(response.text)
        if result_json:
            completed, apply_errors = _apply_results(nodes, result_json)
            errors.extend(apply_errors)
            log_lines.append(f"Applied {completed}/{len(nodes)} changes")
        else:
            errors.append("Could not parse JSON from API output")

    except Exception as e:
        errors.append(f"Exception: {e}")

    wall_ms = int((time.time() - start_time) * 1000)
    stats["duration_ms"] = wall_ms

    success = len(errors) == 0
    return {
        "node_id": file_path,
        "success": success,
        "log_lines": log_lines,
        "stats": stats,
        "errors": errors,
    }
