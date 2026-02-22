"""Edge builder — describes edges from a single caller to all its callees.

Uses the Anthropic API directly for minimal overhead.
"""

import json
import re
import time
from pathlib import Path

import anthropic

AGENT_DIR = Path(__file__).parent
AGENT_NAME = "edge_builder"
MODEL = "claude-haiku-4-5-20251001"

_client: anthropic.AsyncAnthropic | None = None


def _get_client() -> anthropic.AsyncAnthropic:
    """Lazily create a shared async client."""
    global _client
    if _client is None:
        _client = anthropic.AsyncAnthropic()
    return _client


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def _build_prompt(
    caller_id: str,
    caller_source: str,
    callees: list[dict],
) -> str:
    """Build the prompt for a batched edge builder.

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
        completed += 1

    return completed, errors


# ---------------------------------------------------------------------------
# API caller
# ---------------------------------------------------------------------------

def _load_md(filename: str) -> str:
    return (AGENT_DIR / filename).read_text()


async def run_edge_builder_for_caller(
    caller_id: str,
    caller_source: str,
    edges: list[dict],
    node_index: dict[str, dict],
) -> dict:
    """Run the edge_builder for all edges from a single caller.

    Args:
        caller_id: The full node ID of the caller.
        caller_source: Source code of the caller function.
        edges: List of edge dicts (each has "from" and "to" keys).
        node_index: Mapping of node ID to node dict (to look up callee source).

    Applies results (label, description, args, is_returned, condition)
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

    start_time = time.time()
    errors: list[str] = []
    log_lines: list[str] = []
    stats = {"duration_ms": 0, "cost_usd": 0.0, "num_turns": 1}

    try:
        client = _get_client()
        response = await client.messages.create(
            model=MODEL,
            max_tokens=4096,
            system=system_prompt,
            messages=[{"role": "user", "content": prompt}],
        )

        full_text = "".join(
            block.text for block in response.content if block.type == "text"
        )
        log_lines.append(full_text)

        # Compute cost from usage
        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens
        # Haiku pricing: $0.80/M input, $4/M output
        stats["cost_usd"] = (input_tokens * 0.80 + output_tokens * 4.0) / 1_000_000

        result_json = _extract_json_array(full_text)
        if result_json:
            completed, apply_errors = _apply_results(edges, result_json)
            errors.extend(apply_errors)
            log_lines.append(f"Applied {completed}/{len(edges)} edges")
        else:
            errors.append("Could not parse JSON from API output")

    except Exception as e:
        errors.append(f"Exception: {e}")

    wall_ms = int((time.time() - start_time) * 1000)
    stats["duration_ms"] = wall_ms

    success = len(errors) == 0
    return {
        "edge_key": caller_id,
        "success": success,
        "log_lines": log_lines,
        "stats": stats,
        "errors": errors,
    }
