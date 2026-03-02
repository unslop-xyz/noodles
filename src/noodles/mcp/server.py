"""MCP server for noodles - expose PR/repo analysis to AI agents.

Usage:
    # Run with MCP inspector for testing
    mcp dev src/noodles/mcp/server.py

    # Or via CLI
    noodles mcp
"""

import json
import tempfile
import uuid
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from noodles.agents.pr_analyzer.pr_analyzer import (
    analyze_pr as _analyze_pr,
    analyze_local_changes as _analyze_local_changes,
)
from noodles.agents.repo_analyzer.repo_analyzer import (
    analyze_repo as _analyze_repo,
    analyze_local_repo as _analyze_local_repo,
)
from noodles.mcp.graph_queries import (
    find_path as _find_path,
    get_callers as _get_callers,
    get_callees as _get_callees,
    filter_by_type,
    resolve_function_id,
    make_short_id,
    make_file_line,
)
from noodles.mcp.cache import (
    make_cache_key,
    check_cache,
    store_in_cache,
    copy_cached_result,
    get_pr_head_sha,
    get_repo_head_sha,
)

mcp = FastMCP(name="noodles")

# Store analysis results keyed by analysis_id
_analyses: dict[str, Path] = {}


def _load_diagram(result_dir: Path, diagram_name: str) -> str | None:
    """Load a diagram file from the result directory."""
    diagram_file = result_dir / f"diagram_{diagram_name}.mmd"
    if diagram_file.exists():
        return diagram_file.read_text()
    return None


def _load_call_graph(result_dir: Path) -> dict | None:
    """Load the call graph JSON from the result directory."""
    call_graph_file = result_dir / "call_graph.json"
    if call_graph_file.exists():
        return json.loads(call_graph_file.read_text())
    return None


def _list_diagrams(result_dir: Path) -> list[str]:
    """List all available diagram names in the result directory."""
    diagrams = []
    for f in result_dir.glob("diagram_*.mmd"):
        name = f.stem.replace("diagram_", "")
        diagrams.append(name)
    return sorted(diagrams)


def _build_analysis_result(result_dir: Path, analysis_id: str) -> dict:
    """Build a structured result from an analysis directory."""
    call_graph = _load_call_graph(result_dir)
    main_diagram = _load_diagram(result_dir, "main")
    available_diagrams = _list_diagrams(result_dir)

    result = {
        "analysis_id": analysis_id,
        "result_path": str(result_dir),
        "available_diagrams": available_diagrams,
    }

    if call_graph:
        result["summary"] = {
            "node_count": len(call_graph.get("nodes", [])),
            "edge_count": len(call_graph.get("edges", [])),
        }

    if main_diagram:
        result["main_diagram"] = main_diagram

    # Include classification for PR analysis
    classification_file = result_dir / "classification.json"
    if classification_file.exists():
        classification = json.loads(classification_file.read_text())
        result["changes"] = {
            "new_functions": len(classification.get("new", [])),
            "updated_functions": len(classification.get("updated", [])),
            "deleted_functions": len(classification.get("deleted", [])),
        }

    return result


@mcp.tool()
async def analyze_pr(
    pr_url: str,
    enrich: bool = False,
    force_refresh: bool = False,
) -> dict:
    """Analyze a GitHub PR and generate call graph diagram.

    Args:
        pr_url: GitHub PR URL (e.g., https://github.com/owner/repo/pull/123)
        enrich: If True, use LLM to add descriptions (slower, costs API).
                Default False for fast analysis.
        force_refresh: If True, skip cache and re-analyze.

    Returns:
        Analysis result with main_diagram (Mermaid), summary stats,
        and analysis_id for follow-up queries.
    """
    import re

    # Parse PR URL for cache key
    match = re.match(r"https?://github\.com/([^/]+)/([^/]+)/pull/(\d+)", pr_url)
    if not match:
        return {"error": f"Invalid PR URL: {pr_url}"}

    owner, repo, pr_number = match.group(1), match.group(2), int(match.group(3))

    # Get head SHA for cache key
    head_sha = get_pr_head_sha(owner, repo, pr_number)

    # Check cache unless force_refresh
    if not force_refresh and head_sha:
        cache_key = make_cache_key(pr_url, pr_number=pr_number, head_sha=head_sha)
        cached = check_cache(cache_key)
        if cached:
            # Return cached result with new analysis_id
            analysis_id = uuid.uuid4().hex[:12]
            output_dir = Path(tempfile.gettempdir()) / "noodles_mcp"
            output_dir.mkdir(parents=True, exist_ok=True)

            result_dir = copy_cached_result(cached, analysis_id, output_dir)
            _analyses[analysis_id] = result_dir

            result = _build_analysis_result(result_dir, analysis_id)
            result["cached"] = True
            return result

    # Run analysis
    analysis_id = uuid.uuid4().hex[:12]
    output_dir = Path(tempfile.gettempdir()) / "noodles_mcp"
    output_dir.mkdir(parents=True, exist_ok=True)

    result_dir = await _analyze_pr(
        pr_url,
        output_dir=output_dir,
        analysis_id=analysis_id,
        enrich=enrich,
    )

    if result_dir is None:
        return {"error": f"Failed to analyze PR: {pr_url}"}

    _analyses[analysis_id] = result_dir

    # Store in cache
    if head_sha:
        cache_key = make_cache_key(pr_url, pr_number=pr_number, head_sha=head_sha)
        store_in_cache(cache_key, analysis_id, result_dir)

    result = _build_analysis_result(result_dir, analysis_id)
    result["cached"] = False
    return result


@mcp.tool()
async def analyze_repo(
    repo_url: str,
    enrich: bool = False,
    force_refresh: bool = False,
) -> dict:
    """Analyze entire repo structure and generate call graph.

    Args:
        repo_url: GitHub repo URL (e.g., https://github.com/owner/repo)
        enrich: If True, use LLM to add descriptions (slower, costs API).
                Default False for fast analysis.
        force_refresh: If True, skip cache and re-analyze.

    Returns:
        Analysis result with main_diagram (Mermaid), summary stats,
        and analysis_id for follow-up queries.
    """
    # Get head SHA for cache key
    head_sha = get_repo_head_sha(repo_url)

    # Check cache unless force_refresh
    if not force_refresh and head_sha:
        cache_key = make_cache_key(repo_url, head_sha=head_sha)
        cached = check_cache(cache_key)
        if cached:
            # Return cached result with new analysis_id
            analysis_id = uuid.uuid4().hex[:12]
            output_dir = Path(tempfile.gettempdir()) / "noodles_mcp"
            output_dir.mkdir(parents=True, exist_ok=True)

            result_dir = copy_cached_result(cached, analysis_id, output_dir)
            _analyses[analysis_id] = result_dir

            result = _build_analysis_result(result_dir, analysis_id)
            result["cached"] = True
            return result

    # Run analysis
    analysis_id = uuid.uuid4().hex[:12]
    output_dir = Path(tempfile.gettempdir()) / "noodles_mcp"
    output_dir.mkdir(parents=True, exist_ok=True)

    result_dir = await _analyze_repo(
        repo_url,
        output_dir=output_dir,
        analysis_id=analysis_id,
        enrich=enrich,
    )

    if result_dir is None:
        return {"error": f"Failed to analyze repo: {repo_url}"}

    _analyses[analysis_id] = result_dir

    # Store in cache
    if head_sha:
        cache_key = make_cache_key(repo_url, head_sha=head_sha)
        store_in_cache(cache_key, analysis_id, result_dir)

    result = _build_analysis_result(result_dir, analysis_id)
    result["cached"] = False
    return result


@mcp.tool()
async def analyze_local_repo(repo_path: str, enrich: bool = False) -> dict:
    """Analyze a local repository (no cloning needed).

    Args:
        repo_path: Absolute path to local git repo
        enrich: If True, use LLM to add descriptions (slower, costs API).
                Default False for fast analysis.

    Returns:
        Analysis result with main_diagram (Mermaid), summary stats,
        and analysis_id for follow-up queries.
    """
    analysis_id = uuid.uuid4().hex[:12]
    output_dir = Path(tempfile.gettempdir()) / "noodles_mcp" / f"result_{analysis_id}"
    output_dir.mkdir(parents=True, exist_ok=True)

    result_dir = await _analyze_local_repo(
        Path(repo_path),
        output_dir=output_dir,
        enrich=enrich,
    )

    _analyses[analysis_id] = result_dir
    return _build_analysis_result(result_dir, analysis_id)


@mcp.tool()
async def analyze_changes(
    head_path: str,
    base_path: str | None = None,
    enrich: bool = False,
) -> dict:
    """Analyze local changes (uncommitted/staged) before creating a PR.

    Args:
        head_path: Path to the repo with changes (working directory)
        base_path: Path to the base version. If not provided, uses HEAD~1.
        enrich: If True, use LLM to add descriptions (slower, costs API).
                Default False for fast analysis.

    Returns:
        Analysis result with main_diagram (Mermaid), change summary,
        and analysis_id for follow-up queries.
    """
    import subprocess

    analysis_id = uuid.uuid4().hex[:12]
    output_dir = Path(tempfile.gettempdir()) / "noodles_mcp" / f"result_{analysis_id}"
    output_dir.mkdir(parents=True, exist_ok=True)

    head = Path(head_path)

    if base_path:
        base = Path(base_path)
    else:
        # Create a temporary worktree at HEAD~1 for comparison
        base = output_dir / "base"
        result = subprocess.run(
            ["git", "worktree", "add", str(base), "HEAD~1"],
            cwd=head,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            return {"error": f"Failed to create base worktree: {result.stderr}"}

    result_dir = await _analyze_local_changes(
        base_path=base,
        head_path=head,
        output_dir=output_dir,
        enrich=enrich,
    )

    _analyses[analysis_id] = result_dir
    return _build_analysis_result(result_dir, analysis_id)


@mcp.tool()
def get_diagram(analysis_id: str, diagram_name: str = "main") -> dict:
    """Get Mermaid diagram text from a previous analysis.

    Args:
        analysis_id: The analysis_id returned from a previous analyze_* call
        diagram_name: Name of the diagram (default "main"). Use
                      available_diagrams from analysis result to see options.

    Returns:
        Dictionary with diagram content or error message.
    """
    if analysis_id not in _analyses:
        return {
            "error": f"Analysis not found: {analysis_id}. "
            "Run an analyze_* tool first."
        }

    result_dir = _analyses[analysis_id]
    diagram = _load_diagram(result_dir, diagram_name)

    if diagram is None:
        available = _list_diagrams(result_dir)
        return {
            "error": f"Diagram '{diagram_name}' not found",
            "available_diagrams": available,
        }

    return {
        "analysis_id": analysis_id,
        "diagram_name": diagram_name,
        "diagram": diagram,
    }


@mcp.tool()
def get_call_graph(analysis_id: str) -> dict:
    """Get full call graph JSON with all nodes and edges.

    Args:
        analysis_id: The analysis_id returned from a previous analyze_* call

    Returns:
        The complete call graph with nodes (functions) and edges (calls).
    """
    if analysis_id not in _analyses:
        return {
            "error": f"Analysis not found: {analysis_id}. "
            "Run an analyze_* tool first."
        }

    result_dir = _analyses[analysis_id]
    call_graph = _load_call_graph(result_dir)

    if call_graph is None:
        return {"error": "Call graph not found in analysis results"}

    return {
        "analysis_id": analysis_id,
        "call_graph": call_graph,
    }


@mcp.tool()
def find_path(analysis_id: str, from_function: str, to_function: str) -> dict:
    """Find call path between two functions.

    Args:
        analysis_id: The analysis_id returned from a previous analyze_* call
        from_function: Starting function (full ID or partial name like "analyze_pr")
        to_function: Target function (full ID or partial name)

    Returns:
        Dictionary with path as list of function IDs, or error if no path.
    """
    if analysis_id not in _analyses:
        return {
            "error": f"Analysis not found: {analysis_id}. "
            "Run an analyze_* tool first."
        }

    result_dir = _analyses[analysis_id]
    call_graph = _load_call_graph(result_dir)

    if call_graph is None:
        return {"error": "Call graph not found in analysis results"}

    # Resolve partial function names
    from_id = resolve_function_id(from_function, call_graph)
    to_id = resolve_function_id(to_function, call_graph)

    if from_id is None:
        return {"error": f"Function not found: {from_function}"}
    if to_id is None:
        return {"error": f"Function not found: {to_function}"}

    path = _find_path(call_graph, from_id, to_id)

    if path is None:
        return {
            "error": f"No path from {make_short_id(from_id)} to {make_short_id(to_id)}",
            "from": from_id,
            "to": to_id,
        }

    # Build path with file:line info
    node_map = {n["id"]: n for n in call_graph.get("nodes", [])}
    path_info = []
    for nid in path:
        node = node_map.get(nid, {"id": nid})
        path_info.append({
            "id": nid,
            "short_id": make_short_id(nid),
            "file_line": make_file_line(node),
        })

    return {
        "analysis_id": analysis_id,
        "path": path_info,
        "length": len(path),
    }


@mcp.tool()
def get_callers(analysis_id: str, function_id: str, depth: int = 1) -> dict:
    """Get all functions that call this function (transitive).

    Args:
        analysis_id: The analysis_id returned from a previous analyze_* call
        function_id: Function to find callers for (full ID or partial name)
        depth: Maximum depth to traverse (default 1 for direct callers only)

    Returns:
        Dictionary with list of callers and their depths.
    """
    if analysis_id not in _analyses:
        return {
            "error": f"Analysis not found: {analysis_id}. "
            "Run an analyze_* tool first."
        }

    result_dir = _analyses[analysis_id]
    call_graph = _load_call_graph(result_dir)

    if call_graph is None:
        return {"error": "Call graph not found in analysis results"}

    # Resolve partial function name
    resolved_id = resolve_function_id(function_id, call_graph)
    if resolved_id is None:
        return {"error": f"Function not found: {function_id}"}

    callers = _get_callers(call_graph, resolved_id, depth)

    return {
        "analysis_id": analysis_id,
        "function": make_short_id(resolved_id),
        "function_full_id": resolved_id,
        "callers": callers,
        "count": len(callers),
    }


@mcp.tool()
def get_callees(analysis_id: str, function_id: str, depth: int = 1) -> dict:
    """Get all functions this function calls (transitive).

    Args:
        analysis_id: The analysis_id returned from a previous analyze_* call
        function_id: Function to find callees for (full ID or partial name)
        depth: Maximum depth to traverse (default 1 for direct calls only)

    Returns:
        Dictionary with list of callees and their depths.
    """
    if analysis_id not in _analyses:
        return {
            "error": f"Analysis not found: {analysis_id}. "
            "Run an analyze_* tool first."
        }

    result_dir = _analyses[analysis_id]
    call_graph = _load_call_graph(result_dir)

    if call_graph is None:
        return {"error": "Call graph not found in analysis results"}

    # Resolve partial function name
    resolved_id = resolve_function_id(function_id, call_graph)
    if resolved_id is None:
        return {"error": f"Function not found: {function_id}"}

    callees = _get_callees(call_graph, resolved_id, depth)

    return {
        "analysis_id": analysis_id,
        "function": make_short_id(resolved_id),
        "function_full_id": resolved_id,
        "callees": callees,
        "count": len(callees),
    }


@mcp.tool()
def filter_graph(analysis_id: str, filter_type: str) -> dict:
    """Filter nodes by type: entry_points, endpoints, new, updated, orphans.

    Args:
        analysis_id: The analysis_id returned from a previous analyze_* call
        filter_type: One of "entry_points", "endpoints", "new", "updated", "orphans"

    Returns:
        Dictionary with filtered list of functions.
    """
    if analysis_id not in _analyses:
        return {
            "error": f"Analysis not found: {analysis_id}. "
            "Run an analyze_* tool first."
        }

    result_dir = _analyses[analysis_id]
    call_graph = _load_call_graph(result_dir)

    if call_graph is None:
        return {"error": "Call graph not found in analysis results"}

    valid_types = ["entry_points", "endpoints", "new", "updated", "orphans"]
    if filter_type not in valid_types:
        return {
            "error": f"Invalid filter_type: {filter_type}. "
            f"Must be one of: {', '.join(valid_types)}"
        }

    nodes = filter_by_type(call_graph, filter_type)

    return {
        "analysis_id": analysis_id,
        "filter_type": filter_type,
        "nodes": nodes,
        "count": len(nodes),
    }


@mcp.tool()
def get_summary(analysis_id: str) -> dict:
    """Get plain English summary of analysis impact.

    Args:
        analysis_id: The analysis_id returned from a previous analyze_* call

    Returns:
        Dictionary with summary text, metrics, and impact level.
    """
    if analysis_id not in _analyses:
        return {
            "error": f"Analysis not found: {analysis_id}. "
            "Run an analyze_* tool first."
        }

    result_dir = _analyses[analysis_id]
    call_graph = _load_call_graph(result_dir)

    if call_graph is None:
        return {"error": "Call graph not found in analysis results"}

    nodes = call_graph.get("nodes", [])
    total = len(nodes)

    # Count by status and type
    new_funcs = [n for n in nodes if n.get("status") == "new"]
    updated_funcs = [n for n in nodes if n.get("status") == "updated"]
    entry_points = [n for n in nodes if n.get("type") == "start_point"]

    new_count = len(new_funcs)
    updated_count = len(updated_funcs)

    # Find affected callers (unchanged functions that call new/updated)
    changed_ids = {n["id"] for n in new_funcs + updated_funcs}
    affected_callers = set()
    for node in nodes:
        if node["id"] not in changed_ids:
            for callee in node.get("callees", []):
                if callee in changed_ids:
                    affected_callers.add(node["id"])
                    break

    # Find affected entry points
    affected_entries = [
        make_short_id(n["id"])
        for n in entry_points
        if n["id"] in affected_callers or n["id"] in changed_ids
    ]

    # Determine impact level
    if new_count + updated_count == 0:
        impact = "none"
    elif len(affected_callers) > 10 or len(affected_entries) > 3:
        impact = "high"
    elif len(affected_callers) > 3 or len(affected_entries) > 1:
        impact = "medium"
    else:
        impact = "low"

    # Build summary text
    parts = []
    if new_count > 0:
        parts.append(f"adds {new_count} new function{'s' if new_count > 1 else ''}")
    if updated_count > 0:
        parts.append(f"updates {updated_count} function{'s' if updated_count > 1 else ''}")

    if parts:
        summary = f"This analysis {' and '.join(parts)}."
        if affected_callers:
            summary += f" {len(affected_callers)} other function{'s' if len(affected_callers) > 1 else ''} call{'s' if len(affected_callers) == 1 else ''} the modified code."
    else:
        summary = f"This analysis covers {total} functions with no detected changes."

    return {
        "analysis_id": analysis_id,
        "summary": summary,
        "metrics": {
            "total_functions": total,
            "new_functions": new_count,
            "updated_functions": updated_count,
            "affected_callers": len(affected_callers),
            "entry_points_affected": affected_entries,
        },
        "impact_level": impact,
    }


@mcp.tool()
def get_changes(analysis_id: str) -> dict:
    """Get detailed change info for modified functions.

    Args:
        analysis_id: The analysis_id returned from a previous analyze_* call

    Returns:
        Dictionary with lists of updated, new, and deleted functions,
        each with file_line and caller info.
    """
    if analysis_id not in _analyses:
        return {
            "error": f"Analysis not found: {analysis_id}. "
            "Run an analyze_* tool first."
        }

    result_dir = _analyses[analysis_id]
    call_graph = _load_call_graph(result_dir)

    if call_graph is None:
        return {"error": "Call graph not found in analysis results"}

    # Load classification for deleted functions
    classification_file = result_dir / "classification.json"
    deleted_ids = []
    if classification_file.exists():
        classification = json.loads(classification_file.read_text())
        deleted_ids = classification.get("deleted", [])

    nodes = call_graph.get("nodes", [])

    updated_functions = []
    new_functions = []

    for node in nodes:
        nid = node["id"]
        status = node.get("status", "")
        file_line = make_file_line(node)
        short_id = make_short_id(nid)

        # Get callers for this function
        callers = [make_short_id(c) for c in node.get("callers", [])]

        info = {
            "id": short_id,
            "full_id": nid,
            "file_line": file_line,
            "callers": callers,
            "caller_count": len(callers),
        }

        if status == "updated":
            updated_functions.append(info)
        elif status == "new":
            new_functions.append(info)

    # Format deleted functions
    deleted_functions = [
        {"id": make_short_id(d), "full_id": d}
        for d in deleted_ids
    ]

    return {
        "analysis_id": analysis_id,
        "updated_functions": updated_functions,
        "new_functions": new_functions,
        "deleted_functions": deleted_functions,
    }


def main():
    """Run the MCP server."""
    mcp.run()


if __name__ == "__main__":
    main()
