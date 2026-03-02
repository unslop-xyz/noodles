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
async def analyze_pr(pr_url: str, enrich: bool = False) -> dict:
    """Analyze a GitHub PR and generate call graph diagram.

    Args:
        pr_url: GitHub PR URL (e.g., https://github.com/owner/repo/pull/123)
        enrich: If True, use LLM to add descriptions (slower, costs API).
                Default False for fast analysis.

    Returns:
        Analysis result with main_diagram (Mermaid), summary stats,
        and analysis_id for follow-up queries.
    """
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
    return _build_analysis_result(result_dir, analysis_id)


@mcp.tool()
async def analyze_repo(repo_url: str, enrich: bool = False) -> dict:
    """Analyze entire repo structure and generate call graph.

    Args:
        repo_url: GitHub repo URL (e.g., https://github.com/owner/repo)
        enrich: If True, use LLM to add descriptions (slower, costs API).
                Default False for fast analysis.

    Returns:
        Analysis result with main_diagram (Mermaid), summary stats,
        and analysis_id for follow-up queries.
    """
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
    return _build_analysis_result(result_dir, analysis_id)


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


def main():
    """Run the MCP server."""
    mcp.run()


if __name__ == "__main__":
    main()
