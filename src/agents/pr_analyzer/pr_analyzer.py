"""Analyze a GitHub PR and build a pruned, annotated call graph.

Usage:
    python pr_analyzer.py [--output-dir <dir>] [--no-view] <pr-url>
"""

import asyncio
import json
import re
import subprocess
import sys
import uuid
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from agents.diagram_builder import build_diagrams
from agents.pr_analyzer.call_graph_builder import build_pr_call_graph
from agents.pr_analyzer.full_graph_builder import build_full_graph

AGENT_DIR = Path(__file__).resolve().parent


async def analyze_pr(
    pr_url: str,
    output_dir: Path | None = None,
    analysis_id: str | None = None,
) -> Path | None:
    """Analyze a PR and build a pruned call graph with change status.

    Returns the result directory path, or None on failure.
    """
    result_id = analysis_id or uuid.uuid4().hex[:12]
    base_dir = output_dir or AGENT_DIR
    result_dir = base_dir / f"result_{result_id}"
    result_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Parse PR URL and clone repo
    print(f"Analyzing PR: {pr_url}")
    try:
        owner, repo, pr_number = _parse_pr_url(pr_url)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return None

    print("Cloning repository and setting up branches ...")
    setup = _clone_and_setup(owner, repo, pr_number, result_dir)
    if setup is None:
        return None
    base_path, head_path = setup

    # Step 2: Build PR call graph (tree-sitter compares base vs head)
    print("Building PR call graph ...")
    (
        call_graph,
        start_points,
        end_points,
        orphans,
        classification,
    ) = build_pr_call_graph(base_path, head_path)

    # Step 3: Enrich call graph with full tree builder
    print("Enriching call graph with full tree builder ...")
    call_graph = await build_full_graph(
        call_graph, str(head_path), output_dir=result_dir
    )

    # Step 4: Save results (strip source code from nodes to keep output clean)
    for node in call_graph["nodes"]:
        node.pop("source", None)
        node.pop("base_source", None)
    call_graph_file = result_dir / "call_graph.json"
    call_graph_file.write_text(json.dumps(call_graph, indent=2))
    print(f"  Call graph:    {call_graph_file} ({len(call_graph['nodes'])} functions, {len(call_graph['edges'])} edges)")

    start_points_file = result_dir / "start_points.json"
    start_points_file.write_text(json.dumps(start_points, indent=2))
    print(f"  Start points:  {start_points_file} ({len(start_points)} functions)")

    end_points_file = result_dir / "end_points.json"
    end_points_file.write_text(json.dumps(end_points, indent=2))
    print(f"  End points:    {end_points_file} ({len(end_points)} functions)")

    orphans_file = result_dir / "orphans.json"
    orphans_file.write_text(json.dumps(orphans, indent=2))
    print(f"  Orphans:       {orphans_file} ({len(orphans)} functions)")

    classification_file = result_dir / "classification.json"
    classification_file.write_text(
        json.dumps(
            {
                "new": sorted(classification.new),
                "updated": sorted(classification.updated),
                "deleted": sorted(classification.deleted),
            },
            indent=2,
        )
    )
    print(f"  Classification: {classification_file}")

    # Step 5: Generate mermaid diagrams
    print("Generating mermaid diagrams ...")
    diagrams = build_diagrams(call_graph, start_points)

    main_diagram_file = result_dir / "diagram_main.mmd"
    main_diagram_file.write_text(diagrams["main"])
    print(f"  Main diagram:  {main_diagram_file}")

    for sub_name, sub_content in diagrams.get("sub_diagrams", {}).items():
        sub_file = result_dir / f"diagram_{sub_name}.mmd"
        sub_file.write_text(sub_content)
        print(f"  Sub-diagram:   {sub_file}")

    return result_dir


def _parse_pr_url(pr_url: str) -> tuple[str, str, int]:
    """Extract (owner, repo, pr_number) from a GitHub PR URL.

    Supports:
      - https://github.com/owner/repo/pull/123
      - http://github.com/owner/repo/pull/123
    """
    match = re.match(
        r"https?://github\.com/([^/]+)/([^/]+)/pull/(\d+)", pr_url
    )
    if not match:
        raise ValueError(
            f"Cannot parse PR URL: {pr_url}\n"
            "Expected format: https://github.com/owner/repo/pull/123"
        )
    return match.group(1), match.group(2), int(match.group(3))


def _find_base_commit(
    owner: str,
    repo: str,
    pr_number: int,
    repo_dir: Path,
) -> str | None:
    """Find the correct base commit for comparing a PR.

    For merged PRs, uses the first parent of the merge commit (the state of
    the target branch just before the merge). For open PRs, uses the standard
    merge-base between PR head and the default branch.
    """
    # Check if PR is merged and get the merge commit SHA
    merge_commit_sha = _git_output(
        ["gh", "api", f"repos/{owner}/{repo}/pulls/{pr_number}",
         "--jq", 'if .merged then .merge_commit_sha else "" end'],
    )

    if merge_commit_sha:
        # Merged PR: first parent of merge commit = base before merge
        base = _git_output(
            ["git", "rev-parse", f"{merge_commit_sha}^1"], cwd=repo_dir
        )
        if base:
            return base

    # Open PR (or fallback): standard merge-base
    return _git_output(
        ["git", "merge-base", "pr_head", "origin/HEAD"], cwd=repo_dir
    )


def _clone_and_setup(
    owner: str,
    repo: str,
    pr_number: int,
    result_dir: Path,
) -> tuple[Path, Path] | None:
    """Clone repo and set up base/head branches using pure git.

    Strategy:
      1. Clone the repo
      2. Fetch PR head ref
      3. Find fork point via git merge-base
      4. Checkout PR head in main clone
      5. Create worktree at the fork point for the base

    Returns (base_path, head_path) or None on failure.
    """
    repo_dir = result_dir / "repo"
    base_dir = result_dir / "base"
    repo_url = f"https://github.com/{owner}/{repo}"

    # Clone the repository
    print(f"  Cloning {repo_url} ...")
    if not _run_git(["git", "clone", repo_url, str(repo_dir)], timeout=300):
        return None

    # Fetch the PR head ref
    print(f"  Fetching PR #{pr_number} head ...")
    if not _run_git(
        ["git", "fetch", "origin", f"pull/{pr_number}/head:pr_head"],
        cwd=repo_dir,
    ):
        return None

    # Find the base commit to compare against.
    # For merged PRs, merge-base(pr_head, origin/HEAD) == pr_head itself
    # (since pr_head is now an ancestor of main), so we use the first parent
    # of the merge commit instead — the state of main just before the merge.
    merge_base = _find_base_commit(owner, repo, pr_number, repo_dir)
    if merge_base is None:
        print("Error: could not determine merge base.", file=sys.stderr)
        return None
    print(f"  Merge base: {merge_base[:12]}")

    # Checkout PR head in the main clone
    if not _run_git(["git", "checkout", "pr_head"], cwd=repo_dir):
        return None

    # Create worktree for the base commit
    print("  Setting up base worktree ...")
    if not _run_git(
        ["git", "worktree", "add", str(base_dir), merge_base],
        cwd=repo_dir,
    ):
        return None

    return base_dir, repo_dir


def _run_git(
    cmd: list[str],
    cwd: Path | None = None,
    timeout: int = 120,
) -> bool:
    """Run a git command. Returns True on success."""
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd,
        )
    except FileNotFoundError:
        print("Error: 'git' not found.", file=sys.stderr)
        return False
    except subprocess.TimeoutExpired:
        print(f"Error: command timed out: {' '.join(cmd)}", file=sys.stderr)
        return False

    if result.returncode != 0:
        print(
            f"Error: command failed: {' '.join(cmd)}\n{result.stderr}",
            file=sys.stderr,
        )
        return False

    return True


def _git_output(
    cmd: list[str],
    cwd: Path | None = None,
    timeout: int = 30,
) -> str | None:
    """Run a git command and return its stdout, or None on failure."""
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None

    if result.returncode != 0:
        return None

    return result.stdout.strip()


def launch_viewer(result_dir: Path) -> None:
    """Start the viewer server and open the browser."""
    import webbrowser

    from viewer.data_loader import load_result
    from viewer.server import start_server

    data = load_result(str(result_dir))
    server = start_server(data, port=0)
    host, port = server.server_address
    url = f"http://{host}:{port}"
    print(f"Viewer running at {url}")
    webbrowser.open(url)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down viewer.")
        server.shutdown()


async def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]

    if len(args) < 1:
        print(
            f"Usage: {sys.argv[0]} [--output-dir <dir>] [--no-view] <pr-url>",
            file=sys.stderr,
        )
        sys.exit(1)

    output_dir = None
    if "--output-dir" in sys.argv:
        idx = sys.argv.index("--output-dir")
        if idx + 1 < len(sys.argv):
            output_dir = Path(sys.argv[idx + 1])

    no_view = "--no-view" in sys.argv

    pr_url = args[0]
    result = await analyze_pr(pr_url, output_dir=output_dir)
    if result is None:
        sys.exit(1)

    print(f"\nResult saved to: {result}")

    if not no_view:
        launch_viewer(result)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
