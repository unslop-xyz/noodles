import asyncio
import json
import subprocess
import sys
import tempfile
import uuid
import webbrowser
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from noodles.agents.diagram_builder import build_diagrams
from noodles.agents.repo_analyzer.call_graph_builder import build_call_graph
from noodles.agents.repo_analyzer.full_graph_builder import build_full_graph

AGENT_DIR = Path(__file__).parent
AGENTS_DIR = AGENT_DIR.parent
TEST_DIR = AGENT_DIR.parents[1] / "test"


async def analyze_repo(
    repo_url: str,
    output_dir: Path | None = None,
    analysis_id: str | None = None,
    enrich: bool = True,
) -> Path | None:
    """Clone a repo and build its function call graph.

    Args:
        repo_url: GitHub repository URL to clone and analyze
        output_dir: Directory for results (default: agent directory)
        analysis_id: Custom ID for the result directory
        enrich: If True, run LLM enrichment (slower, costs API calls).
                If False, only build the AST-based call graph (fast).

    Produces in the result directory:
      repo/              - the cloned repository
      call_graph.json    - all functions with type and connections
      start_points.json  - functions with out-degree only (not called by anyone)
      end_points.json    - functions with in-degree only (call nobody)
      orphans.json       - functions with no connections at all

    Returns the result directory path, or None on failure.
    """
    result_id = analysis_id or uuid.uuid4().hex[:12]

    # Use temp directory to avoid path issues (e.g., site-packages on Windows)
    temp_base = Path(tempfile.gettempdir()) / "noodles_repo"
    temp_base.mkdir(parents=True, exist_ok=True)
    temp_dir = temp_base / f"result_{result_id}"
    temp_dir.mkdir(parents=True, exist_ok=True)

    # Results go to output_dir if specified, otherwise temp
    result_dir = output_dir or temp_dir
    if output_dir:
        result_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Clone the repo (always to temp)
    repo_dir = temp_dir / "repo"
    print(f"Cloning {repo_url} ...")
    if not _clone_repo(repo_url, repo_dir):
        return None

    # Use shared implementation
    return await _analyze_repo_impl(repo_dir, result_dir, enrich=enrich)


async def analyze_local_repo(
    repo_path: Path,
    output_dir: Path,
    enrich: bool = True,
) -> Path:
    """Analyze a local repository (no cloning needed).

    Args:
        repo_path: Path to the local repository
        output_dir: Directory to write results to
        enrich: If True, run LLM enrichment (slower, costs API calls).
                If False, only build the AST-based call graph (fast).

    Produces in output_dir:
      call_graph.json    - all functions with type and connections
      start_points.json  - functions with out-degree only (not called by anyone)
      end_points.json    - functions with in-degree only (call nobody)
      orphans.json       - functions with no connections at all

    Returns the output directory path.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    return await _analyze_repo_impl(repo_path, output_dir, enrich=enrich)


async def _analyze_repo_impl(
    repo_path: Path,
    result_dir: Path,
    enrich: bool = True,
) -> Path:
    """Shared implementation for repo analysis."""
    # Step 1: Build the AST call graph
    print("Building call graph ...")
    call_graph, start_points, end_points, orphans = build_call_graph(repo_path)

    # Step 2: Optionally enrich call graph with full tree builder
    if enrich:
        print("Enriching call graph with full tree builder ...")
        call_graph = await build_full_graph(
            call_graph, str(repo_path), output_dir=result_dir
        )

    # Step 3: Save results (strip source code from nodes to keep output clean)
    for node in call_graph["nodes"]:
        node.pop("source", None)
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

    # Step 4: Generate mermaid diagrams
    print("Generating mermaid diagrams ...")
    diagrams = build_diagrams(call_graph, start_points)

    main_diagram_file = result_dir / "diagram_main.mmd"
    main_diagram_file.write_text(diagrams["main"])
    print(f"  Main diagram:  {main_diagram_file}")

    for sub_name, sub_content in diagrams.get("sub_diagrams", {}).items():
        sub_file = result_dir / f"diagram_{sub_name}.mmd"
        sub_file.write_text(sub_content)
        print(f"  Sub-diagram:   {sub_file}")

    # Step 5: Generate viewer bundle and HTML
    from noodles.viewer.data_loader import write_viewer_files
    write_viewer_files(result_dir)

    return result_dir


def _clone_repo(repo_url: str, dest: Path) -> bool:
    """Clone a git repository to dest. Returns True on success."""
    try:
        result = subprocess.run(
            ["gh", "repo", "clone", repo_url, str(dest)],
            capture_output=True,
            text=True,
            timeout=300,
        )
    except FileNotFoundError:
        print("Error: 'gh' not found. Install GitHub CLI: https://cli.github.com/", file=sys.stderr)
        return False
    except subprocess.TimeoutExpired:
        print("Error: git clone timed out.", file=sys.stderr)
        return False

    if result.returncode != 0:
        print(f"Error: gh repo clone failed:\n{result.stderr}", file=sys.stderr)
        return False

    return True


def launch_viewer(result_dir: Path) -> None:
    """Start the viewer server and open the browser."""
    from noodles.viewer.data_loader import load_result
    from noodles.viewer.server import start_server

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
            f"Usage: {sys.argv[0]} [--output-dir <dir>] [--no-view] <repo-url>",
            file=sys.stderr,
        )
        sys.exit(1)

    output_dir = None
    if "--output-dir" in sys.argv:
        idx = sys.argv.index("--output-dir")
        if idx + 1 < len(sys.argv):
            output_dir = Path(sys.argv[idx + 1])

    no_view = "--no-view" in sys.argv

    repo_url = args[0]
    result = await analyze_repo(repo_url, output_dir=output_dir)
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
