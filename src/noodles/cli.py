"""Noodles CLI - AI-powered code visualization and analysis."""

import argparse
import asyncio
import sys


def main():
    """Main entry point for the noodles CLI."""
    parser = argparse.ArgumentParser(
        prog="noodles",
        description="AI-powered code visualization and call graph analysis",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # noodles repo
    repo_parser = subparsers.add_parser(
        "repo",
        help="Analyze a repository",
        description="Clone and analyze a repository, building a function call graph",
    )
    repo_parser.add_argument("repo_url", help="Git repository URL to analyze")
    repo_parser.add_argument(
        "--output-dir", type=str, help="Output directory for results"
    )
    repo_parser.add_argument(
        "--no-view", action="store_true", help="Don't open the viewer after analysis"
    )

    # noodles pr
    pr_parser = subparsers.add_parser(
        "pr",
        help="Analyze a GitHub PR",
        description="Analyze a GitHub pull request, showing what changed",
    )
    pr_parser.add_argument("pr_url", help="GitHub PR URL (e.g., https://github.com/owner/repo/pull/123)")
    pr_parser.add_argument(
        "--output-dir", type=str, help="Output directory for results"
    )
    pr_parser.add_argument(
        "--no-view", action="store_true", help="Don't open the viewer after analysis"
    )

    # noodles viewer
    viewer_parser = subparsers.add_parser(
        "viewer",
        help="Open the interactive diagram viewer",
        description="Start the interactive viewer for a previous analysis result",
    )
    viewer_parser.add_argument("result_dir", help="Path to the result folder")
    viewer_parser.add_argument(
        "--port", type=int, default=0, help="Port (0=auto)"
    )

    args = parser.parse_args()

    if args.command == "repo":
        _run_repo(args)
    elif args.command == "pr":
        _run_pr(args)
    elif args.command == "viewer":
        _run_viewer(args)


def _run_repo(args):
    """Run the repository analyzer."""
    from pathlib import Path

    from noodles.agents.repo_analyzer.repo_analyzer import analyze_repo, launch_viewer

    output_dir = Path(args.output_dir) if args.output_dir else None
    result = asyncio.run(analyze_repo(args.repo_url, output_dir=output_dir))

    if result is None:
        sys.exit(1)

    print(f"\nResult saved to: {result}")

    if not args.no_view:
        launch_viewer(result)


def _run_pr(args):
    """Run the PR analyzer."""
    from pathlib import Path

    from noodles.agents.pr_analyzer.pr_analyzer import analyze_pr, launch_viewer

    output_dir = Path(args.output_dir) if args.output_dir else None
    result = asyncio.run(analyze_pr(args.pr_url, output_dir=output_dir))

    if result is None:
        sys.exit(1)

    print(f"\nResult saved to: {result}")

    if not args.no_view:
        launch_viewer(result)


def _run_viewer(args):
    """Run the interactive viewer."""
    import webbrowser

    from noodles.viewer.data_loader import load_result
    from noodles.viewer.server import start_server

    try:
        data = load_result(args.result_dir)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    server = start_server(data, port=args.port)
    host, port = server.server_address
    url = f"http://{host}:{port}"
    print(f"Viewer running at {url}")
    webbrowser.open(url)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.shutdown()


if __name__ == "__main__":
    main()
