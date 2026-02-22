"""Build enriched call graph: one node_builder agent per file, one edge_builder agent per edge."""

import asyncio
import sys
from collections import defaultdict
from pathlib import Path

# Ensure sub-agent directories are importable when this module is loaded
# from a parent directory (e.g., repo_analyzer.py).
sys.path.insert(0, str(Path(__file__).parent))

from node_builder.node_builder import run_node_builder_for_file
from edge_builder.edge_builder import run_edge_builder_for_caller

MAX_CONCURRENT_AGENTS = 20


def _print_progress(label: str, done: int, total: int) -> None:
    """Print an in-place progress bar to stderr."""
    width = 30
    filled = int(width * done / total) if total else width
    bar = "█" * filled + "░" * (width - filled)
    pct = (done / total * 100) if total else 100
    print(f"\r  {label}: |{bar}| {done}/{total} ({pct:.0f}%)", end="", flush=True)


def _file_from_id(node_id: str) -> str:
    """Extract file path from a node ID like 'src/foo.py::ClassName.method'."""
    return node_id.split("::")[0]


async def build_full_graph(
    call_graph: dict,
    repo_path: str,
    output_dir: Path,
) -> dict:
    """Enrich all nodes and edges in the call graph.

    - Nodes: one agent per source file (batched), all files concurrently.
    - Edges: one agent per edge, rate-limited by semaphore.

    Args:
        call_graph: The call graph dict from call_graph_builder (nodes and edges).
                    Each node must have a "source" field with the function source code.
        repo_path: Path to the repository root
        output_dir: Directory to write consolidated logs

    Returns:
        The enriched call_graph (mutated in place and returned).
    """
    nodes = call_graph["nodes"]
    edges = call_graph["edges"]
    node_index = {n["id"]: n for n in nodes}

    if not nodes and not edges:
        print("  Nothing to process.")
        return call_graph

    # -- Phase 1: Node builders (batched by file, max 10 concurrent) --------
    nodes_by_file: dict[str, list[dict]] = defaultdict(list)
    for node in nodes:
        if node.get("source"):
            nodes_by_file[_file_from_id(node["id"])].append(node)

    total_node_tasks = len(nodes_by_file)
    print(f"Enriching {len(nodes)} nodes across {total_node_tasks} files ...")

    node_semaphore = asyncio.Semaphore(MAX_CONCURRENT_AGENTS)
    node_done = {"count": 0}

    async def _run_node(fp, file_nodes):
        async with node_semaphore:
            result = await run_node_builder_for_file(file_path=fp, nodes=file_nodes)
        node_done["count"] += 1
        _print_progress("Nodes", node_done["count"], total_node_tasks)
        return result

    node_tasks = [
        _run_node(fp, file_nodes)
        for fp, file_nodes in nodes_by_file.items()
    ]
    node_results = await asyncio.gather(*node_tasks, return_exceptions=True)
    _write_consolidated_log(output_dir / "node_builder.log", node_results, "node_builder")

    n_ok = sum(1 for r in node_results if isinstance(r, dict) and r.get("success"))
    print(f"\n  Nodes done: {n_ok}/{total_node_tasks} files enriched.")

    # -- Phase 2: Edge builders (batched by caller, rate-limited) ------------
    # Skip edges where the caller node is tagged as "test".
    edges_by_caller: dict[str, list[dict]] = defaultdict(list)
    skipped_edges = 0
    for edge in edges:
        caller = node_index.get(edge["from"])
        if not caller or not caller.get("source"):
            continue
        if caller.get("tag") == "test":
            skipped_edges += 1
            continue
        edges_by_caller[edge["from"]].append(edge)

    if skipped_edges:
        print(f"  Skipped {skipped_edges} edges from test nodes.")

    edge_semaphore = asyncio.Semaphore(MAX_CONCURRENT_AGENTS)
    edge_done = {"count": 0}
    total_edge_tasks = len(edges_by_caller)

    edge_tasks = []
    total_edges = 0
    for caller_id, caller_edges in edges_by_caller.items():
        caller = node_index[caller_id]
        total_edges += len(caller_edges)

        async def _run_edge(cid=caller_id, csrc=caller.get("source", ""), ce=caller_edges):
            async with edge_semaphore:
                result = await run_edge_builder_for_caller(
                    caller_id=cid,
                    caller_source=csrc,
                    edges=ce,
                    node_index=node_index,
                )
            edge_done["count"] += 1
            _print_progress("Edges", edge_done["count"], total_edge_tasks)
            return result

        edge_tasks.append(_run_edge())

    print(f"Enriching {total_edges} edges across {total_edge_tasks} callers (max {MAX_CONCURRENT_AGENTS} concurrent) ...")
    edge_results = await asyncio.gather(*edge_tasks, return_exceptions=True)
    _write_consolidated_log(output_dir / "edge_builder.log", edge_results, "edge_builder")

    e_ok = sum(1 for r in edge_results if isinstance(r, dict) and r.get("success"))
    print(f"\n  Edges done: {e_ok}/{total_edge_tasks} callers enriched.")

    return call_graph


def _write_consolidated_log(
    log_path: Path,
    results: list,
    agent_type: str,
) -> None:
    """Write a consolidated log file with per-agent sections and summary stats."""
    lines: list[str] = []
    total_duration_ms = 0
    total_cost_usd = 0.0
    total_success = 0
    total_errors = 0

    for r in results:
        if isinstance(r, BaseException):
            lines.append("--- EXCEPTION ---")
            lines.append(str(r))
            lines.append("")
            total_errors += 1
            continue

        item_id = r.get("node_id") or r.get("edge_key", "unknown")
        lines.append(f"=== {item_id} ===")
        lines.extend(r.get("log_lines", []))

        if r.get("errors"):
            for err in r["errors"]:
                lines.append(f"  ERROR: {err}")

        stats = r.get("stats", {})
        duration = stats.get("duration_ms", 0)
        cost = stats.get("cost_usd", 0.0)
        turns = stats.get("num_turns", 0)
        lines.append(f"  [{turns} turns, {duration}ms, ${cost:.4f}]")
        lines.append("")

        total_duration_ms += duration
        total_cost_usd += cost
        if r.get("success"):
            total_success += 1
        else:
            total_errors += 1

    summary = [
        f"=== {agent_type} Summary ===",
        f"Total: {len(results)} | Success: {total_success} | Failed: {total_errors}",
        f"Duration: {total_duration_ms}ms | Cost: ${total_cost_usd:.4f}",
        "=" * 50,
        "",
    ]

    log_path.write_text("\n".join(summary + lines))
