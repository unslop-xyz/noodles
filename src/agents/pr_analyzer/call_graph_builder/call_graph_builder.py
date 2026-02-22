"""PR-aware call graph builder.

Builds a call graph from the PR HEAD branch, classifies functions as
new/updated/unchanged by comparing with the base branch, and prunes
the graph to only include nodes transitively connected to changed functions.
"""

import importlib.util
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path

# Import tree-sitter helpers from repo_analyzer's call_graph_builder.
# Use importlib to avoid circular import (both modules share the same
# fully-qualified name "call_graph_builder.call_graph_builder").
_REPO_CG_PATH = (
    Path(__file__).resolve().parents[2]
    / "repo_analyzer" / "call_graph_builder" / "call_graph_builder.py"
)
_spec = importlib.util.spec_from_file_location("_repo_call_graph_builder", _REPO_CG_PATH)
_repo_cg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_repo_cg)

EXTENSION_TO_LANG = _repo_cg.EXTENSION_TO_LANG
_find_functions_in_file = _repo_cg._find_functions_in_file
_get_parser = _repo_cg._get_parser
_iter_source_files = _repo_cg._iter_source_files
build_call_graph = _repo_cg.build_call_graph


@dataclass
class FunctionClassification:
    """Classification of functions between base and head branches."""

    new: set[str] = field(default_factory=set)
    updated: set[str] = field(default_factory=set)
    unchanged: set[str] = field(default_factory=set)
    deleted: set[str] = field(default_factory=set)


def build_pr_call_graph(
    base_path: Path,
    head_path: Path,
    changed_files: list[str] | None = None,
) -> tuple[dict, list[str], list[str], list[str], FunctionClassification]:
    """Build a pruned call graph focused on PR-changed functions.

    Args:
        base_path: Path to repo checked out at the base branch.
        head_path: Path to repo checked out at the PR HEAD.
        changed_files: List of file paths changed in the PR (optimization hint).

    Returns:
        (call_graph, start_points, end_points, orphans, classification)
        call_graph has nodes with "status" field: "new", "updated", "unchanged".
    """
    # Step 1: Classify functions by comparing base and head
    print("  Classifying functions ...")
    classification, base_funcs = _classify_functions(base_path, head_path, changed_files)
    print(
        f"    new={len(classification.new)}, "
        f"updated={len(classification.updated)}, "
        f"deleted={len(classification.deleted)}"
    )

    # Step 2: Build full call graph on HEAD branch
    print("  Building full call graph on HEAD ...")
    full_graph, _, _, _ = build_call_graph(head_path)
    print(
        f"    {len(full_graph['nodes'])} functions, "
        f"{len(full_graph['edges'])} edges"
    )

    # Step 3: Prune to connected subgraph and annotate status
    print("  Pruning to PR-relevant subgraph ...")
    pruned_graph, start_points, end_points, orphans = _prune_graph(
        full_graph, classification
    )
    print(
        f"    {len(pruned_graph['nodes'])} functions, "
        f"{len(pruned_graph['edges'])} edges retained"
    )

    # Attach base source to updated nodes for change description
    for node in pruned_graph["nodes"]:
        if node["status"] == "updated":
            node["base_source"] = base_funcs.get(node["id"], "")

    return pruned_graph, start_points, end_points, orphans, classification


def _classify_functions(
    base_path: Path,
    head_path: Path,
    changed_files: list[str] | None = None,
) -> tuple[FunctionClassification, dict[str, str]]:
    """Compare function sets between base and head to classify changes.

    Parses both branches with tree-sitter and compares function IDs and source.

    Returns:
        (classification, base_funcs) where base_funcs maps func_id -> source text.
    """
    # Build function map for HEAD (all files - needed for call graph)
    head_funcs = _build_function_map(head_path)

    # Build function map for BASE (only changed files for optimization)
    file_filter = set(changed_files) if changed_files else None
    base_funcs = _build_function_map(base_path, file_filter=file_filter)

    classification = FunctionClassification()

    head_ids = set(head_funcs.keys())
    base_ids = set(base_funcs.keys())

    classification.new = head_ids - base_ids
    classification.deleted = base_ids - head_ids

    for func_id in head_ids & base_ids:
        if head_funcs[func_id] != base_funcs[func_id]:
            classification.updated.add(func_id)
        else:
            classification.unchanged.add(func_id)

    return classification, base_funcs


def _build_function_map(
    repo_path: Path,
    file_filter: set[str] | None = None,
) -> dict[str, str]:
    """Build mapping of function_id -> source_text using tree-sitter.

    Args:
        repo_path: Root of the repository to scan.
        file_filter: If provided, only parse files whose relative path is in
                     this set. If None, parse all supported files.

    Returns:
        Dict mapping "rel/path.py::QualifiedName" -> source text.
    """
    func_map: dict[str, str] = {}

    for file_path in _iter_source_files(repo_path):
        rel_path = str(file_path.relative_to(repo_path))

        if file_filter is not None and rel_path not in file_filter:
            continue

        lang = EXTENSION_TO_LANG[file_path.suffix]
        try:
            source = file_path.read_bytes()
        except (OSError, PermissionError):
            continue

        try:
            parser = _get_parser(lang)
        except ValueError:
            continue

        tree = parser.parse(source)
        funcs = _find_functions_in_file(tree.root_node, lang)

        for qualified_name, func_node in funcs:
            func_id = f"{rel_path}::{qualified_name}"
            if func_id not in func_map:
                func_map[func_id] = func_node.text.decode("utf-8")

    return func_map


def _prune_graph(
    call_graph: dict,
    classification: FunctionClassification,
) -> tuple[dict, list[str], list[str], list[str]]:
    """Prune call graph to subgraph connected to new/updated functions.

    Performs bidirectional BFS from seed nodes (new + updated), then filters
    to retained nodes and reclassifies node types.
    """
    nodes = call_graph["nodes"]
    edges = call_graph["edges"]

    all_node_ids = {n["id"] for n in nodes}

    # Build adjacency maps
    forward_adj: dict[str, list[str]] = defaultdict(list)
    reverse_adj: dict[str, list[str]] = defaultdict(list)
    for edge in edges:
        forward_adj[edge["from"]].append(edge["to"])
        reverse_adj[edge["to"]].append(edge["from"])

    # Seed = new + updated functions that exist in the graph
    seeds = (classification.new | classification.updated) & all_node_ids

    if not seeds:
        # No changed functions found in call graph — return empty graph
        return {"nodes": [], "edges": []}, [], [], []

    # BFS forward (callees) and backward (callers)
    forward_reached = _bfs_reach(seeds, forward_adj, all_node_ids)
    backward_reached = _bfs_reach(seeds, reverse_adj, all_node_ids)

    # Only keep edges on paths through changed nodes.
    # An edge A→B is on such a path if both endpoints are in the same
    # directional set: both upstream of a seed (backward) or both
    # downstream (forward). This drops cross-edges like W→Z where W is
    # only upstream and Z is only downstream.
    retained_edges = [
        e for e in edges
        if (e["from"] in backward_reached and e["to"] in backward_reached)
        or (e["from"] in forward_reached and e["to"] in forward_reached)
    ]

    # Retain nodes that participate in at least one path-relevant edge,
    # plus seeds themselves (even if isolated).
    retained: set[str] = set(seeds)
    for e in retained_edges:
        retained.add(e["from"])
        retained.add(e["to"])

    # Filter nodes
    retained_nodes = []
    node_by_id: dict[str, dict] = {}
    for node in nodes:
        if node["id"] not in retained:
            continue

        # Determine status
        nid = node["id"]
        if nid in classification.new:
            status = "new"
        elif nid in classification.updated:
            status = "updated"
        else:
            status = "unchanged"

        # Filter callers/callees to retained set
        pruned_node = {
            "id": nid,
            "type": node["type"],  # will be reclassified below
            "status": status,
            "callers": [c for c in node.get("callers", []) if c in retained],
            "callees": [c for c in node.get("callees", []) if c in retained],
            "source": node.get("source", ""),
        }
        retained_nodes.append(pruned_node)
        node_by_id[nid] = pruned_node

    # Reclassify node types based on pruned connectivity
    start_points, end_points, orphans = _reclassify_nodes(
        retained_nodes, retained_edges
    )

    return (
        {"nodes": retained_nodes, "edges": retained_edges},
        start_points,
        end_points,
        orphans,
    )


def _bfs_reach(
    seeds: set[str],
    adjacency: dict[str, list[str]],
    scope: set[str],
) -> set[str]:
    """BFS from seed nodes through adjacency. Returns all reachable node IDs."""
    visited: set[str] = set()
    queue = deque(seeds)
    visited.update(seeds)

    while queue:
        node = queue.popleft()
        for neighbor in adjacency.get(node, []):
            if neighbor in scope and neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)

    return visited


def _reclassify_nodes(
    nodes: list[dict],
    edges: list[dict],
) -> tuple[list[str], list[str], list[str]]:
    """Reclassify node types based on pruned graph connectivity.

    Returns (start_points, end_points, orphans).
    """
    # Build caller/callee sets from edges
    has_outgoing: set[str] = set()
    has_incoming: set[str] = set()
    for edge in edges:
        has_outgoing.add(edge["from"])
        has_incoming.add(edge["to"])

    start_points: list[str] = []
    end_points: list[str] = []
    orphans: list[str] = []

    for node in nodes:
        nid = node["id"]
        out = nid in has_outgoing
        inc = nid in has_incoming

        if out and inc:
            node["type"] = "process"
        elif out:
            node["type"] = "start_point"
            start_points.append(nid)
        elif inc:
            node["type"] = "end_point"
            end_points.append(nid)
        else:
            node["type"] = "orphan"
            orphans.append(nid)

    return start_points, end_points, orphans
