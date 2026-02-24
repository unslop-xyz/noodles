"""Build mermaid diagrams from an enriched call graph."""

import re
from collections import defaultdict

from noodles.utils import _sanitize_id


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_NODES_PER_DIAGRAM = 20
BRANCH_SIZE_THRESHOLD = 10

SHAPE_BY_TYPE = {
    "start_point": '{{{{"{}"}}}}'  ,  # hexagon
    "end_point":   '(["{}"])'     ,  # stadium
    "process":     '["{}"]'       ,  # rectangle
}

STYLE_PREFIX_BY_TYPE = {
    "start_point": "start",
    "end_point":   "endpoint",
    "process":     "process",
}

STYLE_DEFS = (
    '    classDef start fill:none,stroke:#4db6ac,color:#00695c\n'
    '    classDef process fill:none,stroke:#64b5f6,color:#1565c0\n'
    '    classDef endpoint fill:none,stroke:#e57373,color:#c62828\n'
    '    classDef updated fill:#fff9c4,stroke:#f9a825,color:#f57f17\n'
    '    classDef new fill:#c8e6c9,stroke:#66bb6a,color:#2e7d32'
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_diagrams(
    call_graph: dict,
    start_points: list[str],
) -> dict:
    """Build mermaid diagrams from an enriched call graph.

    Args:
        call_graph: Enriched call graph with nodes and edges.
        start_points: List of start point node IDs.

    Returns:
        {"main": str, "sub_diagrams": {"name": str, ...}}
    """
    # Filter out test and orphan nodes
    exclude_ids = {
        n["id"] for n in call_graph["nodes"]
        if n.get("tag") == "test" or n.get("type") == "orphan"
    }
    nodes = [n for n in call_graph["nodes"] if n["id"] not in exclude_ids]
    edges = [e for e in call_graph["edges"] if e["from"] not in exclude_ids and e["to"] not in exclude_ids]
    start_points = [sp for sp in start_points if sp not in exclude_ids]

    node_index = {n["id"]: n for n in nodes}
    edge_index: dict[str, list[dict]] = defaultdict(list)
    reverse_edge_index: dict[str, list[dict]] = defaultdict(list)
    for e in edges:
        edge_index[e["from"]].append(e)
        reverse_edge_index[e["to"]].append(e)

    all_node_ids = set(node_index.keys())

    # Check if this is a PR analysis (has changed nodes)
    changed_ids = [
        n["id"] for n in nodes
        if n.get("status") in ("new", "updated")
    ]

    if changed_ids:
        # PR analysis: center diagram on changed nodes, show context bidirectionally
        reachable = _collect_bidirectional(
            changed_ids, edge_index, reverse_edge_index, all_node_ids
        )
        # Find roots of the reachable subgraph (nodes with no incoming edges
        # from other reachable nodes) to ensure connecting paths are shown
        entry_ids = _find_subgraph_roots(reachable, reverse_edge_index)
    elif start_points:
        # Repo analysis: traverse from start points
        reachable = _collect_reachable(start_points, edge_index, all_node_ids)
        entry_ids = start_points
    else:
        return {"main": "graph LR\n", "sub_diagrams": {}}

    if not reachable:
        return {"main": "graph LR\n", "sub_diagrams": {}}

    # Step 1: If <= threshold, render flat
    if len(reachable) <= MAX_NODES_PER_DIAGRAM:
        main = _render_flat(reachable, edge_index, node_index)
        return {"main": main, "sub_diagrams": {}}

    # Step 2: Branch-size-based splitting
    main, sub_diagrams = _build_diagrams_branch_based(
        entry_ids, edge_index, node_index, reachable,
    )
    return {"main": main, "sub_diagrams": sub_diagrams}


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def _collect_reachable(
    entry_ids: list[str],
    edge_index: dict[str, list[dict]],
    scope: set[str],
) -> set[str]:
    """DFS from entry_ids within scope, return reachable node IDs."""
    reachable: set[str] = set()
    stack = list(entry_ids)
    while stack:
        nid = stack.pop()
        if nid in reachable or nid not in scope:
            continue
        reachable.add(nid)
        for e in edge_index.get(nid, []):
            if e["to"] in scope:
                stack.append(e["to"])
    return reachable


def _find_subgraph_roots(
    subgraph: set[str],
    reverse_edge_index: dict[str, list[dict]],
) -> list[str]:
    """Find root nodes of a subgraph (nodes with no incoming edges from within).

    These are the natural entry points for rendering the subgraph top-down.
    """
    roots = []
    for nid in subgraph:
        has_internal_caller = False
        for e in reverse_edge_index.get(nid, []):
            if e["from"] in subgraph:
                has_internal_caller = True
                break
        if not has_internal_caller:
            roots.append(nid)
    return sorted(roots)


def _collect_bidirectional(
    center_ids: list[str],
    edge_index: dict[str, list[dict]],
    reverse_edge_index: dict[str, list[dict]],
    scope: set[str],
) -> set[str]:
    """Collect nodes reachable in both directions from center_ids.

    - Downstream: follow edge_index (caller -> callees)
    - Upstream: follow reverse_edge_index (callee -> callers)
    """
    reachable: set[str] = set()
    stack = list(center_ids)

    # Collect downstream (callees)
    while stack:
        nid = stack.pop()
        if nid in reachable or nid not in scope:
            continue
        reachable.add(nid)
        for e in edge_index.get(nid, []):
            if e["to"] in scope:
                stack.append(e["to"])

    # Collect upstream (callers) - start fresh from center
    visited_up: set[str] = set()
    stack = list(center_ids)
    while stack:
        nid = stack.pop()
        if nid in visited_up or nid not in scope:
            continue
        visited_up.add(nid)
        reachable.add(nid)
        for e in reverse_edge_index.get(nid, []):
            if e["from"] in scope:
                stack.append(e["from"])

    return reachable


def _render_flat(
    node_ids: set[str],
    edge_index: dict[str, list[dict]],
    node_index: dict[str, dict],
    sub_roots: set[str] | None = None,
) -> str:
    """Render all nodes and edges between them as a single mermaid diagram."""
    roots = sub_roots or set()
    lines = ["graph LR", STYLE_DEFS]
    for nid in sorted(node_ids):
        node = node_index.get(nid)
        if node:
            lines.append(f"    {_mermaid_node(node, has_sub=nid in roots)}")
    for nid in sorted(node_ids & roots):
        lines.append(f"    style {_sanitize_id(nid)} stroke-width:6px")
    emitted: set[tuple[str, str]] = set()
    for nid in sorted(node_ids):
        for e in edge_index.get(nid, []):
            if e["to"] in node_ids:
                key = (e["from"], e["to"])
                if key not in emitted:
                    emitted.add(key)
                    lines.append(f"    {_mermaid_edge(e['from'], e['to'], e)}")
    return "\n".join(lines) + "\n"


def _compute_branch_sizes(
    reachable: set[str],
    edge_index: dict[str, list[dict]],
    threshold: int,
) -> tuple[dict[str, int], set[str]]:
    """Compute branch sizes bottom-up and identify sub-diagram roots.

    A node becomes a sub-diagram root when its total branch size
    (self + all unique descendants) exceeds *threshold*.

    Children are traversed in edge ``index`` order (call sequence from
    tree-sitter) so the DFS respects source-code execution order.
    Cycles are handled via WHITE/GRAY/BLACK coloring — a back-edge
    contributes 1 (the node itself, no further expansion).

    Returns:
        branch_size: node_id → total branch size (before collapsing)
        sub_roots:   set of node_ids that became sub-diagram roots
    """
    WHITE, GRAY, BLACK = 0, 1, 2
    color: dict[str, int] = {nid: WHITE for nid in reachable}
    branch_size: dict[str, int] = {}
    sub_roots: set[str] = set()

    def dfs(nid: str) -> int:
        """Return effective size visible to parent (1 if collapsed)."""
        if nid not in reachable:
            return 0
        if color[nid] == BLACK:
            return 1 if nid in sub_roots else branch_size[nid]
        if color[nid] == GRAY:
            return 1  # cycle — just the node, no expansion

        color[nid] = GRAY

        # Traverse children in index order
        edges = sorted(
            (e for e in edge_index.get(nid, []) if e["to"] in reachable),
            key=lambda e: e.get("index", 0),
        )

        seen: set[str] = set()
        total = 1  # count self
        for e in edges:
            child = e["to"]
            if child not in seen:
                seen.add(child)
                total += dfs(child)

        branch_size[nid] = total
        color[nid] = BLACK

        if total > threshold:
            sub_roots.add(nid)
            return 1  # collapsed for parent
        return total

    for nid in reachable:
        if color[nid] == WHITE:
            dfs(nid)

    return branch_size, sub_roots


def _collect_main_diagram_nodes(
    entry_ids: list[str],
    edge_index: dict[str, list[dict]],
    reachable: set[str],
    sub_roots: set[str],
) -> set[str]:
    """Collect nodes for the main (top-level) diagram.

    BFS from entry points. Sub-roots are included but never expanded
    — their details live in the sub-diagram.
    """
    nodes: set[str] = set()
    stack = list(entry_ids)
    while stack:
        nid = stack.pop()
        if nid in nodes or nid not in reachable:
            continue
        nodes.add(nid)
        if nid in sub_roots:
            continue  # include but don't expand
        for e in edge_index.get(nid, []):
            if e["to"] in reachable:
                stack.append(e["to"])
    return nodes


def _collect_sub_diagram_nodes(
    root: str,
    edge_index: dict[str, list[dict]],
    reachable: set[str],
    sub_roots: set[str],
) -> set[str]:
    """Collect nodes for a sub-diagram rooted at *root*.

    Includes root + all descendants, stopping at (but including)
    nested sub-roots.
    """
    nodes: set[str] = set()
    stack = [root]
    while stack:
        nid = stack.pop()
        if nid in nodes or nid not in reachable:
            continue
        nodes.add(nid)
        if nid != root and nid in sub_roots:
            continue  # include but don't expand
        for e in edge_index.get(nid, []):
            if e["to"] in reachable:
                stack.append(e["to"])
    return nodes


def _build_diagrams_branch_based(
    entry_ids: list[str],
    edge_index: dict[str, list[dict]],
    node_index: dict[str, dict],
    reachable: set[str],
    threshold: int = BRANCH_SIZE_THRESHOLD,
) -> tuple[str, dict[str, str]]:
    """Build main + sub-diagrams using branch-size-based splitting.

    1. Compute branch sizes from leaves upward.
    2. Mark nodes exceeding *threshold* as sub-diagram roots.
    3. Render main diagram (sub-roots collapsed).
    4. Render each sub-diagram (nested sub-roots collapsed).
    """
    # Phase 1: bottom-up branch sizing
    _sizes, sub_roots = _compute_branch_sizes(reachable, edge_index, threshold)

    # Phase 2: main diagram
    main_nodes = _collect_main_diagram_nodes(
        entry_ids, edge_index, reachable, sub_roots,
    )
    main = _render_flat(
        main_nodes, edge_index, node_index, sub_roots=sub_roots & main_nodes,
    )

    # Phase 3: sub-diagrams
    sub_diagrams: dict[str, str] = {}
    for root_id in sorted(sub_roots):
        sub_nodes = _collect_sub_diagram_nodes(
            root_id, edge_index, reachable, sub_roots,
        )
        name = _unique_sub_name(root_id, node_index, sub_diagrams)
        nested = (sub_roots & sub_nodes) - {root_id}
        sub_diagrams[name] = _render_flat(
            sub_nodes, edge_index, node_index, sub_roots=nested,
        )

    return main, sub_diagrams


# ---------------------------------------------------------------------------
# Naming
# ---------------------------------------------------------------------------

def _unique_sub_name(
    root_id: str,
    node_index: dict[str, dict],
    existing: dict[str, str],
) -> str:
    """Generate a unique filename-safe name for a sub-diagram."""
    node = node_index.get(root_id)
    if node and node.get("name"):
        base = _sanitize_id(node["name"])
    else:
        base = _sanitize_id(root_id.split("::")[-1])
    if base not in existing:
        return base
    i = 2
    while f"{base}_{i}" in existing:
        i += 1
    return f"{base}_{i}"


# ---------------------------------------------------------------------------
# Mermaid formatting
# ---------------------------------------------------------------------------

def _mermaid_node(node: dict, has_sub: bool = False) -> str:
    """Format a single node as a mermaid node definition line."""
    mid = _sanitize_id(node["id"])
    label = _sanitize_text(node.get("name") or node["id"].split("::")[-1])
    if has_sub:
        label += " [+]"
    shape = SHAPE_BY_TYPE.get(node.get("type", "process"), '["{}"]')
    status = node.get("status")
    if status == "updated":
        cls = "updated"
    elif status == "new":
        cls = "new"
    else:
        cls = STYLE_PREFIX_BY_TYPE.get(node.get("type", "process"), "process")
    return f"{mid}{shape.format(label)}:::{cls}"


def _mermaid_edge(from_id: str, to_id: str, edge: dict | None) -> str:
    """Format a single edge as a mermaid edge line."""
    sfrom = _sanitize_id(from_id)
    sto = _sanitize_id(to_id)

    if edge is None:
        return f"{sfrom} --> {sto}"

    # Build label
    label = _sanitize_text(edge.get("label", ""))

    # Arrow style: dashed for returned, solid otherwise
    arrow = "-.->" if edge.get("is_returned") else "-->"

    if label:
        return f'{sfrom} {arrow}|"{label}"| {sto}'
    return f"{sfrom} {arrow} {sto}"


# ---------------------------------------------------------------------------
# Sanitization
# ---------------------------------------------------------------------------

def _sanitize_text(text: str) -> str:
    """Sanitize text for use inside mermaid labels."""
    text = text.replace('"', "'")
    text = text.replace("\n", " ")
    text = text.replace("\r", "")
    text = text.replace("|", "/")
    text = text.replace("`", "'")
    return text
