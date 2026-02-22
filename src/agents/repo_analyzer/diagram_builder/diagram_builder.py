"""Build mermaid diagrams from an enriched call graph."""

import re
from collections import defaultdict


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_NODES_PER_DIAGRAM = 20

SHAPE_BY_TYPE = {
    "start_point": '{{{{"{}"}}}}'  ,  # hexagon
    "end_point":   '(["{}"])'     ,  # stadium
    "process":     '["{}"]'       ,  # rectangle
}

STYLE_PREFIX_BY_TYPE = {
    "start_point": "start",
    "end_point":   "end",
    "process":     "process",
}

STYLE_DEFS = (
    '    classDef startSolid fill:#e8f5e9,stroke:#81c784,color:#2e7d32\n'
    '    classDef startTransparent fill:none,stroke:#81c784,color:#2e7d32\n'
    '    classDef processSolid fill:#e3f2fd,stroke:#64b5f6,color:#1565c0\n'
    '    classDef processTransparent fill:none,stroke:#64b5f6,color:#1565c0\n'
    '    classDef endSolid fill:#fce4ec,stroke:#e57373,color:#c62828\n'
    '    classDef endTransparent fill:none,stroke:#e57373,color:#c62828'
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
    for e in edges:
        edge_index[e["from"]].append(e)

    # Collect all reachable nodes from start points
    all_node_ids = set(node_index.keys())
    reachable = _collect_reachable(start_points, edge_index, all_node_ids)

    if not reachable:
        return {"main": "graph LR\n", "sub_diagrams": {}}

    # Step 1: If <= threshold, render flat
    if len(reachable) <= MAX_NODES_PER_DIAGRAM:
        main = _render_flat(reachable, edge_index, node_index)
        return {"main": main, "sub_diagrams": {}}

    # Step 2: Split
    main, sub_diagrams = _split_diagram(start_points, edge_index, node_index, reachable)
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
    emitted: set[tuple[str, str]] = set()
    for nid in sorted(node_ids):
        for e in edge_index.get(nid, []):
            if e["to"] in node_ids:
                key = (e["from"], e["to"])
                if key not in emitted:
                    emitted.add(key)
                    lines.append(f"    {_mermaid_edge(e['from'], e['to'], e)}")
    return "\n".join(lines) + "\n"


def _split_diagram(
    entry_ids: list[str],
    edge_index: dict[str, list[dict]],
    node_index: dict[str, dict],
    reachable: set[str],
    inline_ids: set[str] | None = None,
) -> tuple[str, dict[str, str]]:
    """Split a diagram with > MAX_NODES nodes into main + sub-diagrams.

    Rules:
      - Non-returned edges stay in the main diagram.
      - Returned edges open a sub-diagram rooted at the caller node,
        containing all downstream content.
      - Each sub-diagram is recursively split if it exceeds the threshold.

    Args:
        inline_ids: Nodes whose edges are always shown directly (no splitting).
                    Used to prevent infinite recursion when a sub-diagram root
                    is re-entered in a recursive split.
    """
    inline = inline_ids or set()
    main_nodes: set[str] = set()
    main_edges: list[dict] = []
    sub_diagrams: dict[str, str] = {}
    sub_roots: set[str] = set()
    visited: set[str] = set()

    def visit(nid: str) -> None:
        if nid in visited or nid not in reachable:
            return
        visited.add(nid)
        main_nodes.add(nid)

        outgoing = [e for e in edge_index.get(nid, []) if e["to"] in reachable]

        # Inline nodes: show all edges directly (prevent infinite recursion)
        if nid in inline:
            for e in outgoing:
                main_edges.append(e)
                visit(e["to"])
            return

        non_ret = [e for e in outgoing if not e.get("is_returned")]
        ret = [e for e in outgoing if e.get("is_returned")]

        # Non-returned edges go in main
        for e in non_ret:
            main_edges.append(e)
            visit(e["to"])

        if not ret:
            return

        # Returned edges open a sub-diagram
        # Collect all downstream nodes from returned targets
        # (ignore visited — nodes may appear in multiple diagrams)
        sub_scope: set[str] = {nid}
        stack = [e["to"] for e in ret]
        while stack:
            sid = stack.pop()
            if sid in sub_scope or sid not in reachable:
                continue
            sub_scope.add(sid)
            for se in edge_index.get(sid, []):
                if se["to"] in reachable:
                    stack.append(se["to"])

        if len(sub_scope) <= 1:
            # All returned targets already visited — just draw edges in main
            for e in ret:
                main_edges.append(e)
            return

        # Mark sub-scope nodes as visited in main (except root)
        visited.update(sub_scope - {nid})

        # Build sub-diagram (recursively split if > threshold)
        sub_roots.add(nid)
        name = _unique_sub_name(nid, node_index, sub_diagrams)
        if len(sub_scope) <= MAX_NODES_PER_DIAGRAM:
            sub_diagrams[name] = _render_flat(sub_scope, edge_index, node_index)
        else:
            sub_main, nested = _split_diagram(
                [nid], edge_index, node_index, sub_scope, inline_ids={nid},
            )
            sub_diagrams[name] = sub_main
            sub_diagrams.update(nested)

    for eid in entry_ids:
        visit(eid)

    # Render main diagram
    lines = ["graph LR", STYLE_DEFS]
    emitted_nodes: set[str] = set()
    for nid in sorted(main_nodes):
        node = node_index.get(nid)
        if node:
            emitted_nodes.add(nid)
            lines.append(f"    {_mermaid_node(node, has_sub=nid in sub_roots)}")

    emitted_edges: set[tuple[str, str]] = set()
    for e in main_edges:
        # Ensure target node is emitted
        target = e["to"]
        if target not in emitted_nodes:
            node = node_index.get(target)
            if node:
                emitted_nodes.add(target)
                lines.append(f"    {_mermaid_node(node, has_sub=target in sub_roots)}")
        key = (e["from"], e["to"])
        if key not in emitted_edges:
            emitted_edges.add(key)
            lines.append(f"    {_mermaid_edge(e['from'], e['to'], e)}")

    main = "\n".join(lines) + "\n"
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
    shape = SHAPE_BY_TYPE.get(node.get("type", "process"), '["{}"]')
    prefix = STYLE_PREFIX_BY_TYPE.get(node.get("type", "process"), "process")
    suffix = "Solid" if has_sub else "Transparent"
    style = f":::{prefix}{suffix}"
    return f"{mid}{shape.format(label)}{style}"


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

def _sanitize_id(node_id: str) -> str:
    """Convert a call graph node ID to a valid mermaid identifier."""
    s = re.sub(r"[^a-zA-Z0-9_]", "_", node_id)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "node"


def _sanitize_text(text: str) -> str:
    """Sanitize text for use inside mermaid labels."""
    text = text.replace('"', "'")
    text = text.replace("\n", " ")
    text = text.replace("\r", "")
    text = text.replace("|", "/")
    text = text.replace("`", "'")
    return text
