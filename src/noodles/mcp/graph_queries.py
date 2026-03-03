"""Graph query utilities for MCP server.

Provides functions for traversing, querying, and filtering call graphs.
"""

from collections import deque


def build_adjacency(call_graph: dict) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """Build forward and reverse adjacency maps from edges.

    Args:
        call_graph: Call graph with "nodes" and "edges" lists.

    Returns:
        (forward_adj, reverse_adj) where:
        - forward_adj[caller] = [callees...]
        - reverse_adj[callee] = [callers...]
    """
    forward: dict[str, list[str]] = {}
    reverse: dict[str, list[str]] = {}

    # Initialize with all nodes
    for node in call_graph.get("nodes", []):
        nid = node["id"]
        forward[nid] = []
        reverse[nid] = []

    # Build from edges
    for edge in call_graph.get("edges", []):
        from_id = edge["from"]
        to_id = edge["to"]
        if from_id in forward:
            forward[from_id].append(to_id)
        if to_id in reverse:
            reverse[to_id].append(from_id)

    return forward, reverse


def bfs_reach(
    seeds: set[str],
    adjacency: dict[str, list[str]],
    max_depth: int = -1,
) -> dict[str, int]:
    """BFS from seeds, return {node_id: depth}.

    Args:
        seeds: Starting node IDs.
        adjacency: Adjacency map (node -> neighbors).
        max_depth: Maximum depth to explore. -1 means unlimited.

    Returns:
        Dict mapping reachable node_id to its depth from nearest seed.
    """
    result: dict[str, int] = {}
    queue: deque[tuple[str, int]] = deque()

    for seed in seeds:
        if seed in adjacency:
            queue.append((seed, 0))
            result[seed] = 0

    while queue:
        node, depth = queue.popleft()

        if max_depth >= 0 and depth >= max_depth:
            continue

        for neighbor in adjacency.get(node, []):
            if neighbor not in result:
                result[neighbor] = depth + 1
                queue.append((neighbor, depth + 1))

    return result


def find_path(
    call_graph: dict,
    from_id: str,
    to_id: str,
) -> list[str] | None:
    """Find shortest path between two functions.

    Args:
        call_graph: Call graph with "nodes" and "edges".
        from_id: Starting function ID.
        to_id: Target function ID.

    Returns:
        List of node IDs forming path, or None if no path exists.
    """
    forward, _ = build_adjacency(call_graph)

    if from_id not in forward or to_id not in forward:
        return None

    if from_id == to_id:
        return [from_id]

    # BFS to find shortest path
    queue: deque[tuple[str, list[str]]] = deque()
    queue.append((from_id, [from_id]))
    visited: set[str] = {from_id}

    while queue:
        node, path = queue.popleft()

        for neighbor in forward.get(node, []):
            if neighbor == to_id:
                return path + [neighbor]

            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))

    return None


def get_callers(
    call_graph: dict,
    func_id: str,
    depth: int = 1,
) -> list[dict]:
    """Get transitive callers up to depth.

    Args:
        call_graph: Call graph with "nodes" and "edges".
        func_id: Function ID to find callers for.
        depth: Maximum depth to traverse.

    Returns:
        List of dicts with "id", "depth", and "file_line" for each caller.
    """
    _, reverse = build_adjacency(call_graph)
    node_map = {n["id"]: n for n in call_graph.get("nodes", [])}

    if func_id not in reverse:
        return []

    reachable = bfs_reach({func_id}, reverse, max_depth=depth)

    # Exclude the function itself
    reachable.pop(func_id, None)

    result = []
    for nid, d in sorted(reachable.items(), key=lambda x: (x[1], x[0])):
        node = node_map.get(nid, {})
        result.append({
            "id": nid,
            "short_id": make_short_id(nid),
            "depth": d,
            "file_line": make_file_line(node),
        })

    return result


def get_callees(
    call_graph: dict,
    func_id: str,
    depth: int = 1,
) -> list[dict]:
    """Get transitive callees up to depth.

    Args:
        call_graph: Call graph with "nodes" and "edges".
        func_id: Function ID to find callees for.
        depth: Maximum depth to traverse.

    Returns:
        List of dicts with "id", "depth", and "file_line" for each callee.
    """
    forward, _ = build_adjacency(call_graph)
    node_map = {n["id"]: n for n in call_graph.get("nodes", [])}

    if func_id not in forward:
        return []

    reachable = bfs_reach({func_id}, forward, max_depth=depth)

    # Exclude the function itself
    reachable.pop(func_id, None)

    result = []
    for nid, d in sorted(reachable.items(), key=lambda x: (x[1], x[0])):
        node = node_map.get(nid, {})
        result.append({
            "id": nid,
            "short_id": make_short_id(nid),
            "depth": d,
            "file_line": make_file_line(node),
        })

    return result


def filter_by_type(call_graph: dict, filter_type: str) -> list[dict]:
    """Filter nodes by type: entry_points, endpoints, new, updated, orphans.

    Args:
        call_graph: Call graph with "nodes" and "edges".
        filter_type: One of "entry_points", "endpoints", "new", "updated", "orphans".

    Returns:
        List of dicts with "id", "short_id", "file_line" for matching nodes.
    """
    nodes = call_graph.get("nodes", [])

    type_mapping = {
        "entry_points": "start_point",
        "endpoints": "end_point",
        "orphans": "orphan",
    }

    status_filters = {"new", "updated"}

    result = []
    for node in nodes:
        nid = node["id"]
        node_type = node.get("type", "")
        node_status = node.get("status", "")

        match = False
        if filter_type in type_mapping:
            match = node_type == type_mapping[filter_type]
        elif filter_type in status_filters:
            match = node_status == filter_type

        if match:
            result.append({
                "id": nid,
                "short_id": make_short_id(nid),
                "file_line": make_file_line(node),
                "type": node_type,
                "status": node_status,
            })

    return sorted(result, key=lambda x: x["id"])


def fuzzy_match(query: str, node_ids: list[str]) -> list[str]:
    """Match partial function names or short IDs (e.g., 'analyze_pr' or 'cli:main').

    Args:
        query: Partial function name or short ID to search for.
        node_ids: List of full node IDs to search in.

    Returns:
        List of matching node IDs, sorted by best match first.
    """
    query_lower = query.lower()
    exact = []
    short_id_exact = []
    short_id_suffix = []
    func_exact = []
    suffix = []
    contains = []

    for nid in node_ids:
        nid_lower = nid.lower()
        # Extract function name (after ::)
        func_name = nid.split("::")[-1].lower()
        # Get short_id format (e.g., "mcp/server:analyze_pr")
        short_id = make_short_id(nid).lower()

        # Priority 1: Exact match on full ID
        if nid_lower == query_lower:
            exact.append(nid)
        # Priority 2: Exact match on short_id
        elif short_id == query_lower:
            short_id_exact.append(nid)
        # Priority 3: Short ID ends with query (e.g., "cli:main" matches "noodles/cli:main")
        elif short_id.endswith(query_lower):
            short_id_suffix.append(nid)
        # Priority 4: Exact match on function name
        elif func_name == query_lower:
            func_exact.append(nid)
        # Priority 5: Function name ends with query
        elif func_name.endswith(query_lower):
            suffix.append(nid)
        # Priority 6: Query contained in short_id, func_name, or full ID
        elif query_lower in short_id or query_lower in func_name or query_lower in nid_lower:
            contains.append(nid)

    return (
        sorted(exact)
        + sorted(short_id_exact)
        + sorted(short_id_suffix)
        + sorted(func_exact)
        + sorted(suffix)
        + sorted(contains)
    )


def make_short_id(node_id: str) -> str:
    """Convert full node ID to short readable format.

    'src/noodles/mcp/server.py::analyze_pr' -> 'mcp/server:analyze_pr'

    Args:
        node_id: Full node ID in format "path/to/file.py::func_name"

    Returns:
        Short ID like "module/file:func_name"
    """
    if "::" not in node_id:
        return node_id

    file_part, func_part = node_id.rsplit("::", 1)

    # Remove leading src/ and file extension
    path = file_part
    if path.startswith("src/"):
        path = path[4:]

    # Remove common prefix paths and extension
    parts = path.split("/")
    if len(parts) > 2:
        # Keep last 2 path components (module/file)
        parts = parts[-2:]

    # Remove extension from last part
    if parts:
        last = parts[-1]
        for ext in (".py", ".js", ".ts", ".tsx", ".jsx"):
            if last.endswith(ext):
                parts[-1] = last[: -len(ext)]
                break

    short_path = "/".join(parts)
    return f"{short_path}:{func_part}"


def make_file_line(node: dict) -> str:
    """Format 'src/path/file.py:42' from node with line field.

    Args:
        node: Node dict with "id" and optionally "line" field.

    Returns:
        String like "src/path/file.py:42" or just "src/path/file.py" if no line.
    """
    node_id = node.get("id", "")
    if "::" not in node_id:
        return node_id

    file_part = node_id.rsplit("::", 1)[0]
    line = node.get("line")

    if line:
        return f"{file_part}:{line}"
    return file_part


def resolve_function_id(
    query: str,
    call_graph: dict,
) -> str | None:
    """Resolve a partial function name to a full node ID.

    Args:
        query: Full or partial function name.
        call_graph: Call graph to search in.

    Returns:
        Resolved full node ID, or None if not found or ambiguous.
    """
    node_ids = [n["id"] for n in call_graph.get("nodes", [])]

    # Check for exact match first
    if query in node_ids:
        return query

    # Try fuzzy match
    matches = fuzzy_match(query, node_ids)
    if len(matches) == 1:
        return matches[0]
    elif matches:
        # Return first match if multiple found
        return matches[0]

    return None
