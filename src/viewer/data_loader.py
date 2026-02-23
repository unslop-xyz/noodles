"""Load a result folder and build the data bundle for the viewer."""

import json
import re
from pathlib import Path

from utils import _sanitize_id


def load_result(result_dir: str | Path) -> dict:
    """Load a result folder and return the viewer data bundle.

    Args:
        result_dir: Path to the result folder containing call_graph.json
                    and diagram_*.mmd files.

    Returns:
        Dict with keys: diagrams, navigation, nodes, edges, id_map
    """
    result_path = Path(result_dir)

    # Load call graph
    cg_path = result_path / "call_graph.json"
    with open(cg_path, encoding="utf-8") as f:
        call_graph = json.load(f)

    # Load all diagram files
    diagrams: dict[str, str] = {}
    for mmd_file in sorted(result_path.glob("diagram_*.mmd")):
        # "diagram_main.mmd" -> "main", "diagram_Execute_CLI_tool.mmd" -> "Execute_CLI_tool"
        name = mmd_file.stem.removeprefix("diagram_")
        diagrams[name] = mmd_file.read_text(encoding="utf-8")

    # Build id_map: sanitized_mermaid_id -> original_node_id
    id_map: dict[str, str] = {}
    for node in call_graph["nodes"]:
        sanitized = _sanitize_id(node["id"])
        id_map[sanitized] = node["id"]

    # Build nodes metadata lookup: original_id -> metadata
    nodes: dict[str, dict] = {}
    for node in call_graph["nodes"]:
        nid = node["id"]
        # Extract file path and function name from the id
        parts = nid.split("::")
        file_path = parts[0] if len(parts) > 1 else ""
        func_name = parts[-1]

        nodes[nid] = {
            "name": node.get("name", func_name),
            "description": node.get("description", ""),
            "type": node.get("type", "process"),
            "tag": node.get("tag", ""),
            "file_path": file_path,
            "function": func_name,
            "status": node.get("status", "unchanged"),
            "update": node.get("update", ""),
        }

    # Build edges metadata lookup: "sanitized_from -> sanitized_to" -> metadata
    edges: dict[str, dict] = {}
    for edge in call_graph["edges"]:
        key = f"{_sanitize_id(edge['from'])} -> {_sanitize_id(edge['to'])}"
        edges[key] = {
            "label": edge.get("label", ""),
            "description": edge.get("description", ""),
            "args": edge.get("args", ""),
            "condition": edge.get("condition"),
            "is_returned": edge.get("is_returned", False),
            "index": edge.get("index"),
        }

    # Build navigation map: for each diagram, which nodes link to which sub-diagrams
    # A node is clickable if a sub-diagram file exists matching its sanitized name
    available_diagrams = set(diagrams.keys())
    mermaid_id_to_sub: dict[str, str] = {}
    for node in call_graph["nodes"]:
        # Extract function name from node ID (e.g., "src/foo.py::my_func" -> "my_func")
        # or use the explicit name field if present
        name = node.get("name", "")
        if not name:
            node_id = node.get("id", "")
            if "::" in node_id:
                name = node_id.split("::")[-1]
        if not name:
            continue
        sanitized_name = _sanitize_id(name)
        mermaid_id = _sanitize_id(node["id"])
        if sanitized_name in available_diagrams:
            mermaid_id_to_sub[mermaid_id] = sanitized_name
        else:
            for i in range(2, 10):
                candidate = f"{sanitized_name}_{i}"
                if candidate in available_diagrams:
                    mermaid_id_to_sub[mermaid_id] = candidate
                    break

    # Parse each diagram to extract node IDs and build nav links
    navigation: dict[str, dict[str, str]] = {}
    for diagram_name, source in diagrams.items():
        nav: dict[str, str] = {}
        for line in source.split("\n"):
            stripped = line.strip()
            # Node definition: ID followed by shape bracket or brace
            match = re.match(r"(\w+)\s*[\(\[\{\"']", stripped)
            if not match:
                continue
            mermaid_id = match.group(1)
            if mermaid_id in mermaid_id_to_sub:
                target = mermaid_id_to_sub[mermaid_id]
                if target != diagram_name:  # Don't allow self-references
                    nav[mermaid_id] = target
        if nav:
            navigation[diagram_name] = nav

    return {
        "diagrams": diagrams,
        "navigation": navigation,
        "nodes": nodes,
        "edges": edges,
        "id_map": id_map,
    }
