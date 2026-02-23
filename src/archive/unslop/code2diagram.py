import asyncio
import json
import logging
from pathlib import Path
from typing import Optional
from .llm import get_llm_client

class AuthenticationError(Exception):
    pass

logger = logging.getLogger(__name__)
_LAST_API_KEY_STATUS: Optional[str] = None


def _report_api_key_status(status: str) -> None:
    global _LAST_API_KEY_STATUS
    if status == _LAST_API_KEY_STATUS:
        return
    _LAST_API_KEY_STATUS = status
    logger.warning(
        "LLM API key status: %s",
        status,
        extra={"unslop_api_key_status": status},
    )


def _maybe_report_auth_error(exc: Exception) -> None:
    if isinstance(exc, AuthenticationError):
        _report_api_key_status("invalid")
        return
    status_code = getattr(exc, "status_code", None)
    if status_code == 401:
        _report_api_key_status("invalid")
        return
    text = str(exc)
    if "invalid_api_key" in text or "Incorrect API key" in text:
        _report_api_key_status("invalid")

def _overview_prompt() -> str:
    return (
        "You are a code analyst. Identify ONLY user-facing, interactive entry points"
        " in the project: CLI commands, HTTP routes, UI pages/components, public API endpoints."
        " Exclude internal helpers, private functions, background jobs, and indirect calls"
        " unless directly invoked by a user or external system."
        " For each user entry point, follow the flow through the code to user-visible outcomes."
        " Identify feature blocks along the flow and the concrete end(s)."
        " Keep each feature block focused; avoid stuffing multiple unrelated ideas into one block."
        " Feature blocks must include an array of related files and line numbers."
        " Every node must belong to at least one flow; no orphan nodes are allowed."
        " Return JSON only with top-level keys: nodes, flows."
        " nodes: array of objects {id, type, name, description, status, files:[{file, lines:[start,end]}]}."
        " - type must be one of: entry_point, feature_block, end."
        " - status must be one of: added, updated, unchanged."
        " - id must be unique across all nodes and used directly in flows."
        " - id format: for entry_point use prefix 'ep_', for feature_block use prefix 'fb_',"
        "   for end use prefix 'end_'. IDs must contain only lowercase letters and underscores."
        " - files is required for entry_point, feature_block, and end nodes."
        " flows: array of objects {from, to, description, status}."
        " - from and to are node ids."
        " - description is a short, user-visible transition label."
        " - status must be one of: added, updated, unchanged."
        " Graph constraints:"
        " - entry_point nodes have only outgoing edges (no incoming)."
        " - feature_block nodes have at least one incoming and at least one outgoing edge."
        " - end nodes have only incoming edges (no outgoing)."
        " Coverage constraint:"
        " - The union of all nodes' files/lines must cover all lines in all files referenced."
        " - Try not to let the same files/lines appear in multiple nodes."
        " For a fresh generation, mark every node and flow with status=added."
    )


def generate_overview_schema(combined, model=None):
    """Use LLM to identify user-facing entry points; return JSON string."""

    client = get_llm_client(model)
    system_prompt = "You are a precise code analyst."
    user_prompt = _overview_prompt() + "\n\n" + combined

    try:
        output_text = client.generate(system_prompt, user_prompt, json_format=True)
    except Exception as exc:
        _maybe_report_auth_error(exc)
        raise
    _report_api_key_status("valid")
    return output_text


def update_overview_schema(combined, model=None, previous_schema=""):
    """Update an overview schema using changed files and a previous schema."""
    client = get_llm_client(model)
    system_prompt = "You are a precise code analyst."
    user_prompt = (
        _overview_prompt()
        + " You are updating an existing overview schema."
        " Input files are ONLY the files that changed."
        " Use the previous schema for unchanged parts of the system."
        " Add, update, or remove nodes and flows based on the changed files."
        " Deleted nodes and flows must be removed and not appear in the output."
        " If a node references updated files, re-match its files and line ranges"
        " to the latest content in the changed files."
        " Output the full updated overview schema with status labels."
        " Use status=added for new nodes/flows, status=updated for retained ones"
        " that are affected by changes, and status=unchanged for retained ones"
        " that are not affected by changes."
        + "\n\nPREVIOUS_SCHEMA_JSON:\n"
        + (previous_schema or "")
        + "\n\nCHANGED_FILES:\n"
        + combined
    )

    try:
        output_text = client.generate(system_prompt, user_prompt, json_format=True)
    except Exception as exc:
        _maybe_report_auth_error(exc)
        raise
    _report_api_key_status("valid")
    return output_text


def get_overview_d2_diagram(schema_path, output_dir=None):
    """Generate a D2 diagram from the entry points JSON file path."""

    payload = json.loads(Path(schema_path).read_text(encoding="utf-8"))

    def escape_label(text):
        return (text or "").replace('"', '\\"')

    def wrap_words(text, max_words_per_line=5):
        words = (text or "").split()
        if not words:
            return ""
        lines = []
        for i in range(0, len(words), max_words_per_line):
            lines.append(" ".join(words[i : i + max_words_per_line]))
        return "\\n".join(lines)

    nodes = payload.get("nodes", [])
    flows = payload.get("flows", [])
    id_map = {}
    lines = []

    for node in nodes:
        raw_id = node.get("id", "")
        d2_id = raw_id
        id_map[raw_id] = d2_id
        label_base = node.get("name", raw_id)
        status = (node.get("status") or "").lower()
        status_tag = ""
        status_stroke = ""
        status_font = ""
        if status == "added":
            status_tag = "[added]"
            status_stroke = "#2E8B57"
            status_font = "#2E8B57"
        elif status == "updated":
            status_tag = "[updated]"
            status_stroke = "#D97706"
            status_font = "#D97706"
        if status_tag:
            label_base = f"{label_base}\\n{status_tag}"
        label = escape_label(label_base)
        node_type = node.get("type", "")
        tooltip = escape_label(node.get("description") or raw_id)
        fill = ""
        if node_type == "entry_point":
            shape = "oval"
            fill = "#D9EFF2"
        elif node_type == "feature_block":
            shape = "rectangle"
            fill = "#FCE2A7"
        elif node_type == "end":
            shape = "diamond"
            fill = "#F4DAD1"
        lines.append(f'{d2_id}: "{label}"')
        lines.append(f"{d2_id}.shape: {shape}")
        lines.append(f'{d2_id}.tooltip: "{tooltip}"')
        lines.append(f'{d2_id}.link: "unslop://node/{escape_label(raw_id)}"')
        if fill:
            lines.append(f'{d2_id}.style.fill: "{fill}"')
        if status_stroke:
            lines.append(f'{d2_id}.style.stroke: "{status_stroke}"')
            lines.append(f'{d2_id}.style.stroke-width: 10')
        if status_font:
            lines.append(f'{d2_id}.style.font-color: "{status_font}"')

    seen_edges = set()
    for flow in flows:
        from_id = id_map.get(flow.get("from"))
        to_id = id_map.get(flow.get("to"))
        if not from_id or not to_id:
            continue
        desc = escape_label(wrap_words(flow.get("description", ""), max_words_per_line=5))
        edge_key = (from_id, to_id, desc)
        if edge_key in seen_edges:
            continue
        seen_edges.add(edge_key)
        if desc:
            lines.append(f'{from_id} -> {to_id}: "{desc}"')
        else:
            lines.append(f"{from_id} -> {to_id}")

    d2_text = "\n".join(lines).rstrip() + "\n"
    if output_dir:
        output_root = Path(output_dir)
        output_root.mkdir(parents=True, exist_ok=True)
        (output_root / "overview.d2").write_text(d2_text, encoding="utf-8")
    return d2_text


async def generate_node_schema_and_diagram(
    schema_path,
    src_dir="src",
    model="gpt-4.1-mini",
    output_dir=None,
    include_node_ids=None,
):
    """
    Load entry points JSON and return per-node D2 diagrams.
    """
    schema_json = json.loads(Path(schema_path).read_text(encoding="utf-8"))
    src_root = Path(src_dir)

    def extract_code_context():
        flows_list = schema_json.get("flows", [])
        nodes_list = schema_json.get("nodes", [])
        node_by_id = {node.get("id"): node for node in nodes_list}
        contexts = []
        for node in nodes_list:
            sections = []
            for file_ref in node.get("files", []):
                file_path = file_ref["file"]
                lines = file_ref["lines"]
                abs_path = src_root / file_path
                try:
                    text = abs_path.read_text(encoding="utf-8")
                except UnicodeDecodeError:
                    text = abs_path.read_text(encoding="latin-1")
                start, end = lines
                # Keep line numbers 1-based and inclusive.
                selected = text.splitlines()[start - 1 : end]
                formatted = [f"### FILE: {file_path}"]
                for i, line in enumerate(selected, 1):
                    formatted.append(f"[{i}] {line}")
                sections.append("\n".join(formatted))

            node_id = node.get("id")
            incoming = [
                {
                    "from_id": flow.get("from", ""),
                    "from_name": node_by_id.get(flow.get("from"), {}).get("name", ""),
                    "from_description": node_by_id.get(flow.get("from"), {}).get(
                        "description", ""
                    ),
                    "flow_description": flow.get("description", ""),
                }
                for flow in flows_list
                if flow.get("to") == node_id
            ]
            outgoing = [
                {
                    "to_id": flow.get("to", ""),
                    "to_name": node_by_id.get(flow.get("to"), {}).get("name", ""),
                    "to_description": node_by_id.get(flow.get("to"), {}).get(
                        "description", ""
                    ),
                    "flow_description": flow.get("description", ""),
                }
                for flow in flows_list
                if flow.get("from") == node_id
            ]

            context_parts = [
                "BLOCK:",
                f"id: {node_id}",
                f"type: {node.get('type', '')}",
                f"name: {node.get('name', '')}",
                f"description: {node.get('description', '')}",
                "",
                "FILES:",
                "\n\n".join(sections) if sections else "(none)",
                "",
                "INCOMING CONNECTIONS:",
                "\n".join(
                    "name: "
                    f"{item.get('from_name', '')} | id: {item.get('from_id', '')} | description: {item.get('from_description', '')}"
                    f" | flow_description: {item.get('flow_description', '')}"
                    for item in incoming
                ) or "(none)",
                "",
                "OUTGOING CONNECTIONS:",
                "\n".join(
                    "name: "
                    f"{item.get('to_name', '')} | id: {item.get('to_id', '')} | description: {item.get('to_description', '')}"
                    f" | flow_description: {item.get('flow_description', '')}"
                    for item in outgoing
                ) or "(none)",
            ]
            contexts.append({"id": node_id, "context": "\n".join(context_parts)})
        return contexts

    async def run_request(client, node):
        node_id = node.get("id")
        payload = node.get("context", "")
        system_prompt = "You are a precise code analyst."
        user_prompt = (
            "You are a code analyst. Build a graph that shows the high-level"
            " data flow and control flow from incoming to outgoing using the block's"
            " files/code and its incoming/outgoing connections."
            " Requirements:"
            " 1) Make each function a node."
            " 2) If there's an external call inside a function, create exactly two"
            "    connections between the function node and the external function node:"
            "    (a) from the caller function to the callee function,"
            "    and (b) from the callee function to the caller function."
            " 3) Add nodes and connections for each incoming and outgoing. You should use the original "
            "    name, id, description and flow_description."
            " Return JSON only. Ignore the old JSON format and use this new schema:"
            " {"
            '   "nodes": ['
            "     {"
            '       "id": "unique_node_id",'
            '       "type": "function|incoming|outgoing",'
            '       "name": "function_name_if_type_function",'
            '       "description": "concise description of the node behavior"'
            "     }"
            "   ],"
            '   "connections": ['
            "     {"
            '       "from": "node_or_function_id",'
            '       "to": "node_or_function_id",'
            '       "description": "data or control description of the sent/return value (non-empty)"'
            "     }"
            "   ]"
            " }"
            " ID rules:"
            " - ids must contain only lowercase letters and underscores."
            + "\n\nNODE CONTEXT:\n" + payload
        )
        try:
            output_text = await client.generate_async(system_prompt, user_prompt, json_format=True)
        except Exception as exc:
            _maybe_report_auth_error(exc)
            raise
        _report_api_key_status("valid")
        return {"id": node_id, "json": output_text}
    
    node_contexts = extract_code_context()
    if include_node_ids is not None:
        node_contexts = [
            node for node in node_contexts if node.get("id") in include_node_ids
        ]
        if not node_contexts:
            return []
    client = get_llm_client(model)
    tasks = [run_request(client, node) for node in node_contexts]
    results = await asyncio.gather(*tasks)
    if output_dir:
        output_root = Path(output_dir)
        output_root.mkdir(parents=True, exist_ok=True)
        for entry in results:
            node_id = entry.get("id")
            if node_id:
                schema_path = output_root / f"{node_id}.json"
                schema_path.write_text(entry.get("json", ""), encoding="utf-8")
            result = get_node_d2_diagram(entry, output_dir=output_dir)
            diagram_id = result.get("id")
            if diagram_id:
                logger.info("Generated node diagram: %s", diagram_id)
    return results


def get_node_d2_diagram(node_schema, output_dir=None):
    """Generate a D2 diagram for a single node schema."""

    def escape_label(text):
        return (text or "").replace('"', '\\"')

    def wrap_words(text, max_words_per_line=4):
        words = (text or "").split()
        if not words:
            return ""
        lines = []
        for i in range(0, len(words), max_words_per_line):
            lines.append(" ".join(words[i : i + max_words_per_line]))
        return "\\n".join(lines)

    def shape_for_node(node_type):
        if node_type == "incoming":
            return "oval"
        elif node_type == "outgoing":
            return "diamond"
        return "rectangle"

    def fill_for_node(node_type):
        if node_type == "function":
            return "#FAEBC9"
        return ""

    output_root = Path(output_dir) if output_dir else None
    if output_root:
        output_root.mkdir(parents=True, exist_ok=True)

    node_id = node_schema.get("id")
    json_text = node_schema.get("json", "")
    try:
        payload = json.loads(json_text)
    except json.JSONDecodeError:
        payload = {}

    nodes = payload.get("nodes", [])
    connections = payload.get("connections", [])
    lines = []
    lines.append("direction: right")

    for node in nodes:
        d2_id = node.get("id")
        label = escape_label(node.get("name") or d2_id)
        tooltip = escape_label(node.get("description", ""))
        lines.append(f'{d2_id}: "{label}"')
        lines.append(f"{d2_id}.shape: {shape_for_node(node.get('type'))}")
        fill = fill_for_node(node.get("type"))
        if fill:
            lines.append(f'{d2_id}.style.fill: "{fill}"')
        if tooltip:
            lines.append(f'{d2_id}.tooltip: "{tooltip}"')
        if node.get("type") in {"incoming", "outgoing"}:
            lines.append(f'{d2_id}.link: "unslop://node/{escape_label(d2_id)}"')

    seen_edges = set()
    for connection in connections:
        from_id = connection.get("from")
        to_id = connection.get("to")
        value = connection.get("description", "")
        if not from_id or not to_id:
            continue
        wrapped_label = wrap_words(value, max_words_per_line=5)
        label = escape_label(wrapped_label)
        edge_key = (from_id, to_id, label)
        if edge_key in seen_edges:
            continue
        seen_edges.add(edge_key)
        lines.append(f'{from_id} -> {to_id}: "{label}"')

    d2_text = "\n".join(lines).rstrip() + "\n"
    if output_root and node_id:
        out_path = output_root / f"{node_id}.d2"
        out_path.write_text(d2_text, encoding="utf-8")

    return {"id": node_id, "d2": d2_text}
