import sys
from collections import defaultdict
from pathlib import Path

try:
    from tree_sitter_language_pack import get_parser as _ts_get_parser
    from tree_sitter import Node
except ImportError as e:
    print(
        f"Error: tree-sitter-language-pack not installed. Run:\n"
        f"  pip install tree-sitter-language-pack",
        file=sys.stderr,
    )
    raise


# Directories to skip when walking the repo
SKIP_DIRS = {
    ".git", "node_modules", "__pycache__", ".venv", "venv",
    "dist", "build", ".mypy_cache", ".pytest_cache", "site-packages",
}

# File extensions and their tree-sitter language modules
EXTENSION_TO_LANG = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "tsx",
}


def _get_parser(lang: str):
    """Return a tree-sitter parser for the given language name."""
    try:
        return _ts_get_parser(lang)
    except Exception:
        raise ValueError(f"Unsupported language: {lang}")


def build_call_graph(
    repo_path: Path,
) -> tuple[dict, list[str], list[str], list[str]]:
    """Analyze a repository and build a function call graph.

    Each function is classified by its degree:
      start_point - out-degree only (calls others, not called by anyone)
      end_point   - in-degree only  (called by others, calls nobody)
      process     - both in and out degree
      orphan      - no connections at all

    Returns:
        call_graph:   {"nodes": [{"id": ..., "type": ..., "callers": [...], "callees": [...]}],
                       "edges": [{"from": ..., "to": ...}]}
        start_points: list of indices classified as start_point
        end_points:   list of indices classified as end_point
        orphans:      list of indices classified as orphan
    """
    repo_path = repo_path.resolve()

    # Phase 1: Collect all function definitions across all files
    # Maps index -> (file_path, func_node)
    func_defs: dict[str, tuple[Path, Node]] = {}
    # Maps bare function name -> list of full indices (for call resolution)
    name_to_indices: dict[str, list[str]] = defaultdict(list)

    for file_path in _iter_source_files(repo_path):
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
        rel_path = str(file_path.relative_to(repo_path))

        funcs = _find_functions_in_file(tree.root_node, lang)
        for qualified_name, func_node in funcs:
            index = f"{rel_path}::{qualified_name}"
            if index in func_defs:
                # Duplicate — skip (e.g., same name in same class)
                continue
            func_defs[index] = (file_path, func_node)
            # Register by bare name (last component) for call resolution
            bare_name = qualified_name.split(".")[-1]
            name_to_indices[bare_name].append(index)

    # Phase 2: Build call graph
    call_graph_map: dict[str, list[str]] = {}
    is_returned_map: dict[tuple[str, str], bool] = {}
    call_row_map: dict[tuple[str, str], int] = {}

    for index, (file_path, func_node) in func_defs.items():
        lang = EXTENSION_TO_LANG[file_path.suffix]
        rel_path = str(file_path.relative_to(repo_path))

        call_info = _get_direct_calls(func_node, lang)
        connections: dict[str, tuple[bool, int]] = {}

        for call_name, (is_returned, row) in call_info.items():
            if call_name not in name_to_indices:
                continue
            candidates = name_to_indices[call_name]
            # Prefer same-file functions, otherwise include all candidates
            same_file = [c for c in candidates if c.startswith(f"{rel_path}::")]
            matched = same_file if same_file else candidates
            for callee_idx in matched:
                if callee_idx in connections:
                    prev_is_ret, prev_row = connections[callee_idx]
                    connections[callee_idx] = (prev_is_ret or is_returned, min(prev_row, row))
                else:
                    connections[callee_idx] = (is_returned, row)

        # Remove self-references
        connections.pop(index, None)
        call_graph_map[index] = sorted(connections.keys())
        for callee_idx, (is_ret, row) in connections.items():
            is_returned_map[(index, callee_idx)] = is_ret
            call_row_map[(index, callee_idx)] = row

    # Phase 3: Compute callers (reverse map) and classify each function
    callers_map: dict[str, list[str]] = defaultdict(list)
    for caller_idx, callees in call_graph_map.items():
        for callee_idx in callees:
            callers_map[callee_idx].append(caller_idx)

    start_points: list[str] = []
    end_points: list[str] = []
    orphans: list[str] = []

    nodes: list[dict] = []
    edges: list[dict] = []

    for idx, callees in sorted(call_graph_map.items()):
        callers = sorted(callers_map.get(idx, []))
        has_out = len(callees) > 0
        has_in = len(callers) > 0

        if has_out and has_in:
            node_type = "process"
        elif has_out:
            node_type = "start_point"
            start_points.append(idx)
        elif has_in:
            node_type = "end_point"
            end_points.append(idx)
        else:
            node_type = "orphan"
            orphans.append(idx)

        source_code = func_defs[idx][1].text.decode("utf-8") if idx in func_defs else ""
        # tree-sitter is 0-indexed, add 1 for human-readable line numbers
        line_num = func_defs[idx][1].start_point[0] + 1 if idx in func_defs else None
        nodes.append({"id": idx, "type": node_type, "line": line_num, "callers": callers, "callees": callees, "source": source_code})

        # Sort callees by their row position in the caller, then assign index
        sorted_callees = sorted(callees, key=lambda c: call_row_map.get((idx, c), 0))
        for seq, callee_idx in enumerate(sorted_callees):
            edges.append({
                "from": idx,
                "to": callee_idx,
                "is_returned": is_returned_map.get((idx, callee_idx), True),
                "index": seq,
            })

    call_graph = {"nodes": nodes, "edges": edges}
    return call_graph, start_points, end_points, orphans


def _iter_source_files(repo_path: Path):
    """Yield all supported source files, skipping common non-source directories."""
    for file_path in repo_path.rglob("*"):
        if not file_path.is_file():
            continue
        if file_path.suffix not in EXTENSION_TO_LANG:
            continue
        # Skip non-source directories
        if any(part in SKIP_DIRS for part in file_path.parts):
            continue
        yield file_path


def _find_functions_in_file(root: Node, lang: str) -> list[tuple[str, Node]]:
    """Find all named function definitions, returning (qualified_name, node) pairs."""
    results: list[tuple[str, Node]] = []
    _walk_for_funcs(root, lang, results, class_stack=[])
    return results


def _walk_for_funcs(
    node: Node, lang: str, results: list, class_stack: list[str]
) -> None:
    """Recursively walk the AST to collect function definitions with class context."""
    if lang == "python":
        if node.type == "class_definition":
            name_node = node.child_by_field_name("name")
            cls_name = name_node.text.decode("utf-8") if name_node else "<class>"
            body = node.child_by_field_name("body")
            if body:
                for child in body.children:
                    _walk_for_funcs(child, lang, results, class_stack + [cls_name])
            return

        if node.type == "function_definition":
            name_node = node.child_by_field_name("name")
            if name_node:
                func_name = name_node.text.decode("utf-8")
                qualified = ".".join(class_stack + [func_name])
                results.append((qualified, node))
            # Recurse into the function body for nested functions/classes
            body = node.child_by_field_name("body")
            if body:
                for child in body.children:
                    _walk_for_funcs(child, lang, results, class_stack)
            return

    elif lang in ("javascript", "typescript", "tsx"):
        if node.type == "class_declaration":
            name_node = node.child_by_field_name("name")
            cls_name = name_node.text.decode("utf-8") if name_node else "<class>"
            body = node.child_by_field_name("body")
            if body:
                for child in body.children:
                    _walk_for_funcs(child, lang, results, class_stack + [cls_name])
            return

        if node.type == "function_declaration":
            name_node = node.child_by_field_name("name")
            if name_node:
                func_name = name_node.text.decode("utf-8")
                qualified = ".".join(class_stack + [func_name])
                results.append((qualified, node))
            body = node.child_by_field_name("body")
            if body:
                for child in body.children:
                    _walk_for_funcs(child, lang, results, class_stack)
            return

        if node.type == "method_definition":
            name_node = node.child_by_field_name("name")
            if name_node:
                func_name = name_node.text.decode("utf-8")
                qualified = ".".join(class_stack + [func_name])
                results.append((qualified, node))
            body = node.child_by_field_name("body")
            if body:
                for child in body.children:
                    _walk_for_funcs(child, lang, results, class_stack)
            return

        if node.type == "variable_declarator":
            # Handle: const foo = () => { ... } or const foo = function() { ... }
            name_node = node.child_by_field_name("name")
            value_node = node.child_by_field_name("value")
            if (
                name_node
                and value_node
                and value_node.type in ("arrow_function", "function")
            ):
                func_name = name_node.text.decode("utf-8")
                qualified = ".".join(class_stack + [func_name])
                results.append((qualified, value_node))
                body = value_node.child_by_field_name("body")
                if body:
                    for child in body.children:
                        _walk_for_funcs(child, lang, results, class_stack)
            return

    for child in node.children:
        _walk_for_funcs(child, lang, results, class_stack)


def _get_direct_calls(func_node: Node, lang: str) -> dict[str, tuple[bool, int]]:
    """Get all function call names with is_returned flag and row position.

    Returns dict mapping call name to (is_returned, row). If the same function
    is called multiple times, is_returned is True if any call's return value is
    used, and row is the position of the earliest call.
    """
    calls: list[tuple[str, bool, int]] = []
    body = func_node.child_by_field_name("body")
    if body is None:
        # For arrow functions with expression body, the body is the expression
        for child in func_node.children:
            if child.type not in ("formal_parameters", "parameters", "=>", "async"):
                _collect_calls(child, calls, lang)
    else:
        _collect_calls(body, calls, lang)
    result: dict[str, tuple[bool, int]] = {}
    for name, is_ret, row in calls:
        if name in result:
            prev_is_ret, prev_row = result[name]
            result[name] = (prev_is_ret or is_ret, min(prev_row, row))
        else:
            result[name] = (is_ret, row)
    return result


def _collect_calls(node: Node, calls: list, lang: str) -> None:
    """Walk node collecting calls, stopping at nested function definitions."""
    if lang == "python":
        is_func_def = node.type == "function_definition"
        is_call = node.type == "call"
    else:
        is_func_def = node.type in (
            "function_declaration", "function", "arrow_function", "method_definition"
        )
        is_call = node.type == "call_expression"

    # Stop recursing into nested function bodies
    if is_func_def:
        return

    if is_call:
        name = _extract_callee_name(node, lang)
        if name:
            # Parent is expression_statement → fire-and-forget (is_returned=False)
            is_returned = not (node.parent and node.parent.type == "expression_statement")
            row = node.start_point[0]
            calls.append((name, is_returned, row))

    # Detect JSX component usage as calls (React/Next.js)
    if lang in ("javascript", "typescript", "tsx"):
        if node.type in ("jsx_self_closing_element", "jsx_opening_element"):
            component_name = _extract_jsx_component_name(node)
            if component_name and _is_custom_component(component_name):
                row = node.start_point[0]
                calls.append((component_name, True, row))

    for child in node.children:
        _collect_calls(child, calls, lang)


def _extract_callee_name(call_node: Node, lang: str) -> str | None:
    """Extract the bare function/method name from a call expression node."""
    if lang == "python":
        func = call_node.child_by_field_name("function")
    else:
        func = call_node.child_by_field_name("function")

    if func is None:
        return None

    if func.type == "identifier":
        return func.text.decode("utf-8")

    if func.type == "attribute" and lang == "python":
        attr = func.child_by_field_name("attribute")
        return attr.text.decode("utf-8") if attr else None

    if func.type == "member_expression":
        prop = func.child_by_field_name("property")
        return prop.text.decode("utf-8") if prop else None

    return None


def _extract_jsx_component_name(jsx_node: Node) -> str | None:
    """Extract the component name from a JSX element node.

    Handles:
    - <MyComponent /> -> "MyComponent"
    - <components.MyComponent /> -> "MyComponent"
    """
    name_node = jsx_node.child_by_field_name("name")
    if name_node is None:
        return None

    # Simple identifier: <MyComponent />
    if name_node.type == "identifier":
        return name_node.text.decode("utf-8")

    # Member expression: <components.MyComponent />
    if name_node.type == "member_expression":
        prop = name_node.child_by_field_name("property")
        return prop.text.decode("utf-8") if prop else None

    return None


def _is_custom_component(name: str) -> bool:
    """Check if a JSX element is a custom component vs HTML element.

    React convention: custom components use PascalCase (start with uppercase).
    HTML elements are lowercase (div, span, button, etc.).
    """
    return name[0].isupper() if name else False
