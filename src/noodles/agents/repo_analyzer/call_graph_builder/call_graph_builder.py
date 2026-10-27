import re
import sys
from collections import defaultdict
from pathlib import Path

import pathspec

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
    ".claude", "vendor", "target", "obj",
}

# File extensions and their tree-sitter language modules
EXTENSION_TO_LANG = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".go": "go",
    ".java": "java",
    ".rs": "rust",
    ".cs": "csharp",
}

JS_LANGS = ("javascript", "typescript", "tsx")

# AST node types for languages handled by the spec-driven walker.
#   containers:  type-like nodes whose name qualifies the functions inside them
#   functions:   named function/method definitions
#   nested:      anonymous function scopes (calls inside are not attributed to the parent)
#   calls:       call-like nodes (including constructor calls via `new`)
#   instance:    receivers that refer to the current object/module, never external
LANG_SPECS = {
    "go": {
        "containers": set(),
        "functions": {"function_declaration", "method_declaration"},
        "nested": {"func_literal"},
        "calls": {"call_expression"},
        "instance": set(),
    },
    "java": {
        "containers": {
            "class_declaration", "interface_declaration", "enum_declaration",
            "record_declaration",
        },
        "functions": {
            "method_declaration", "constructor_declaration",
            "compact_constructor_declaration",
        },
        "nested": {"lambda_expression"},
        "calls": {"method_invocation", "object_creation_expression"},
        "instance": {"this", "super"},
    },
    "rust": {
        "containers": {"impl_item", "trait_item"},
        "functions": {"function_item"},
        "nested": {"closure_expression"},
        "calls": {"call_expression"},
        "instance": {"self", "Self", "super", "crate"},
    },
    "csharp": {
        "containers": {
            "class_declaration", "struct_declaration", "interface_declaration",
            "record_declaration", "enum_declaration",
        },
        "functions": {
            "method_declaration", "constructor_declaration", "local_function_statement",
        },
        "nested": {"lambda_expression", "anonymous_method_expression"},
        "calls": {"invocation_expression", "object_creation_expression"},
        "instance": {"this", "base"},
    },
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

    # Phase 1: Collect all function definitions and imports across all files
    # Maps index -> (file_path, func_node)
    func_defs: dict[str, tuple[Path, Node]] = {}
    # Maps bare function name -> list of full indices (for call resolution)
    name_to_indices: dict[str, list[str]] = defaultdict(list)
    # Maps rel_path -> dict of imports (local_name -> source_module)
    file_imports: dict[str, dict[str, str]] = {}
    # Maps rel_path -> language, used to filter out imports of the repo's own code
    file_langs: dict[str, str] = {}
    # Java packages / C# namespaces declared in the repo
    declared_packages: set[str] = set()

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

        # Collect imports for this file
        file_imports[rel_path] = _collect_file_imports(tree.root_node, lang)
        file_langs[rel_path] = lang
        if lang in ("java", "csharp"):
            declared_packages.update(_collect_declared_packages(tree.root_node))

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

    # Imports that resolve to the repo's own code are internal, not external libraries
    internal_modules = _find_internal_modules(repo_path, declared_packages)
    for rel_path, imports in file_imports.items():
        lang = file_langs[rel_path]
        file_imports[rel_path] = {
            name: src for name, src in imports.items()
            if not _is_internal_import(src, lang, internal_modules)
        }

    # Phase 2: Build call graph
    call_graph_map: dict[str, list[str]] = {}
    is_returned_map: dict[tuple[str, str], bool] = {}
    call_row_map: dict[tuple[str, str], int] = {}

    for index, (file_path, func_node) in func_defs.items():
        lang = EXTENSION_TO_LANG[file_path.suffix]
        rel_path = str(file_path.relative_to(repo_path))
        imports = file_imports.get(rel_path, {})

        call_info = _get_direct_calls(func_node, lang)
        connections: dict[str, tuple[bool, int]] = {}

        for call_name, (is_returned, row, receiver) in call_info.items():
            # Skip calls to external libraries (e.g., requests.get() should not match internal get())
            if _is_external_receiver(receiver, imports, lang):
                continue
            # Skip direct calls to imported functions (e.g., from subprocess import run; run())
            if _is_imported_function(call_name, receiver, imports):
                continue
            if call_name not in name_to_indices:
                continue
            candidates = _narrow_by_receiver(
                name_to_indices[call_name], receiver, index
            )
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


SELF_RECEIVERS = {"self", "cls", "this", "Self"}


def _owner(index: str) -> str:
    """Class/type part of a function index: "a.py::Foo.bar" → "Foo", "a.py::bar" → ""."""
    qualified = index.split("::", 1)[-1]
    return qualified.rpartition(".")[0]


def _narrow_by_receiver(
    candidates: list[str], receiver: str | None, caller: str
) -> list[str]:
    """Narrow name-matched candidates using the call's receiver, when it is informative.

    - self.foo() / this.foo() / Self::foo() → methods of the caller's own class
    - Foo.bar() / Foo::bar() where Foo is a class in the repo → Foo's methods
    Otherwise (or if narrowing leaves nothing) the candidates are returned unchanged.
    """
    if receiver is None:
        return candidates
    if receiver in SELF_RECEIVERS:
        caller_owner = _owner(caller)
        narrowed = [c for c in candidates if caller_owner and _owner(c) == caller_owner]
    else:
        type_name = receiver.split(".")[-1]
        narrowed = [c for c in candidates if _owner(c).split(".")[-1] == type_name]
    return narrowed or candidates


def _iter_source_files(repo_path: Path):
    """Yield all supported source files, skipping common non-source directories and gitignored files."""
    gitignore_spec = _load_gitignore(repo_path)

    for file_path in repo_path.rglob("*"):
        if not file_path.is_file():
            continue
        if file_path.suffix not in EXTENSION_TO_LANG:
            continue
        # Skip non-source directories
        if any(part in SKIP_DIRS for part in file_path.parts):
            continue
        # Skip files matched by .gitignore
        if gitignore_spec is not None:
            rel = str(file_path.relative_to(repo_path))
            if gitignore_spec.match_file(rel):
                continue
        yield file_path


def _load_gitignore(repo_path: Path) -> pathspec.PathSpec | None:
    """Load .gitignore patterns from the repo root, or return None if absent."""
    gitignore_path = repo_path / ".gitignore"
    if not gitignore_path.is_file():
        return None
    try:
        text = gitignore_path.read_text(encoding="utf-8", errors="replace")
        return pathspec.PathSpec.from_lines("gitwildmatch", text.splitlines())
    except OSError:
        return None


def _find_internal_modules(repo_path: Path, declared_packages: set[str]) -> dict[str, set[str]]:
    """Find module names that belong to the repo itself, per language.

    - python: top-level packages/modules at the repo root or under src/
    - go:     module paths from go.mod files
    - rust:   crate names from Cargo.toml files
    - java/csharp: packages/namespaces declared in the repo's source files
    """
    internal: dict[str, set[str]] = {
        "python": set(), "go": set(), "rust": set(),
        "java": set(declared_packages), "csharp": set(declared_packages),
    }

    for base in (repo_path, repo_path / "src"):
        if not base.is_dir():
            continue
        for child in base.iterdir():
            if child.is_dir() and (child / "__init__.py").is_file():
                internal["python"].add(child.name)
            elif child.is_file() and child.suffix == ".py":
                internal["python"].add(child.stem)

    for manifest in _iter_manifests(repo_path, ("go.mod", "Cargo.toml")):
        try:
            text = manifest.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if manifest.name == "go.mod":
            match = re.search(r"^module\s+(\S+)", text, re.MULTILINE)
            if match:
                internal["go"].add(match.group(1).strip('"'))
        else:
            # name = "..." inside [package] or [lib]
            for section in re.finditer(r"^\[(package|lib)\]([^\[]*)", text, re.MULTILINE):
                match = re.search(r'^\s*name\s*=\s*"([^"]+)"', section.group(2), re.MULTILINE)
                if match:
                    internal["rust"].add(match.group(1).replace("-", "_"))

    return internal


def _iter_manifests(repo_path: Path, names: tuple[str, ...]):
    """Yield project manifest files (go.mod, Cargo.toml, ...) outside skipped directories."""
    for name in names:
        for path in repo_path.rglob(name):
            rel_parts = path.relative_to(repo_path).parts
            if not any(part in SKIP_DIRS for part in rel_parts):
                yield path


def _is_internal_import(source: str, lang: str, internal_modules: dict[str, set[str]]) -> bool:
    """Check if an import source refers to the repo's own code."""
    modules = internal_modules.get(lang)
    if not modules:
        return False
    if lang in ("python", "rust"):
        return source in modules
    sep = "/" if lang == "go" else "."
    return any(source == m or source.startswith(m + sep) for m in modules)


def _collect_declared_packages(root: Node) -> set[str]:
    """Collect Java package / C# namespace names declared in a file."""
    packages: set[str] = set()
    stack = [root]
    while stack:
        node = stack.pop()
        if node.type == "package_declaration":
            for child in node.named_children:
                if child.type in ("scoped_identifier", "identifier"):
                    packages.add(child.text.decode("utf-8"))
        elif node.type in ("namespace_declaration", "file_scoped_namespace_declaration"):
            name_node = node.child_by_field_name("name")
            if name_node:
                packages.add(name_node.text.decode("utf-8"))
        # Packages/namespaces only appear at the top level or nested in namespaces
        if node.type in ("program", "compilation_unit", "declaration_list") or \
                node.type == "namespace_declaration":
            stack.extend(node.named_children)
    return packages


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

    elif lang in LANG_SPECS:
        spec = LANG_SPECS[lang]
        if node.type in spec["containers"]:
            cls_name = _container_name(node) or "<class>"
            body = node.child_by_field_name("body")
            if body:
                for child in body.children:
                    _walk_for_funcs(child, lang, results, class_stack + [cls_name])
            return

        if node.type in spec["functions"]:
            name_node = node.child_by_field_name("name")
            if name_node:
                func_name = name_node.text.decode("utf-8")
                # Go methods are declared at top level; qualify them by receiver type
                owner = None
                if lang == "go" and node.type == "method_declaration":
                    owner = _go_receiver_type(node)
                qualified = ".".join(
                    class_stack + ([owner] if owner else []) + [func_name]
                )
                results.append((qualified, node))
            body = node.child_by_field_name("body")
            if body:
                for child in body.children:
                    _walk_for_funcs(child, lang, results, class_stack)
            return

    for child in node.children:
        _walk_for_funcs(child, lang, results, class_stack)


def _container_name(node: Node) -> str | None:
    """Name of a class-like node. Rust impl blocks are named after the implemented type."""
    if node.type == "impl_item":
        type_node = node.child_by_field_name("type")
        return _base_type_name(type_node)
    name_node = node.child_by_field_name("name")
    return name_node.text.decode("utf-8") if name_node else None


def _base_type_name(type_node: Node | None) -> str | None:
    """Strip generics/pointers/paths from a type: `*Foo`, `Foo[T]`, `a::Foo<T>` → "Foo"."""
    if type_node is None:
        return None
    if type_node.named_child_count == 0:
        return type_node.text.decode("utf-8")
    for field in ("type", "name"):
        inner = type_node.child_by_field_name(field)
        if inner is not None:
            return _base_type_name(inner)
    for child in type_node.named_children:
        name = _base_type_name(child)
        if name:
            return name
    return None


def _go_receiver_type(method_node: Node) -> str | None:
    """Get the receiver type of a Go method: `func (s *Server) Run()` → "Server"."""
    receiver = method_node.child_by_field_name("receiver")
    if receiver is None:
        return None
    for param in receiver.named_children:
        if param.type == "parameter_declaration":
            return _base_type_name(param.child_by_field_name("type"))
    return None


def _get_direct_calls(func_node: Node, lang: str) -> dict[str, tuple[bool, int, str | None]]:
    """Get all function call names with is_returned flag, row position, and receiver.

    Returns dict mapping call name to (is_returned, row, receiver). If the same function
    is called multiple times, is_returned is True if any call's return value is
    used, and row is the position of the earliest call.
    """
    calls: list[tuple[str, bool, int, str | None]] = []
    body = func_node.child_by_field_name("body")
    if body is None:
        # For arrow functions with expression body, the body is the expression
        for child in func_node.children:
            if child.type not in ("formal_parameters", "parameters", "=>", "async"):
                _collect_calls(child, calls, lang)
    else:
        _collect_calls(body, calls, lang)
    result: dict[str, tuple[bool, int, str | None]] = {}
    for name, is_ret, row, receiver in calls:
        if name in result:
            prev_is_ret, prev_row, prev_receiver = result[name]
            result[name] = (prev_is_ret or is_ret, min(prev_row, row), prev_receiver or receiver)
        else:
            result[name] = (is_ret, row, receiver)
    return result


def _collect_calls(node: Node, calls: list, lang: str) -> None:
    """Walk node collecting calls, stopping at nested function definitions."""
    if lang == "python":
        is_func_def = node.type == "function_definition"
        is_call = node.type == "call"
    elif lang in LANG_SPECS:
        spec = LANG_SPECS[lang]
        is_func_def = node.type in spec["functions"] or node.type in spec["nested"]
        is_call = node.type in spec["calls"]
    else:
        is_func_def = node.type in (
            "function_declaration", "function", "arrow_function", "method_definition"
        )
        is_call = node.type == "call_expression"

    # Stop recursing into nested function bodies
    if is_func_def:
        return

    if is_call:
        name, receiver = _extract_callee_info(node, lang)
        if name:
            # Parent is expression_statement → fire-and-forget (is_returned=False)
            is_returned = not (node.parent and node.parent.type == "expression_statement")
            row = node.start_point[0]
            calls.append((name, is_returned, row, receiver))

    # Detect JSX component usage as calls (React/Next.js)
    if lang in JS_LANGS:
        if node.type in ("jsx_self_closing_element", "jsx_opening_element"):
            component_name = _extract_jsx_component_name(node)
            if component_name and _is_custom_component(component_name):
                row = node.start_point[0]
                calls.append((component_name, True, row, None))

    for child in node.children:
        _collect_calls(child, calls, lang)


def _extract_callee_info(call_node: Node, lang: str) -> tuple[str | None, str | None]:
    """Extract the function/method name and receiver from a call expression node.

    Returns:
        (method_name, receiver_name) tuple:
        - requests.get() → ("get", "requests")
        - self.helper() → ("helper", "self")
        - os.path.join() → ("join", "os.path")
        - helper() → ("helper", None)
        - HashMap::new() → ("new", "HashMap")   (Rust)
        - new Foo() → ("Foo", None)              (Java/C# constructor)
    """
    # Java: obj.method(...) has object/name fields instead of a function field
    if call_node.type == "method_invocation":
        name_node = call_node.child_by_field_name("name")
        receiver = _get_full_receiver_name(call_node.child_by_field_name("object"))
        return (name_node.text.decode("utf-8") if name_node else None, receiver)

    # Java/C#: `new Foo()` is a call to Foo's constructor
    if call_node.type == "object_creation_expression":
        return (_base_type_name(call_node.child_by_field_name("type")), None)

    func = call_node.child_by_field_name("function")

    if func is None:
        return (None, None)

    # Rust turbofish `foo::<T>()` / `x.foo::<T>()`: unwrap to the inner function
    if func.type == "generic_function":
        inner = func.child_by_field_name("function")
        func = inner if inner is not None else func

    if func.type == "identifier":
        return (func.text.decode("utf-8"), None)

    # C# generic call `Foo<T>()`
    if func.type == "generic_name":
        return (_base_type_name(func), None)

    member_fields = {
        "selector_expression": ("operand", "field"),         # Go
        "field_expression": ("value", "field"),              # Rust
        "scoped_identifier": ("path", "name"),               # Rust
        "member_access_expression": ("expression", "name"),  # C#
    }
    if func.type in member_fields:
        obj_field, name_field = member_fields[func.type]
        name_node = func.child_by_field_name(name_field)
        method_name = _base_type_name(name_node) if name_node else None
        receiver = _get_full_receiver_name(func.child_by_field_name(obj_field))
        return (method_name, receiver)

    if func.type == "attribute" and lang == "python":
        attr = func.child_by_field_name("attribute")
        method_name = attr.text.decode("utf-8") if attr else None
        receiver = _get_full_receiver_name(func.child_by_field_name("object"))
        return (method_name, receiver)

    if func.type == "member_expression":
        prop = func.child_by_field_name("property")
        method_name = prop.text.decode("utf-8") if prop else None
        receiver = _get_full_receiver_name(func.child_by_field_name("object"))
        return (method_name, receiver)

    return (None, None)


def _get_full_receiver_name(node: Node | None) -> str | None:
    """Get the full receiver name, handling chained attributes like os.path.

    Examples:
        - `requests` → "requests"
        - `os.path` → "os.path"
        - `self` → "self"
        - `this.service` → "this.service"
    """
    if node is None:
        return None

    if node.type in ("identifier", "type_identifier", "package_identifier"):
        return node.text.decode("utf-8")

    # (object field, attribute field) for member-access-like nodes
    member_fields = {
        "attribute": ("object", "attribute"),                # Python
        "member_expression": ("object", "property"),         # JS/TS
        "selector_expression": ("operand", "field"),         # Go
        "field_access": ("object", "field"),                 # Java
        "field_expression": ("value", "field"),              # Rust
        "scoped_identifier": ("path", "name"),               # Rust
        "member_access_expression": ("expression", "name"),  # C#
    }
    if node.type in member_fields:
        # For chained attributes like os.path, recursively build the full name
        obj_field, attr_field = member_fields[node.type]
        obj = node.child_by_field_name(obj_field)
        attr = node.child_by_field_name(attr_field)

        obj_name = _get_full_receiver_name(obj)
        attr_name = attr.text.decode("utf-8") if attr else None

        if obj_name and attr_name:
            return f"{obj_name}.{attr_name}"
        return obj_name or attr_name

    # Keywords referring to the current object/module (this, self, super, crate, base)
    if node.type in ("this", "self", "super", "crate", "base"):
        return node.text.decode("utf-8")

    return None


def _collect_file_imports(root: Node, lang: str) -> dict[str, str]:
    """Collect imports from a file, returning a map of local_name → source.

    Examples:
        Python:
        - import requests → {"requests": "requests"}
        - from subprocess import run → {"run": "subprocess"}
        - from . import utils → skipped (relative import)
        - from ..models import User → skipped (relative import)

        JavaScript/TypeScript:
        - import axios from 'axios' → {"axios": "axios"}
        - const fs = require('fs') → {"fs": "fs"}
        - import { get } from 'lodash' → {"get": "lodash"}
        - import * as path from 'path' → {"path": "path"}
        - import './local' → skipped (relative import)

        Go / Java / Rust / C# (source is the full import path, so it can be
        matched against the repo's own modules):
        - import "github.com/a/pkg" → {"pkg": "github.com/a/pkg"}
        - import java.util.List; → {"List": "java.util.List"}
        - use serde::{Serialize}; → {"Serialize": "serde"}
        - using IO = System.IO; → {"IO": "System.IO"}
    """
    imports: dict[str, str] = {}

    if lang == "python":
        _collect_python_imports(root, imports)
    elif lang in JS_LANGS:
        _collect_js_imports(root, imports)
    elif lang == "go":
        _collect_go_imports(root, imports)
    elif lang == "java":
        _collect_java_imports(root, imports)
    elif lang == "rust":
        _collect_rust_imports(root, imports)
    elif lang == "csharp":
        _collect_csharp_imports(root, imports)

    return imports


def _collect_go_imports(root: Node, imports: dict[str, str]) -> None:
    """Collect Go imports: the local name is the alias or the last path segment."""
    for node in _iter_descendants(root, "import_spec"):
        path_node = node.child_by_field_name("path")
        if path_node is None:
            continue
        path = path_node.text.decode("utf-8").strip('"`')
        name_node = node.child_by_field_name("name")
        if name_node is not None:
            local = name_node.text.decode("utf-8")
            if local in ("_", "."):
                continue
        else:
            segments = path.split("/")
            local = segments[-1]
            # Major version suffix: github.com/a/pkg/v2 is imported as `pkg`
            if re.fullmatch(r"v\d+", local) and len(segments) > 1:
                local = segments[-2]
        imports[local] = path


def _collect_java_imports(root: Node, imports: dict[str, str]) -> None:
    """Collect Java imports (classes and static members). Wildcards bind no known names."""
    for node in root.named_children:
        if node.type != "import_declaration":
            continue
        if any(child.type == "asterisk" for child in node.children):
            continue
        for child in node.named_children:
            if child.type in ("scoped_identifier", "identifier"):
                full = child.text.decode("utf-8")
                imports[full.split(".")[-1]] = full


def _collect_rust_imports(root: Node, imports: dict[str, str]) -> None:
    """Collect Rust `use` declarations, mapping each bound name to its root crate."""
    for node in _iter_descendants(root, "use_declaration"):
        arg = node.child_by_field_name("argument")
        if arg is not None:
            _collect_rust_use_tree(arg, None, imports)


def _collect_rust_use_tree(node: Node, root_crate: str | None, imports: dict[str, str]) -> None:
    """Walk a Rust use tree (`a::b::{c, d as e}`) recording name → root crate."""
    def leftmost(n: Node | None) -> str | None:
        while n is not None and n.type == "scoped_identifier":
            n = n.child_by_field_name("path")
        return n.text.decode("utf-8") if n is not None else None

    if node.type in ("identifier", "scoped_identifier"):
        crate = root_crate or leftmost(node)
        name = node.text.decode("utf-8").split("::")[-1]
        if crate not in ("crate", "self", "super") and name != "self":
            imports[name] = crate
    elif node.type == "use_as_clause":
        path = node.child_by_field_name("path")
        alias = node.child_by_field_name("alias")
        crate = root_crate or leftmost(path)
        if alias is not None and crate not in ("crate", "self", "super"):
            imports[alias.text.decode("utf-8")] = crate
    elif node.type == "scoped_use_list":
        crate = root_crate or leftmost(node.child_by_field_name("path"))
        use_list = node.child_by_field_name("list")
        if use_list is not None:
            _collect_rust_use_tree(use_list, crate, imports)
    elif node.type == "use_list":
        for child in node.named_children:
            _collect_rust_use_tree(child, root_crate, imports)


def _collect_csharp_imports(root: Node, imports: dict[str, str]) -> None:
    """Collect C# using aliases. Plain `using System;` binds no names, so it's skipped."""
    for node in _iter_descendants(root, "using_directive"):
        alias = node.child_by_field_name("name")
        if alias is None:
            continue
        for child in node.named_children:
            if child is not alias and child.type in ("qualified_name", "identifier"):
                imports[alias.text.decode("utf-8")] = child.text.decode("utf-8")
                break


def _iter_descendants(root: Node, node_type: str):
    """Yield all descendants of the given type, without descending into matches."""
    stack = [root]
    while stack:
        node = stack.pop()
        if node.type == node_type:
            yield node
            continue
        stack.extend(reversed(node.named_children))


def _collect_python_imports(node: Node, imports: dict[str, str]) -> None:
    """Collect Python imports recursively."""
    if node.type == "import_statement":
        # import requests, os
        for child in node.children:
            if child.type == "dotted_name":
                name = child.text.decode("utf-8")
                # Use root module as the source
                root_module = name.split(".")[0]
                imports[root_module] = root_module
            elif child.type == "aliased_import":
                # import requests as req
                name_node = child.child_by_field_name("name")
                alias_node = child.child_by_field_name("alias")
                if name_node and alias_node:
                    source = name_node.text.decode("utf-8").split(".")[0]
                    alias = alias_node.text.decode("utf-8")
                    imports[alias] = source
                elif name_node:
                    name = name_node.text.decode("utf-8")
                    root_module = name.split(".")[0]
                    imports[root_module] = root_module

    elif node.type == "import_from_statement":
        # from subprocess import run
        # from . import utils (relative - skip)
        module_node = node.child_by_field_name("module_name")
        if module_node:
            module_name = module_node.text.decode("utf-8")
            # Skip relative imports (those starting with .)
            if module_name.startswith("."):
                pass  # Skip relative imports - they're internal
            else:
                root_module = module_name.split(".")[0]
                # Collect imported names
                for child in node.children:
                    if child.type == "dotted_name":
                        name = child.text.decode("utf-8")
                        imports[name] = root_module
                    elif child.type == "aliased_import":
                        name_node = child.child_by_field_name("name")
                        alias_node = child.child_by_field_name("alias")
                        if alias_node:
                            imports[alias_node.text.decode("utf-8")] = root_module
                        elif name_node:
                            imports[name_node.text.decode("utf-8")] = root_module
        else:
            # Check for relative imports like "from . import foo"
            has_relative_prefix = any(
                c.type == "relative_import" or (c.type == "." and c.text == b".")
                for c in node.children
            )
            if has_relative_prefix:
                pass  # Skip relative imports

    for child in node.children:
        _collect_python_imports(child, imports)


def _collect_js_imports(node: Node, imports: dict[str, str]) -> None:
    """Collect JavaScript/TypeScript imports recursively."""
    if node.type == "import_statement":
        # import axios from 'axios'
        # import { get } from 'lodash'
        # import * as path from 'path'
        source_node = node.child_by_field_name("source")
        if source_node:
            source = source_node.text.decode("utf-8").strip("'\"")
            # Skip relative imports
            if source.startswith(".") or source.startswith("/"):
                pass
            else:
                # Get the package name (first part before /)
                package = source.split("/")[0]
                # Handle scoped packages like @types/node
                if package.startswith("@") and "/" in source:
                    parts = source.split("/")
                    package = f"{parts[0]}/{parts[1]}" if len(parts) > 1 else parts[0]

                # Collect imported names
                for child in node.children:
                    if child.type == "identifier":
                        imports[child.text.decode("utf-8")] = package
                    elif child.type == "import_clause":
                        for clause_child in child.children:
                            if clause_child.type == "identifier":
                                imports[clause_child.text.decode("utf-8")] = package
                            elif clause_child.type == "namespace_import":
                                # import * as fs from 'fs'
                                for ns_child in clause_child.children:
                                    if ns_child.type == "identifier":
                                        imports[ns_child.text.decode("utf-8")] = package
                            elif clause_child.type == "named_imports":
                                # import { get, post } from 'axios'
                                for spec in clause_child.children:
                                    if spec.type == "import_specifier":
                                        alias = spec.child_by_field_name("alias")
                                        name = spec.child_by_field_name("name")
                                        if alias:
                                            imports[alias.text.decode("utf-8")] = package
                                        elif name:
                                            imports[name.text.decode("utf-8")] = package

    elif node.type == "lexical_declaration" or node.type == "variable_declaration":
        # const fs = require('fs')
        for child in node.children:
            if child.type == "variable_declarator":
                name_node = child.child_by_field_name("name")
                value_node = child.child_by_field_name("value")
                if name_node and value_node and value_node.type == "call_expression":
                    func = value_node.child_by_field_name("function")
                    if func and func.type == "identifier" and func.text == b"require":
                        args = value_node.child_by_field_name("arguments")
                        if args:
                            for arg in args.children:
                                if arg.type == "string":
                                    source = arg.text.decode("utf-8").strip("'\"")
                                    if not source.startswith(".") and not source.startswith("/"):
                                        package = source.split("/")[0]
                                        if package.startswith("@") and "/" in source:
                                            parts = source.split("/")
                                            package = f"{parts[0]}/{parts[1]}" if len(parts) > 1 else parts[0]
                                        imports[name_node.text.decode("utf-8")] = package
                                    break

    for child in node.children:
        _collect_js_imports(child, imports)


def _is_external_receiver(receiver: str | None, imports: dict[str, str], lang: str) -> bool:
    """Check if a receiver refers to an external (imported) module.

    Returns False for instance method receivers (self, cls, this).
    Returns True if the root name of the receiver is in the imports dict.

    Args:
        receiver: The receiver name (e.g., "requests", "os.path", "self")
        imports: Dict mapping local names to their source modules
        lang: Programming language

    Examples:
        - receiver="self" → False (instance method)
        - receiver="this" → False (instance method)
        - receiver="requests", imports={"requests": "requests"} → True
        - receiver="os.path", imports={"os": "os"} → True
        - receiver="helper", imports={} → False
    """
    if receiver is None:
        return False

    # Instance method receivers are not external
    if lang == "python" and receiver in ("self", "cls"):
        return False
    if lang in JS_LANGS and receiver in ("this",):
        return False
    if lang in LANG_SPECS and receiver.split(".")[0] in LANG_SPECS[lang]["instance"]:
        return False
    # Also handle this.something - it's still an instance method
    if receiver.startswith("this."):
        return False
    if receiver.startswith("self.") or receiver.startswith("cls."):
        return False

    # Extract root name from receiver (os.path → os)
    root_name = receiver.split(".")[0]

    return root_name in imports


def _is_imported_function(call_name: str, receiver: str | None, imports: dict[str, str]) -> bool:
    """Check if a direct function call is to an imported external function.

    This handles cases like `from subprocess import run` where `run()` is called
    without a receiver.

    Args:
        call_name: The function name being called (e.g., "run")
        receiver: The receiver (if any) - if present, this function returns False
        imports: Dict mapping local names to their source modules

    Returns:
        True if the call is to an imported function (no receiver, name in imports)

    Examples:
        - call_name="run", receiver=None, imports={"run": "subprocess"} → True
        - call_name="run", receiver="self", imports={"run": "subprocess"} → False
        - call_name="helper", receiver=None, imports={} → False
    """
    # If there's a receiver, this isn't a direct imported function call
    if receiver is not None:
        return False

    return call_name in imports


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
