"""Fill node names and descriptions from the code's own docstrings.

Runs before the LLM enrichment, offline and for free. Two reasons it matters:

1. **Without an API key the graph was unusable as documentation** — every node
   arrived with an empty description and the viewer's tooltips showed nothing,
   even for codebases that document themselves thoroughly.
2. **A docstring beats a guess.** When the author wrote "Calcula la 2PACF
   analítica (He+2022) para una rebanada de catálogo", no summariser is going to
   improve on that. The LLM step still runs afterwards and overwrites whatever it
   describes better, so nothing is lost — this is a floor, not a ceiling.

Python only: it is the language whose docstrings are addressable from the AST.
Other languages fall through untouched.
"""

from __future__ import annotations

import ast
from pathlib import Path


def _first_sentence(doc: str, limit: int = 160) -> str:
    """First meaningful line of a docstring, trimmed to something tooltip-sized."""
    for raw in doc.strip().splitlines():
        line = raw.strip()
        if line:
            return line[:limit]
    return ""


def _docstrings_for_file(path: Path) -> dict[str, str]:
    """Map qualified function name -> first docstring line, for one Python file.

    Qualified names match the call-graph convention: bare name for a top-level
    function, "Class.method" for a method.
    """
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError, UnicodeDecodeError):
        return {}

    out: dict[str, str] = {}

    module_doc = ast.get_docstring(tree)
    if module_doc:
        out["<module>"] = _first_sentence(module_doc)

    def visit(node: ast.AST, prefix: str = "") -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ClassDef):
                visit(child, f"{prefix}{child.name}.")
            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                doc = ast.get_docstring(child)
                if doc:
                    out[f"{prefix}{child.name}"] = _first_sentence(doc)
                # nested defs keep the enclosing prefix, matching the builder
                visit(child, prefix)

    visit(tree)
    return out


def enrich_from_docstrings(nodes: list[dict], repo_path: str | Path) -> int:
    """Fill empty `name`/`description` on nodes from their own docstrings.

    Only fills what is empty, so a previous (or later) enrichment always wins.

    Args:
        nodes: Call-graph nodes; ids look like "path/to/file.py::qualified_name".
        repo_path: Repository root the ids are relative to.

    Returns:
        How many nodes got a description.
    """
    repo = Path(repo_path)
    cache: dict[str, dict[str, str]] = {}
    filled = 0

    for node in nodes:
        node_id = node.get("id", "")
        if "::" not in node_id or node.get("description"):
            continue

        rel_path, _, qualified = node_id.partition("::")
        if not rel_path.endswith(".py"):
            continue

        if rel_path not in cache:
            cache[rel_path] = _docstrings_for_file(repo / rel_path)

        # Module-scope pseudo-nodes describe the script itself, so their
        # description is the file's module docstring.
        key = "<module>" if qualified == "<module>" else qualified
        doc = cache[rel_path].get(key)
        if not doc:
            continue

        node["description"] = doc
        if not node.get("name"):
            node["name"] = qualified.split(".")[-1]
        filled += 1

    return filled
