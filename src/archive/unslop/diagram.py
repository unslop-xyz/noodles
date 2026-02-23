"""Diagram generation pipeline for selected folders."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import subprocess
import threading
from pathlib import Path

from .code2diagram import (
    generate_node_schema_and_diagram,
    generate_overview_schema,
    get_overview_d2_diagram,
    update_overview_schema
)
from .manifest import find_previous_manifest, read_manifest_entries, summarize_changes
from .unslop_dir_management import create_run_dir, list_run_dirs

logger = logging.getLogger(__name__)


def generate_diagram(
    folder: Path,
    *,
    output_dir: Path | None = None,
    is_update: bool = False,
    overview_model: str | None = None,
) -> Path | None:
    """Generate or update a D2 diagram for a folder."""
    folder = folder.resolve()

    logger.info(
        "Preparing overview diagram for %s ...",
        folder,
        extra={"unslop_loading": "overview_start"},
    )
    if output_dir is None:
        output_dir = create_run_dir(folder)
    output_dir.mkdir(parents=True, exist_ok=True)

    diagram_path = (
        _update_overview_diagram(folder, output_dir, overview_model=overview_model)
        if is_update
        else _generate_overview_diagram(folder, output_dir, overview_model=overview_model)
    )
    if diagram_path is None:
        return None
    logger.info(
        "Generated D2 diagram at %s",
        diagram_path,
        extra={"unslop_loading": "overview_stop"},
    )
    return diagram_path


def latest_diagram(folder: Path) -> Path | None:
    """Return the most recent stored overview D2 diagram for the folder, if any."""
    for run_dir in reversed(list_run_dirs(folder)):
        overview = run_dir / "overview.d2"
        if overview.is_file():
            return overview
        diagrams = sorted(run_dir.glob("*.d2"))
        if diagrams:
            return diagrams[-1]
    return None


def _generate_overview_diagram(
    folder: Path,
    output_dir: Path,
    *,
    overview_model: str | None = None,
) -> Path | None:
    src_root = folder / "src"
    source_dir = src_root if src_root.is_dir() else folder
    combined_path = output_dir / "combined_files.txt"
    schema_path = output_dir / "overview.json"
    overview_path = output_dir / "overview.d2"

    allowed_files = _load_all_files(folder, source_dir, output_dir)
    if allowed_files is not None:
        allowed_files = {path: "added" for path in allowed_files}
    combined = combine_src_files(
        str(source_dir),
        str(combined_path),
        allowed_files=allowed_files,
    )

    try:
        overview_schema = generate_overview_schema(
            combined,
            model=overview_model or "gpt-4.1",
        )
    except Exception as exc:
        logger.exception("OpenAI request failed: %s", exc)
        return None

    schema_path.write_text(overview_schema, encoding="utf-8")

    get_overview_d2_diagram(schema_path, output_dir=output_dir)
    _start_node_diagrams(schema_path, source_dir, output_dir, previous_run_dir=None)

    if overview_path.is_file():
        return overview_path
    return None


def _update_overview_diagram(
    folder: Path,
    output_dir: Path,
    *,
    overview_model: str | None = None,
) -> Path | None:
    src_root = folder / "src"
    source_dir = src_root if src_root.is_dir() else folder
    combined_path = output_dir / "combined_files.txt"
    schema_path = output_dir / "overview.json"
    overview_path = output_dir / "overview.d2"

    allowed_files = _load_changed_files(folder, source_dir, output_dir)
    if not allowed_files:
        logger.info("No source changes detected; skipping overview update.")
        return None

    combined = combine_src_files(
        str(source_dir),
        str(combined_path),
        allowed_files=allowed_files,
    )

    previous_schema = ""
    previous_manifest = find_previous_manifest(folder, output_dir)
    previous_run_dir = previous_manifest.parent if previous_manifest else None
    if previous_manifest is not None:
        previous_schema_path = previous_manifest.parent / "overview.json"
        try:
            previous_schema = previous_schema_path.read_text(encoding="utf-8")
        except Exception:
            previous_schema = ""

    try:
        overview_schema = update_overview_schema(
            combined,
            model=overview_model or "gpt-4.1",
            previous_schema=previous_schema,
        )
    except Exception as exc:
        logger.exception("OpenAI request failed: %s", exc)
        return None

    schema_path.write_text(overview_schema, encoding="utf-8")

    get_overview_d2_diagram(schema_path, output_dir=output_dir)
    _start_node_diagrams(schema_path, source_dir, output_dir, previous_run_dir)

    if overview_path.is_file():
        return overview_path
    return None


def combine_src_files(
    src_dir="src",
    output_path="combined_files.txt",
    *,
    allowed_files: dict[str, str] | None = None,
):
    """Combine files under src_dir into one text file with line numbers."""
    src_path = Path(src_dir)
    if not src_path.exists():
        raise FileNotFoundError(f"Missing source directory: {src_path}")

    files = sorted(p for p in src_path.rglob("*") if p.is_file())
    parts = []

    for file_path in files:
        rel_path = file_path.relative_to(src_path)
        rel_path_str = rel_path.as_posix()
        if allowed_files is not None and rel_path_str not in allowed_files:
            continue
        if any(part.startswith(".") for part in rel_path.parts):
            continue
        status = allowed_files.get(rel_path_str) if allowed_files else None
        label = f" ({status})" if status else ""
        parts.append(f"### FILE: {rel_path_str}{label}")
        try:
            text = file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            # Fallback for non-UTF8 sources.
            text = file_path.read_text(encoding="latin-1")
        for i, line in enumerate(text.splitlines(), 1):
            parts.append(f"[{i}] {line}")
        parts.append("")

    if allowed_files:
        deleted_files = sorted(
            path for path, status in allowed_files.items() if status == "deleted"
        )
        for rel_path_str in deleted_files:
            parts.append(f"### FILE: {rel_path_str} (deleted)")
            parts.append("")

    combined = "\n".join(parts).rstrip() + "\n"
    if output_path:
        Path(output_path).write_text(combined, encoding="utf-8")
    return combined


def _load_all_files(
    root: Path, source_dir: Path, output_dir: Path
) -> set[str] | None:
    manifest_path = output_dir / "manifest.json"
    if not manifest_path.is_file():
        return None
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        logger.exception("Failed to read manifest: %s", manifest_path)
        return None
    entries = payload.get("files", {})
    if not isinstance(entries, dict):
        return None
    prefix = ""
    try:
        relative_source = source_dir.relative_to(root)
        prefix = relative_source.as_posix()
    except ValueError:
        return None
    if not prefix or prefix == ".":
        return set(entries.keys())
    prefix = prefix.rstrip("/") + "/"
    allowed = {
        key[len(prefix) :]
        for key in entries.keys()
        if isinstance(key, str) and key.startswith(prefix)
    }
    return allowed


def _load_changed_files(
    root: Path, source_dir: Path, output_dir: Path
) -> dict[str, str] | None:
    current_manifest = output_dir / "manifest.json"
    if not current_manifest.is_file():
        return None

    previous_manifest = find_previous_manifest(root, output_dir)
    if previous_manifest is None:
        return None

    current_entries = read_manifest_entries(current_manifest)
    previous_entries = read_manifest_entries(previous_manifest)
    if current_entries is None or previous_entries is None:
        return None

    summary = summarize_changes(previous_entries, current_entries)
    changed_paths = summary["added"] + summary["modified"] + summary["deleted"]
    if not changed_paths:
        return {}

    prefix = ""
    try:
        relative_source = source_dir.relative_to(root)
        prefix = relative_source.as_posix()
    except ValueError:
        return None

    if prefix and prefix != ".":
        prefix = prefix.rstrip("/") + "/"
    else:
        prefix = ""

    changed_files: dict[str, str] = {}
    for path_str in changed_paths:
        if not isinstance(path_str, str):
            continue
        if prefix and not path_str.startswith(prefix):
            continue
        relative_path = path_str[len(prefix) :] if prefix else path_str
        if not relative_path:
            continue
        if relative_path in summary["deleted"]:
            changed_files[relative_path] = "deleted"
            continue
        file_path = source_dir / relative_path
        if not file_path.is_file():
            continue
        if relative_path in summary["added"]:
            changed_files[relative_path] = "added"
        elif relative_path in summary["modified"]:
            changed_files[relative_path] = "updated"
        else:
            changed_files[relative_path] = "updated"

    return changed_files


def _start_node_diagrams(
    schema_path: Path,
    source_dir: Path,
    output_dir: Path,
    previous_run_dir: Path | None = None,
) -> None:
    logger.info(
        "Preparing per-node diagrams in background.",
        extra={"unslop_loading": "node_start"},
    )
    def run() -> None:
        try:
            include_nodes = _prepare_node_diagram_inputs(
                schema_path, output_dir, previous_run_dir
            )
            if include_nodes is not None and not include_nodes:
                logger.info("No node diagrams need regeneration.")
                return
            asyncio.run(
                generate_node_schema_and_diagram(
                    schema_path,
                    src_dir=str(source_dir),
                    output_dir=output_dir,
                    include_node_ids=include_nodes,
                )
            )
        except Exception as exc:
            logger.exception("Node diagram generation failed: %s", exc)
        finally:
            logger.info(
                "Per-node diagram generation complete.",
                extra={"unslop_loading": "node_stop"},
            )

    thread = threading.Thread(
        target=run, name="unslop-node-diagrams", daemon=True
    )
    thread.start()


def _prepare_node_diagram_inputs(
    schema_path: Path,
    output_dir: Path,
    previous_run_dir: Path | None,
) -> set[str] | None:
    if previous_run_dir is None:
        return None
    try:
        payload = json.loads(schema_path.read_text(encoding="utf-8"))
    except Exception:
        return None

    nodes = payload.get("nodes", [])
    if not isinstance(nodes, list):
        return None

    unchanged_ids = {
        node.get("id")
        for node in nodes
        if node.get("status") == "unchanged" and node.get("id")
    }
    if not unchanged_ids:
        return None

    include_nodes: set[str] = set()
    for node in nodes:
        node_id = node.get("id")
        if not node_id:
            continue
        if node_id not in unchanged_ids:
            include_nodes.add(node_id)
            continue
        if not _reuse_previous_node_diagram(node_id, previous_run_dir, output_dir):
            include_nodes.add(node_id)

    return include_nodes


def _reuse_previous_node_diagram(
    node_id: str,
    previous_run_dir: Path,
    output_dir: Path,
) -> bool:
    copied = False
    for suffix in ("json", "d2"):
        src = previous_run_dir / f"{node_id}.{suffix}"
        if not src.is_file():
            continue
        dest = output_dir / src.name
        try:
            shutil.copy2(src, dest)
            copied = True
        except Exception:
            continue
    if copied:
        logger.info("Reused node diagram: %s", node_id)
    return copied


def render_diagram_image(diagram_path: Path) -> Path | None:
    """
    Render a D2 diagram to a PNG alongside the .d2 file using the d2 CLI.
    """
    image_path = diagram_path.with_suffix(".png")
    try:
        content = diagram_path.read_text(encoding="utf-8")
    except Exception as exc:
        logger.warning("Failed to read diagram for rendering: %s", exc)
        return None

    rendered = _render_to_png(content, image_path)
    if rendered is None:
        return None
    try:
        logger.info("Rendered diagram image at %s", image_path)
        return image_path
    except Exception:
        return image_path


def _render_to_png(content: str, output_path: Path) -> Path | None:
    """
    Render D2 content to PNG using the d2 CLI. Prints guidance if missing.
    """
    d2_bin = _find_d2_bin()
    if not d2_bin:
        logger.warning(
            "Install the d2 CLI to render diagrams to images (e.g., brew install d2) "
            "or set UNSLOP_D2_BIN to the binary path."
        )
        return None

    try:
        with subprocess.Popen(
            [d2_bin, "-", str(output_path)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        ) as proc:
            stdout, stderr = proc.communicate(content, timeout=30)
            if proc.returncode != 0:
                logger.error("d2 CLI failed (path=%s): %s", d2_bin, stderr or stdout)
                return None
    except Exception as exc:
        logger.exception("d2 CLI render error (path=%s): %s", d2_bin, exc)
        return None
    return output_path


def _find_d2_bin() -> str | None:
    """Locate the d2 CLI binary, considering common install paths."""
    candidate_env = os.environ.get("UNSLOP_D2_BIN")
    candidates = [
        candidate_env,
        "/opt/homebrew/bin/d2",
        "/usr/local/bin/d2",
        "/usr/bin/d2",
        shutil.which("d2"),
    ]
    for cand in candidates:
        if not cand:
            continue
        if _is_valid_d2_bin(cand):
            return cand
    return None


def find_d2_bin() -> str | None:
    """Public helper to locate the d2 CLI binary."""
    return _find_d2_bin()


def _is_valid_d2_bin(path: str) -> bool:
    if not (os.path.isfile(path) and os.access(path, os.X_OK)):
        return False
    try:
        proc = subprocess.run(
            [path, "--version"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if proc.returncode != 0:
            return False
        return True
    except Exception:
        return False
