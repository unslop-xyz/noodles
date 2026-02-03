"""Filesystem manifest generation utilities."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from pathspec import PathSpec

from .unslop_dir_management import create_run_dir, list_run_dirs


def snapshot(root: Path) -> dict[str, dict[str, object]]:
    """Return the current metadata snapshot for ``root``."""
    return _collect_file_metadata(root)


def generate_manifest(root: Path) -> Path:
    """
    Scan ``root`` and store a manifest inside ``.unslop``.

    Returns the path to the manifest file that was written.
    """
    return write_manifest(root, snapshot(root))


def write_manifest(root: Path, entries: dict[str, dict[str, object]]) -> Path:
    """Persist ``entries`` to a run-scoped manifest file."""
    workspace = create_run_dir(root)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    manifest = {
        "generated_at": timestamp,
        "files": entries,
    }
    manifest_path = workspace / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path


def load_latest_manifest(root: Path) -> tuple[Path, dict[str, dict[str, object]]] | None:
    """Return the most recent manifest and its data if present."""
    for run_dir in reversed(list_run_dirs(root)):
        manifest_path = run_dir / "manifest.json"
        if not manifest_path.is_file():
            continue
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        return manifest_path, data.get("files", {})
    return None


def find_previous_manifest(root: Path, output_dir: Path) -> Path | None:
    """Return the manifest path from the run before ``output_dir``."""
    run_dirs = list_run_dirs(root)
    if not run_dirs:
        return None
    output_dir_resolved = output_dir.resolve()
    previous_dir = None
    for idx, run_dir in enumerate(run_dirs):
        if run_dir.resolve() == output_dir_resolved:
            if idx > 0:
                previous_dir = run_dirs[idx - 1]
            break
    else:
        if len(run_dirs) >= 2:
            previous_dir = run_dirs[-2]
    if previous_dir is None:
        return None
    manifest_path = previous_dir / "manifest.json"
    return manifest_path if manifest_path.is_file() else None


def read_manifest_entries(
    manifest_path: Path,
) -> dict[str, dict[str, object]] | None:
    """Read a manifest file and return its entries mapping."""
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    entries = payload.get("files", {})
    if not isinstance(entries, dict):
        return None
    return entries


def summarize_changes(
    previous: dict[str, dict[str, object]], current: dict[str, dict[str, object]]
) -> dict[str, list[str]]:
    """Return a summary of additions, deletions, and modifications."""
    previous_keys = set(previous)
    current_keys = set(current)

    added = sorted(current_keys - previous_keys)
    deleted = sorted(previous_keys - current_keys)
    modified = sorted(
        path
        for path in previous_keys & current_keys
        if previous[path].get("hash") != current[path].get("hash")
    )

    return {"added": added, "deleted": deleted, "modified": modified}


def _collect_file_metadata(root: Path) -> dict[str, dict[str, object]]:
    """Build the mapping of relative path -> metadata."""
    gitignore = _compile_gitignore(root)
    metadata: dict[str, dict[str, object]] = {}

    for file_path in _iter_files(root, gitignore):
        relative = file_path.relative_to(root).as_posix()
        stats = file_path.stat()
        metadata[relative] = {
            "hash": _hash_file(file_path),
            "size": stats.st_size,
            "mtime": datetime.fromtimestamp(stats.st_mtime, tz=timezone.utc).isoformat(),
        }

    return metadata


def _iter_files(root: Path, gitignore: PathSpec | None) -> Iterator[Path]:
    """Yield eligible files beneath ``root``."""
    for dirpath, dirnames, filenames in os.walk(root):
        current_dir = Path(dirpath)
        relative_dir = current_dir.relative_to(root)
        if str(relative_dir) == ".":
            relative_dir = Path()

        dirnames[:] = [
            name
            for name in dirnames
            if not _should_skip(relative_dir / name, gitignore, is_dir=True)
        ]

        for filename in filenames:
            relative_file = relative_dir / filename
            if _should_skip(relative_file, gitignore, is_dir=False):
                continue
            yield current_dir / filename


def _should_skip(relative: Path, gitignore: PathSpec | None, *, is_dir: bool) -> bool:
    """Return True if the relative path should be excluded."""
    if _is_hidden(relative):
        return True
    if _is_dependency_or_build(relative):
        return True
    if not is_dir and _is_asset(relative):
        return True
    if not is_dir and _is_non_logic(relative):
        return True
    if gitignore is None:
        return False
    rel_str = relative.as_posix()
    if is_dir and rel_str:
        rel_str = f"{rel_str}/"
    return gitignore.match_file(rel_str)


def _is_hidden(relative: Path) -> bool:
    """Return True if any component of the relative path is hidden."""
    for part in relative.parts:
        if part in {"", "."}:
            continue
        if part.startswith("."):
            return True
    return False


def _is_dependency_or_build(relative: Path) -> bool:
    """Return True if the path is inside a dependency or build directory."""
    blacklisted = {
        "node_modules",
        "dist",
        "build",
        ".turbo",
        ".next",
        "out",
        "venv",
        ".venv",
        "target",
        ".unslop",
    }
    for part in relative.parts:
        if part in blacklisted:
            return True
    return False


def _is_asset(relative: Path) -> bool:
    """Return True if the path looks like a non-logic asset file."""
    return relative.suffix.lower() in {
        ".css",
        ".scss",
        ".sass",
        ".less",
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".svg",
        ".webp",
        ".ico",
        ".mp3",
        ".wav",
        ".ogg",
        ".mp4",
        ".mov",
        ".avi",
        ".webm",
        ".pdf",
        ".ttf",
        ".otf",
        ".woff",
        ".woff2",
    }


def _is_non_logic(relative: Path) -> bool:
    """Return True for files that are not part of core logic."""
    filename = relative.name
    lower_name = filename.lower()
    if any(part in {"tests", "__tests__", "__mocks__"} for part in relative.parts):
        return True
    if lower_name.startswith("test_") or lower_name.endswith("_test.py"):
        return True
    if ".spec." in lower_name or ".test." in lower_name:
        return True
    if relative.suffix.lower() in {".css", ".scss", ".sass", ".less"}:
        return True
    return False


def _compile_gitignore(root: Path) -> PathSpec | None:
    """Compile the gitignore patterns if a file exists."""
    gitignore_path = root / ".gitignore"
    if not gitignore_path.is_file():
        return None
    lines = gitignore_path.read_text(encoding="utf-8").splitlines()
    return PathSpec.from_lines("gitwildmatch", lines)


def _hash_file(path: Path) -> str:
    """Return the SHA-256 digest for the provided file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()
