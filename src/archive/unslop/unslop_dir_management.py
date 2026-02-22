"""Helpers for organizing .unslop run folders."""

from __future__ import annotations

import re
import secrets
from datetime import datetime, timezone
from pathlib import Path

_RUN_DIR_RE = re.compile(r"^\d{8}T\d{6}Z-\d{4}$")


def ensure_unslop_dir(root: Path) -> Path:
    """Ensure the `.unslop` directory exists and return its path."""
    unslop_dir = root / ".unslop"
    if unslop_dir.exists() and not unslop_dir.is_dir():
        unslop_dir.unlink()
    unslop_dir.mkdir(exist_ok=True)
    return unslop_dir


def create_run_dir(root: Path) -> Path:
    """Create a new run folder using timestamp + random suffix."""
    unslop_dir = ensure_unslop_dir(root)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    suffix = f"{secrets.randbelow(10000):04d}"
    run_dir = unslop_dir / f"{timestamp}-{suffix}"
    run_dir.mkdir()
    return run_dir


def list_run_dirs(root: Path) -> list[Path]:
    """Return sorted run directories within `.unslop`."""
    unslop_dir = root / ".unslop"
    if not unslop_dir.is_dir():
        return []
    runs = [
        path
        for path in unslop_dir.iterdir()
        if path.is_dir() and _RUN_DIR_RE.match(path.name)
    ]
    return sorted(runs, key=lambda path: path.name)


def latest_run_dir(root: Path) -> Path | None:
    """Return the most recent run directory, if any."""
    runs = list_run_dirs(root)
    return runs[-1] if runs else None
