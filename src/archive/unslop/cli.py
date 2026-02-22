"""Command-line interface for unslop."""

from __future__ import annotations

import argparse
import logging
import os
import multiprocessing
from pathlib import Path
from typing import Dict, Iterator, List

from dotenv import load_dotenv

load_dotenv(".env.local", override=True)
load_dotenv()

from .diagram import find_d2_bin, generate_diagram, latest_diagram, render_diagram_image
from .manifest import load_latest_manifest, snapshot, summarize_changes, write_manifest
from .overlay import launch_overlay

logger = logging.getLogger(__name__)
_LOG_FORMAT = "%(levelname)s:%(message)s"


def main(argv: List[str] | None = None) -> int:
    """Entry point for the ``unslop`` CLI."""
    _configure_logging()
    parser = argparse.ArgumentParser(
        prog="unslop", description="Utility commands for the unslop project."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser(
        "run",
        help="Compare against the latest manifest and persist changes.",
    )
    run_parser.add_argument("--headless", action="store_true", help="Run without UI overlay.")
    run_parser.set_defaults(func=_run_command)

    args = parser.parse_args(argv)
    result = args.func(args)
    return 0 if result is None else int(result)


def _configure_logging() -> None:
    level_name = os.getenv("UNSLOP_LOG_LEVEL", "INFO").upper()
    level = logging.getLevelName(level_name)
    if not isinstance(level, int):
        level = logging.INFO
    logging.basicConfig(level=level, format=_LOG_FORMAT)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


class _OverlayLogHandler(logging.Handler):
    def __init__(self, updates_queue: multiprocessing.Queue) -> None:
        super().__init__()
        self._updates_queue = updates_queue

    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = self.format(record)
            payload = {"type": "log", "message": message}
            self._updates_queue.put_nowait(payload)
            api_key_status = record.__dict__.get("unslop_api_key_status")
            if api_key_status:
                self._updates_queue.put_nowait(
                    {"type": "api_key_status", "status": api_key_status}
                )
            loading_tag = record.__dict__.get("unslop_loading")
            if loading_tag:
                loading_map = {
                    "overview_start": ("overview", True),
                    "overview_stop": ("overview", False),
                    "node_start": ("node", True),
                    "node_stop": ("node", False),
                }
                scope_state = loading_map.get(loading_tag)
                if scope_state:
                    scope, active = scope_state
                    self._updates_queue.put_nowait(
                        {"type": "loading", "scope": scope, "active": active}
                    )
        except Exception:
            pass


def _attach_overlay_logger(updates_queue: multiprocessing.Queue) -> None:
    root_logger = logging.getLogger()
    for handler in root_logger.handlers:
        if isinstance(handler, _OverlayLogHandler):
            return
    handler = _OverlayLogHandler(updates_queue)
    handler.setFormatter(logging.Formatter(_LOG_FORMAT))
    root_logger.addHandler(handler)


def _run_command(args: argparse.Namespace) -> int | None:
    """Handle the ``unslop run`` command."""
    if not _check_requirements():
        return 1
    roots, selection_queue, updates_queue = _launch_overlay_and_wait(
        headless=args.headless
    )
    if updates_queue is not None:
        _attach_overlay_logger(updates_queue)

    processed_any = False
    for root_path in roots:
        processed_any = True
        _process_folder(root_path, updates_queue)

    if not processed_any:
        logger.info("No folder selected; exiting.")
    _shutdown_queues(selection_queue, updates_queue)
    return


def _check_requirements() -> bool:
    """Ensure required dependencies are present before running."""
    missing = False
    openai_key = os.getenv("OPENAI_API_KEY") or os.getenv("UNSLOP_OPENAI_API_KEY")
    gemini_key = os.getenv("GEMINI_API_KEY")

    if not openai_key and not gemini_key:
        logger.warning(
            "Missing LLM API keys. Please set OPENAI_API_KEY or GEMINI_API_KEY."
        )

    if not find_d2_bin():
        logger.warning("Missing d2 CLI. Run: brew install d2")
        missing = True

    if missing:
        logger.warning("Exiting due to missing requirements.")
        return False
    return True

# Step 1
def _launch_overlay_and_wait(
    headless: bool = False,
) -> tuple[
    Iterator[dict[str, str] | str] | list[str],
    multiprocessing.Queue | None,
    multiprocessing.Queue | None,
]:
    """Step 1: open overlay and return an iterator of roots to process."""
    is_headless = headless or (os.getenv("UNSLOP_HEADLESS") == "1")
    
    if is_headless:
        overlay_queues = None
    else:
        logger.info("Launch overlay and wait for folder selection.")
        overlay_queues = launch_overlay()

    selection_queue: multiprocessing.Queue[dict | str | None] | None = None
    updates_queue: multiprocessing.Queue[dict | None] | None = None
    if overlay_queues is None:
        logger.info("Overlay disabled (headless mode); defaulting to current directory.")
        roots: Iterator[str] | list[str] = [str(Path.cwd())]
    else:
        selection_queue, updates_queue = overlay_queues
        logger.info(
            "Waiting for folder selection in overlay... (close the overlay window to finish)",
        )
        roots = _iter_selections(selection_queue)
    return roots, selection_queue, updates_queue

# Step 2
def _process_folder(
    selection: dict[str, str] | str, updates_queue: multiprocessing.Queue | None
) -> None:
    """Step 2: react to the selected folder."""
    action = "select"
    root_path = selection
    overview_model: str | None = None
    if isinstance(selection, dict):
        action = selection.get("action") or "select"
        if action == "set_key":
            _apply_openai_key(selection.get("key"))
            return
        root_path = selection.get("path") or ""
        overview_model = selection.get("model") or None
    if not isinstance(root_path, str) or not root_path:
        logger.info("No folder selected; skipping.")
        return

    logger.info("Folder selected (%s) -> %s", action, root_path)
    root = Path(root_path)

    if action == "update":
        run_dir = _update_manifest(root, updates_queue, verbose=True, overview_model=overview_model)
        if run_dir:
            _update_diagram(root, updates_queue, output_dir=run_dir, overview_model=overview_model)
        return
    if action == "rerun":
        run_dir = _create_manifest(root, updates_queue, verbose=True, overview_model=overview_model)
        _generate_diagram(root, updates_queue, output_dir=run_dir, overview_model=overview_model)
        return

    existing = latest_diagram(root)
    if existing:
        logger.info("Using existing diagram on selection.")
        if updates_queue is not None:
            _send_diagram(updates_queue, existing, render_image=False)
        return
    run_dir = _create_manifest(root, updates_queue, verbose=True, overview_model=overview_model)
    _generate_diagram(root, updates_queue, output_dir=run_dir, overview_model=overview_model)


def _apply_openai_key(value: str | None) -> None:
    key = (value or "").strip()
    if not key:
        os.environ.pop("OPENAI_API_KEY", None)
        os.environ.pop("UNSLOP_OPENAI_API_KEY", None)
        logger.info("Cleared OpenAI API key from overlay.")
        return
    os.environ["OPENAI_API_KEY"] = key
    os.environ["UNSLOP_OPENAI_API_KEY"] = key
    logger.info("Updated OpenAI API key from overlay.")

# Step 3
def _create_manifest(
    root: Path,
    updates_queue: multiprocessing.Queue | None,
    *,
    verbose: bool = False,
    overview_model: str | None = None,
) -> Path:
    """Steps 3: create manifest"""
    if verbose:
        logger.info("Processing folder: %s", root)

    logger.info("Create manifest in %s.", root / ".unslop")
    previous_entries: Dict[str, Dict[str, object]] = {}
    current_entries = snapshot(root)
    summary = summarize_changes(previous_entries, current_entries)

    if _has_changes(summary):
        _print_summary(summary)
    else:
        logger.info("Initializing manifest (no files to track).")

    manifest_path = write_manifest(root, current_entries)
    run_dir = manifest_path.parent
    logger.info("Wrote manifest: %s", manifest_path)
    return run_dir

# Step 3b
def _update_manifest(
    root: Path,
    updates_queue: multiprocessing.Queue | None,
    *,
    verbose: bool = False,
    overview_model: str | None = None,
) -> Path | None:
    """Step 3: update manifest and diagram."""
    if verbose:
        logger.info("Processing folder: %s", root)

    existing_diagram = latest_diagram(root)
    if existing_diagram is None:
        return _create_manifest(root, updates_queue, verbose=verbose, overview_model=overview_model)

    logger.info("Load latest manifest in %s.", root / ".unslop")
    latest = load_latest_manifest(root)
    previous_entries: Dict[str, Dict[str, object]] = {} if latest is None else latest[1]
    current_entries = snapshot(root)
    summary = summarize_changes(previous_entries, current_entries)

    if not _has_changes(summary):
        logger.info("No change detected.")
        if existing_diagram and updates_queue is not None:
            _send_diagram(updates_queue, existing_diagram, render_image=False)
        return None

    _print_summary(summary)
    manifest_path = write_manifest(root, current_entries)
    run_dir = manifest_path.parent
    logger.info("Wrote manifest: %s", manifest_path)
    return run_dir

# Step 4
def _generate_diagram(
    root: Path,
    updates_queue: multiprocessing.Queue,
    *,
    output_dir: Path | None,
    overview_model: str | None = None,
) -> None:
    logger.info("Ask LLM to generate D2 diagram from manifest.")
    diagram_path = generate_diagram(
        root,
        output_dir=output_dir,
        is_update=False,
        overview_model=overview_model,
    )
    if diagram_path is None:
        diagram_path = latest_diagram(root)
        if diagram_path:
            logger.info("Reusing existing diagram at %s", diagram_path)
        elif diagram_path is None:
            # No diagram generated and no existing diagram
            _send_warning(updates_queue,
                "Could not generate diagram. Please set OPENAI_API_KEY or UNSLOP_OPENAI_API_KEY environment variable.")
            return
    _send_diagram(updates_queue, diagram_path, render_image=False)

# Step 4b
def _update_diagram(
    root: Path,
    updates_queue: multiprocessing.Queue,
    *,
    output_dir: Path | None,
    overview_model: str | None = None,
) -> None:
    logger.info("Ask LLM to generate D2 diagram from manifest.")
    diagram_path = generate_diagram(
        root,
        output_dir=output_dir,
        is_update=True,
        overview_model=overview_model,
    )
    if diagram_path is None:
        diagram_path = latest_diagram(root)
        if diagram_path:
            logger.info("Reusing existing diagram at %s", diagram_path)
        elif diagram_path is None:
            # No diagram generated and no existing diagram
            _send_warning(updates_queue,
                "Could not generate diagram. Please set OPENAI_API_KEY or UNSLOP_OPENAI_API_KEY environment variable.")
            return
    _send_diagram(updates_queue, diagram_path, render_image=False)

# Step 5
def _send_diagram(
    updates_queue: multiprocessing.Queue,
    diagram_path: Path | None,
    *,
    render_image: bool,
) -> None:
    logger.info("Send diagram to overlay for display.")
    image_path = render_diagram_image(diagram_path) if render_image and diagram_path else None
    _send_diagram_to_overlay(updates_queue, diagram_path, image_path)


def _shutdown_queues(
    selection_queue: multiprocessing.Queue | None, updates_queue: multiprocessing.Queue | None
) -> None:
    """Best-effort queue cleanup."""
    if updates_queue is not None:
        try:
            updates_queue.put_nowait(None)
        except Exception:
            pass
    if selection_queue is not None:
        try:
            selection_queue.close()
        except Exception:
            pass


def _iter_selections(selection_queue: multiprocessing.Queue) -> Iterator[dict[str, str] | str]:
    """Yield folder selections until the overlay sends a sentinel or closes."""
    while True:
        try:
            selection = selection_queue.get()
        except KeyboardInterrupt:
            raise
        except Exception:
            break
        if selection is None:
            break
        yield selection


def _has_changes(summary: Dict[str, List[str]]) -> bool:
    """Return True if there are any changes listed in the summary."""
    return any(summary.values())


def _print_summary(summary: Dict[str, List[str]]) -> None:
    """Print change details for the given summary."""
    if summary["added"]:
        logger.info("New files:")
        for path in summary["added"]:
            logger.info("+ added file: %s", path)
    if summary["deleted"]:
        logger.info("Deleted files:")
        for path in summary["deleted"]:
            logger.info("- deleted file: %s", path)
    if summary["modified"]:
        logger.info("Modified files:")
        for path in summary["modified"]:
            logger.info("~ modified file: %s", path)


def _send_diagram_to_overlay(
    updates_queue: multiprocessing.Queue | None,
    diagram_path: Path | None,
    image_path: Path | None,
) -> None:
    """Send generated D2 to the overlay for display, if possible."""
    if updates_queue is None or diagram_path is None:
        return
    try:
        content = diagram_path.read_text(encoding="utf-8")
    except Exception:
        return
    payload = {
        "type": "diagram",
        "path": str(diagram_path),
        "content": content,
        "image": str(image_path) if image_path else None,
    }
    logger.info("Sending diagram to overlay: %s", diagram_path)
    try:
        updates_queue.put_nowait(payload)
    except Exception:
        try:
            updates_queue.put(payload)
        except Exception:
            pass


def _send_warning(
    updates_queue: multiprocessing.Queue | None,
    message: str,
) -> None:
    """Send a warning message to the overlay for display."""
    if updates_queue is None:
        return
    payload = {
        "type": "warning",
        "message": message,
    }
    logger.warning("Sending warning to overlay: %s", message)
    try:
        updates_queue.put_nowait(payload)
    except Exception:
        try:
            updates_queue.put(payload)
        except Exception:
            pass
