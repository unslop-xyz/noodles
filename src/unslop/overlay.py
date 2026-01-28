"""Overlay window utilities using pywebview."""

from __future__ import annotations

import base64
import json
import multiprocessing
import os
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
from pathlib import Path


def launch_overlay() -> tuple[multiprocessing.Queue, multiprocessing.Queue] | None:
    """
    Start a small always-on-top overlay window (best effort, non-blocking).

    Returns (selection_queue, updates_queue) where:
    - selection_queue receives the selected folder path (or None)
    - updates_queue accepts messages (e.g., diagrams) to render
    Returns None if the overlay could not be started.
    """
    if os.environ.get("UNSLOP_DISABLE_OVERLAY") == "1":
        return None
    if os.environ.get("PYTEST_CURRENT_TEST"):
        return None
    if not _has_display():
        return None
    if not _has_pywebview():
        return None
    selection_queue: multiprocessing.Queue[dict | str | None] = multiprocessing.Queue()
    updates_queue: multiprocessing.Queue[dict | None] = multiprocessing.Queue()
    try:
        process = multiprocessing.Process(
            target=_run_overlay_process,
            args=(selection_queue, updates_queue),
            name="unslop-overlay",
            daemon=False,
        )
        process.start()
        return selection_queue, updates_queue
    except Exception:
        # Overlay is a convenience; failures here must not break the CLI.
        return None


def _has_display() -> bool:
    """Best-effort guard for headless environments."""
    if sys.platform.startswith("linux"):
        return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    return True


def _has_pywebview() -> bool:
    try:
        import webview  # type: ignore  # noqa: F401

        return True
    except Exception:
        return False


def _run_overlay_process(
    selection_queue: multiprocessing.Queue | None,
    updates_queue: multiprocessing.Queue | None,
) -> None:
    try:
        _show_overlay(selection_queue, updates_queue)
    except Exception:
        _signal_selection(selection_queue, None)
        pass


class _OverlayContext:
    """Holds shared overlay state."""

    def __init__(
        self,
        selection_queue: multiprocessing.Queue | None,
        updates_queue: multiprocessing.Queue | None,
    ) -> None:
        self.selection_queue = selection_queue
        self.updates_queue = updates_queue
        self.tempdir = Path(tempfile.mkdtemp(prefix="unslop_overlay_"))
        self.last_selection: str | None = None
        self.overview_model: str = "gpt-4.1"

    def cleanup(self) -> None:
        try:
            if self.tempdir.exists():
                shutil.rmtree(self.tempdir, ignore_errors=True)
        except Exception:
            pass


def _show_overlay(
    selection_queue: multiprocessing.Queue | None, updates_queue: multiprocessing.Queue | None
) -> None:
    import webview  # type: ignore

    html = _load_overlay_html()
    ctx = _OverlayContext(selection_queue, updates_queue)
    api = _OverlayAPI(None, ctx)
    window = webview.create_window(
        "unslop overlay",
        html=html,
        width=980,
        height=880,
        resizable=True,
        js_api=api,
    )
    api._window = window

    def on_closed() -> None:
        ctx.cleanup()
        _signal_selection(selection_queue, None)
        if updates_queue is not None:
            try:
                updates_queue.put_nowait(None)
            except Exception:
                pass

    window.events.closed += on_closed

    def start_loop() -> None:
        _start_update_loop(window, ctx)

    webview.start(start_loop, None, debug=False)


def _start_update_loop(window, ctx: _OverlayContext) -> None:
    """Poll updates queue and push data into the webview."""

    def poll() -> None:
        while True:
            try:
                message = ctx.updates_queue.get(timeout=0.25) if ctx.updates_queue else None
            except queue.Empty:
                continue
            except Exception:
                break

            if message is None:
                try:
                    window.destroy()
                except Exception:
                    pass
                break

            if not isinstance(message, dict):
                continue

            try:
                if message.get("type") == "diagram":
                    payload = _build_diagram_payload(message, ctx.tempdir)
                    _dispatch_to_window(window, payload)
                elif message.get("type") == "log":
                    payload = {
                        "type": "log",
                        "message": message.get("message", ""),
                    }
                    _dispatch_to_window(window, payload)
                elif message.get("type") == "loading":
                    payload = {
                        "type": "loading",
                        "scope": message.get("scope"),
                        "active": message.get("active"),
                    }
                    _dispatch_to_window(window, payload)
                elif message.get("type") == "api_key_status":
                    payload = {
                        "type": "api_key_status",
                        "status": message.get("status"),
                    }
                    _dispatch_to_window(window, payload)
                elif message.get("type") == "warning":
                    warning_text = message.get("message", "Warning")
                    payload = {
                        "type": "warning",
                        "message": warning_text,
                    }
                    _dispatch_to_window(window, payload)
            except Exception:
                continue

    thread = threading.Thread(target=poll, name="unslop-overlay-updates", daemon=True)
    thread.start()


def _dispatch_to_window(window, payload: dict) -> None:
    try:
        js = f"window.__unslopUpdate({json.dumps(payload)});"
        window.evaluate_js(js)
    except Exception:
        pass


class _OverlayAPI:
    """Python<->JS bridge for pywebview."""

    def __init__(self, window, ctx: _OverlayContext) -> None:
        self._window = window
        self._ctx = ctx

    def choose_folder(self, overview_model: str | None = None) -> dict:
        import webview  # type: ignore

        if self._window is None:
            return {"error": "Window not ready yet."}
        try:
            self._window.hide()  # ensure dialog is visible above always-on-top window
            result = self._window.create_file_dialog(webview.FOLDER_DIALOG)
        except Exception as exc:
            return {"error": f"Failed to open folder dialog: {exc}"}
        finally:
            try:
                self._window.show()
            except Exception:
                pass
        if not result:
            return {"cancelled": True}
        folder = result[0]
        self._ctx.last_selection = folder
        self._ctx.overview_model = _normalize_overview_model(overview_model)
        _signal_selection(
            self._ctx.selection_queue,
            {
                "action": "select",
                "path": folder,
                "model": self._ctx.overview_model,
            },
        )
        return {"selected": folder}

    def rerun(self, overview_model: str | None = None) -> dict:
        if not self._ctx.last_selection:
            return {"error": "No folder selected yet; choose a folder first."}
        self._ctx.overview_model = _normalize_overview_model(overview_model)
        _signal_selection(
            self._ctx.selection_queue,
            {
                "action": "rerun",
                "path": self._ctx.last_selection,
                "model": self._ctx.overview_model,
            },
        )
        return {"selected": self._ctx.last_selection}

    def update(self, overview_model: str | None = None) -> dict:
        if not self._ctx.last_selection:
            return {"error": "No folder selected yet; choose a folder first."}
        self._ctx.overview_model = _normalize_overview_model(overview_model)
        _signal_selection(
            self._ctx.selection_queue,
            {
                "action": "update",
                "path": self._ctx.last_selection,
                "model": self._ctx.overview_model,
            },
        )
        return {"selected": self._ctx.last_selection}

    def get_openai_key(self) -> dict:
        """Return the OpenAI API key, preferring .env if available."""
        env_path = _resolve_env_path(self._ctx)
        file_key = _read_env_key(env_path) if env_path else None
        if file_key:
            return {"key": file_key, "source": "env_file", "path": str(env_path)}
        key = os.environ.get("OPENAI_API_KEY") or os.environ.get("UNSLOP_OPENAI_API_KEY")
        return {"key": key or "", "source": "env_var"}

    def set_openai_key(self, key: str | None) -> dict:
        """Set the OpenAI API key for the current process environment and .env."""
        value = (key or "").strip()
        env_path = _resolve_env_path(self._ctx)
        env_written = False
        if env_path:
            try:
                _write_env_key(env_path, value)
                env_written = True
            except Exception:
                env_written = False
        if not value:
            os.environ.pop("OPENAI_API_KEY", None)
            os.environ.pop("UNSLOP_OPENAI_API_KEY", None)
            _signal_selection(
                self._ctx.selection_queue,
                {"action": "set_key", "key": ""},
            )
            return {
                "key": "",
                "cleared": True,
                "env_path": str(env_path) if env_path else "",
                "env_written": env_written,
            }
        os.environ["OPENAI_API_KEY"] = value
        os.environ["UNSLOP_OPENAI_API_KEY"] = value
        _signal_selection(
            self._ctx.selection_queue,
            {"action": "set_key", "key": value},
        )
        return {
            "key": value,
            "cleared": False,
            "env_path": str(env_path) if env_path else "",
            "env_written": env_written,
        }

    def close_overlay(self) -> dict:
        """Close the overlay window when requested by the frontend."""
        if self._window is None:
            return {"error": "Window not ready yet."}
        try:
            self._window.destroy()
            return {"status": "closed"}
        except Exception as exc:
            return {"error": f"Failed to close: {exc}"}

    def load_node_diagram(self, node_id: str, overview_path: str | None = None) -> dict:
        if not node_id:
            return {"type": "diagram", "error": "Missing node id."}
        if not overview_path:
            return {"type": "diagram", "error": "Missing overview path."}
        try:
            run_dir = Path(overview_path).parent
            node_path = run_dir / f"{node_id}.d2"
            if not node_path.is_file():
                return {
                    "type": "diagram",
                    "error": f"No node diagram found for {node_id}.",
                    "missing": True,
                    "node_id": node_id,
                }
            content = node_path.read_text(encoding="utf-8")
            message = {"content": content, "path": str(node_path)}
            return _build_diagram_payload(message, self._ctx.tempdir)
        except Exception as exc:
            return {"type": "diagram", "error": f"Failed to load node diagram: {exc}"}


def _build_diagram_payload(message: dict, tmpdir: Path) -> dict:
    content = message.get("content") or ""
    path = message.get("path") or ""
    label = f"D2: {path}" if path else "D2: (no path)"

    svg_path, svg_err = _render_to_svg(content, tmpdir)
    error = svg_err

    svg_inline = None
    svg_data = None
    if svg_path:
        svg_inline = _read_svg_text(svg_path)
        if not svg_inline:
            svg_path = _flatten_svg(svg_path)
            svg_inline = _read_svg_text(svg_path)
        # Intentionally skip data URL fallback so SVG remains interactive.
        svg_data = None

    if not svg_inline:
        error = error or "D2 render failed: no SVG output produced."

    if error:
        pass

    return {
        "type": "diagram",
        "label": label,
        "path": path,
        "svg_inline": svg_inline,
        "svg": svg_data,
        "error": error,
    }


def _render_to_svg(content: str, tmpdir: Path) -> tuple[Path | None, str | None]:
    """Render D2 content to SVG using the d2 CLI."""
    d2_bin = _find_d2_bin()
    if not d2_bin:
        return None, "Install the d2 CLI to render diagrams."
    tmpdir.mkdir(parents=True, exist_ok=True)
    fd, path_str = tempfile.mkstemp(prefix="unslop_overlay_", suffix=".svg", dir=tmpdir)
    os.close(fd)
    svg_path = Path(path_str)
    try:
        proc = subprocess.run(
            [d2_bin, "-", str(svg_path)],
            input=content,
            text=True,
            capture_output=True,
            timeout=30,
        )
        if proc.returncode != 0:
            detail = (proc.stderr or proc.stdout or "").strip()
            return None, f"d2 render failed ({proc.returncode}): {detail or 'unknown error'}"
        return svg_path, None
    except Exception as exc:
        return None, f"d2 render error: {exc}"


def _to_data_url(path: Path | None, mime: str) -> str | None:
    if not path or not path.exists():
        return None
    try:
        data = path.read_bytes()
        encoded = base64.b64encode(data).decode("ascii")
        return f"data:{mime};base64,{encoded}"
    except Exception:
        return None


def _read_svg_text(svg_path: Path) -> str | None:
    """Read SVG text and strip XML declarations to allow inline HTML usage."""
    if not svg_path.exists():
        return None
    try:
        text = svg_path.read_text(encoding="utf-8")
    except Exception:
        return None
    stripped = text.lstrip("\ufeff").lstrip()
    if stripped.startswith("<?xml") and "?>" in stripped:
        stripped = stripped.split("?>", 1)[1]
    if stripped.startswith("<!DOCTYPE"):
        parts = stripped.split(">", 1)
        stripped = parts[1] if len(parts) > 1 else ""
    stripped = _ensure_svg_xmlns(stripped)
    return stripped.strip() or None


def _ensure_svg_xmlns(text: str) -> str:
    """Ensure inline SVG has an xmlns attribute so browsers render it."""
    svg_index = text.find("<svg")
    if svg_index == -1:
        return text
    tag_end = text.find(">", svg_index)
    if tag_end == -1:
        return text
    start_tag = text[svg_index:tag_end]
    if "xmlns=" in start_tag:
        return text
    insert_at = svg_index + 4
    return f"{text[:insert_at]} xmlns=\"http://www.w3.org/2000/svg\"{text[insert_at:]}"


def _find_d2_bin() -> str | None:
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


def _signal_selection(queue: multiprocessing.Queue | None, value: str | None) -> None:
    if queue is None:
        return
    try:
        queue.put_nowait(value)
    except Exception:
        try:
            queue.put(value)
        except Exception:
            pass

def _normalize_overview_model(value: str | None) -> str:
    allowed = {"gpt-4.1", "gpt-5-mini"}
    if isinstance(value, str) and value in allowed:
        return value
    return "gpt-4.1"


def _resolve_env_path(ctx: _OverlayContext) -> Path | None:
    try:
        return Path.cwd() / ".env"
    except Exception:
        return None


def _read_env_key(env_path: Path | None) -> str | None:
    if not env_path or not env_path.is_file():
        return None
    try:
        lines = env_path.read_text(encoding="utf-8").splitlines()
    except Exception:
        return None
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("export "):
            stripped = stripped[len("export ") :].lstrip()
        if stripped.startswith("OPENAI_API_KEY="):
            value = stripped.split("=", 1)[1].strip()
            return _strip_env_value(value)
        if stripped.startswith("UNSLOP_OPENAI_API_KEY="):
            value = stripped.split("=", 1)[1].strip()
            return _strip_env_value(value)
    return None


def _strip_env_value(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def _write_env_key(env_path: Path, value: str) -> None:
    existing = []
    if env_path.is_file():
        existing = env_path.read_text(encoding="utf-8").splitlines()
    keys = ("OPENAI_API_KEY", "UNSLOP_OPENAI_API_KEY")
    if not value:
        updated = [
            line
            for line in existing
            if not any(_is_env_key_line(line, key) for key in keys)
        ]
    else:
        updated = existing[:]
        for key in keys:
            updated = _upsert_env_line(updated, key, value)
    content = "\n".join(updated).rstrip()
    if content:
        content += "\n"
    env_path.write_text(content, encoding="utf-8")


def _upsert_env_line(lines: list[str], key: str, value: str) -> list[str]:
    updated = []
    replaced = False
    for line in lines:
        if _is_env_key_line(line, key):
            updated.append(f"{key}={value}")
            replaced = True
        else:
            updated.append(line)
    if not replaced:
        updated.append(f"{key}={value}")
    return updated


def _is_env_key_line(line: str, key: str) -> bool:
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return False
    if stripped.startswith("export "):
        stripped = stripped[len("export ") :].lstrip()
    return stripped.startswith(f"{key}=")
        
def _load_overlay_html() -> str:
    template_path = Path(__file__).resolve().parent / "templates" / "overlay.html"
    try:
        return template_path.read_text(encoding="utf-8")
    except Exception:
        return "<html><body><p>unslop overlay template missing.</p></body></html>"
