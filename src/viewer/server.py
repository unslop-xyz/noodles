"""HTTP server for the mermaid diagram viewer."""

import json
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path


class ViewerHandler(SimpleHTTPRequestHandler):
    """Serves the viewer static files and the data API."""

    data_bundle: str = "{}"

    def do_GET(self):
        if self.path == "/api/data":
            self._serve_json(self.data_bundle)
        elif self.path == "/" or self.path == "/index.html":
            self._serve_static("index.html")
        else:
            self.send_error(404)

    def _serve_json(self, body: str):
        data = body.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _serve_static(self, filename: str):
        static_dir = Path(__file__).parent / "static"
        filepath = static_dir / filename
        if not filepath.is_file():
            self.send_error(404)
            return
        data = filepath.read_bytes()
        content_type = "text/html; charset=utf-8"
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format, *args):
        # Suppress default request logging
        pass


def start_server(data_bundle: dict, port: int = 0) -> HTTPServer:
    """Start the viewer HTTP server.

    Args:
        data_bundle: The data bundle dict from load_result().
        port: Port to bind to. 0 = auto-assign.

    Returns:
        The running HTTPServer instance.
    """
    ViewerHandler.data_bundle = json.dumps(data_bundle, ensure_ascii=False)
    server = HTTPServer(("127.0.0.1", port), ViewerHandler)
    return server
