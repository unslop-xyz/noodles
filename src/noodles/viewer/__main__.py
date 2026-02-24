"""CLI entry point: python -m noodles.viewer <result_dir>"""

import argparse
import sys
import webbrowser

from .data_loader import load_result
from .server import start_server


def main():
    parser = argparse.ArgumentParser(description="Interactive mermaid diagram viewer")
    parser.add_argument("result_dir", help="Path to the result folder")
    parser.add_argument("--port", type=int, default=0, help="Port (0=auto)")
    args = parser.parse_args()

    try:
        data = load_result(args.result_dir)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    server = start_server(data, port=args.port)
    host, port = server.server_address
    url = f"http://{host}:{port}"
    print(f"Viewer running at {url}")
    webbrowser.open(url)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.shutdown()


if __name__ == "__main__":
    main()
