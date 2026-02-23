"""Rerun diagram_builder from saved call_graph.json and start_points.json."""

import json
import sys
from pathlib import Path

AGENTS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(AGENTS_DIR))

from diagram_builder.diagram_builder import build_diagrams


def main() -> None:
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <result_dir>", file=sys.stderr)
        sys.exit(1)

    result_dir = Path(sys.argv[1])
    call_graph = json.loads((result_dir / "call_graph.json").read_text())
    start_points = json.loads((result_dir / "start_points.json").read_text())

    diagrams = build_diagrams(call_graph, start_points)

    main_file = result_dir / "diagram_main.mmd"
    main_file.write_text(diagrams["main"])
    print(f"Main diagram: {main_file}")

    for sub_name, sub_content in diagrams.get("sub_diagrams", {}).items():
        sub_file = result_dir / f"diagram_{sub_name}.mmd"
        sub_file.write_text(sub_content)
        print(f"Sub-diagram:  {sub_file}")


if __name__ == "__main__":
    main()
