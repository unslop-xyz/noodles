"""Tests that imports of the repo's own code are not treated as external libraries."""

import tempfile
from pathlib import Path

from noodles.agents.repo_analyzer.call_graph_builder.call_graph_builder import (
    build_call_graph,
)


def _edges(files: dict[str, str]) -> set[tuple[str, str]]:
    """Build a call graph for the given files; return edges with '/' separators."""
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_path = Path(tmpdir)
        for rel, content in files.items():
            path = repo_path / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
        call_graph, _, _, _ = build_call_graph(repo_path)
    return {
        (e["from"].replace("\\", "/"), e["to"].replace("\\", "/"))
        for e in call_graph["edges"]
    }


class TestPythonInternalImports:
    def test_absolute_import_of_own_package_is_followed(self):
        """from mypkg.core import compute: mypkg is part of the repo, not a library."""
        edges = _edges({
            "mypkg/__init__.py": "",
            "mypkg/core.py": "def compute():\n    return 1\n",
            "app.py": "from mypkg.core import compute\n\ndef main():\n    return compute()\n",
        })
        assert ("app.py::main", "mypkg/core.py::compute") in edges

    def test_module_receiver_of_own_package_is_followed(self):
        """import mypkg.core; mypkg.core.compute()"""
        edges = _edges({
            "mypkg/__init__.py": "",
            "mypkg/core.py": "def compute():\n    return 1\n",
            "app.py": "import mypkg.core\n\ndef main():\n    return mypkg.core.compute()\n",
        })
        assert ("app.py::main", "mypkg/core.py::compute") in edges

    def test_src_layout(self):
        edges = _edges({
            "src/mypkg/__init__.py": "",
            "src/mypkg/core.py": "def compute():\n    return 1\n",
            "src/mypkg/cli.py": "from mypkg.core import compute\n\ndef main():\n    return compute()\n",
        })
        assert ("src/mypkg/cli.py::main", "src/mypkg/core.py::compute") in edges

    def test_top_level_module(self):
        edges = _edges({
            "helpers.py": "def compute():\n    return 1\n",
            "app.py": "from helpers import compute\n\ndef main():\n    return compute()\n",
        })
        assert ("app.py::main", "helpers.py::compute") in edges

    def test_external_import_still_filtered(self):
        edges = _edges({
            "mypkg/__init__.py": "",
            "mypkg/runner.py": "def run():\n    pass\n",
            "app.py": "from subprocess import run\n\ndef main():\n    run()\n",
        })
        assert not any(frm == "app.py::main" for frm, _ in edges)
