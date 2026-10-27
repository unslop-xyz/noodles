"""Tests for which files the call graph builder walks."""

import tempfile
from pathlib import Path

from noodles.agents.repo_analyzer.call_graph_builder.call_graph_builder import (
    build_call_graph,
)


def _node_ids(repo_path: Path) -> set[str]:
    call_graph, _, _, _ = build_call_graph(repo_path)
    return {n["id"].replace("\\", "/") for n in call_graph["nodes"]}


class TestSkipDirs:
    def test_repo_inside_skipped_dir_name_is_still_analyzed(self):
        """SKIP_DIRS applies to paths inside the repo, not to where the repo lives.

        e.g. a repo checked out under `.claude/worktrees/` or `~/build/` used to
        produce an empty call graph.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_path = Path(tmpdir) / "build" / ".claude" / "myrepo"
            repo_path.mkdir(parents=True)
            (repo_path / "main.py").write_text("def hello():\n    pass\n")

            assert "main.py::hello" in _node_ids(repo_path)

    def test_skip_dirs_inside_repo_are_skipped(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_path = Path(tmpdir)
            (repo_path / "main.py").write_text("def hello():\n    pass\n")
            (repo_path / "node_modules" / "lib").mkdir(parents=True)
            (repo_path / "node_modules" / "lib" / "index.js").write_text(
                "function vendored() {}\n"
            )

            ids = _node_ids(repo_path)
            assert "main.py::hello" in ids
            assert not any("vendored" in i for i in ids)

    def test_gitignored_subdirectory_is_skipped(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_path = Path(tmpdir)
            (repo_path / ".gitignore").write_text("generated/\n")
            (repo_path / "main.py").write_text("def hello():\n    pass\n")
            (repo_path / "generated").mkdir()
            (repo_path / "generated" / "out.py").write_text("def gen():\n    pass\n")

            ids = _node_ids(repo_path)
            assert "main.py::hello" in ids
            assert not any("gen" in i.split("::")[-1] for i in ids)
