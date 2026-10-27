"""Tests for narrowing name-matched call candidates by the call's receiver."""

import tempfile
from pathlib import Path

from noodles.agents.repo_analyzer.call_graph_builder.call_graph_builder import (
    build_call_graph,
)


def _callees(files: dict[str, str], caller: str) -> set[str]:
    """Build a call graph for the given files and return the callees of `caller`."""
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_path = Path(tmpdir)
        for rel, content in files.items():
            (repo_path / rel).write_text(content)
        call_graph, _, _, _ = build_call_graph(repo_path)
    return {
        e["to"].replace("\\", "/")
        for e in call_graph["edges"]
        if e["from"].replace("\\", "/") == caller
    }


SHAPES_PY = """class Circle:
    def area(self):
        return 1

    def describe(self):
        return self.area()

    @classmethod
    def unit(cls):
        return cls.create()

    @classmethod
    def create(cls):
        return Circle()


class Square:
    def area(self):
        return 2

    @classmethod
    def create(cls):
        return Square()


def build():
    return Square.create()


def total(shape):
    return shape.area()
"""


class TestReceiverNarrowing:
    def test_self_receiver_prefers_own_class(self):
        assert _callees({"shapes.py": SHAPES_PY}, "shapes.py::Circle.describe") == {
            "shapes.py::Circle.area"
        }

    def test_cls_receiver_prefers_own_class(self):
        assert _callees({"shapes.py": SHAPES_PY}, "shapes.py::Circle.unit") == {
            "shapes.py::Circle.create"
        }

    def test_class_receiver_picks_that_classes_method(self):
        """Square.create() links to Square.create only, not every `create`."""
        assert _callees({"shapes.py": SHAPES_PY}, "shapes.py::build") == {
            "shapes.py::Square.create"
        }

    def test_unknown_receiver_keeps_all_candidates(self):
        """A variable receiver can't be typed, so all same-named methods remain."""
        assert _callees({"shapes.py": SHAPES_PY}, "shapes.py::total") == {
            "shapes.py::Circle.area",
            "shapes.py::Square.area",
        }

    def test_js_this_receiver_prefers_own_class(self):
        code = """class A {
  run() { return this.step(); }
  step() { return 1; }
}
class B {
  step() { return 2; }
}
"""
        assert _callees({"shapes.js": code}, "shapes.js::A.run") == {"shapes.js::A.step"}
