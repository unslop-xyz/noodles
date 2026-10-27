"""Tests for call graph support in Go, Java, Rust and C#."""

import tempfile
from pathlib import Path

from noodles.agents.repo_analyzer.call_graph_builder.call_graph_builder import (
    _collect_file_imports,
    build_call_graph,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_code(code: str, lang: str):
    """Parse code and return the root node."""
    from tree_sitter_language_pack import get_parser

    parser = get_parser(lang)
    return parser.parse(code.encode("utf-8")).root_node


def _write_repo(repo_path: Path, files: dict[str, str]) -> None:
    for rel, content in files.items():
        path = repo_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)


def _graph(files: dict[str, str]) -> tuple[set[str], set[tuple[str, str]]]:
    """Build a call graph for the given files; return (node ids, edges) with '/' separators."""
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_path = Path(tmpdir)
        _write_repo(repo_path, files)
        call_graph, _, _, _ = build_call_graph(repo_path)
    def norm(s: str) -> str:
        return s.replace("\\", "/")

    ids = {norm(n["id"]) for n in call_graph["nodes"]}
    edges = {(norm(e["from"]), norm(e["to"])) for e in call_graph["edges"]}
    return ids, edges


def _callees(edges: set[tuple[str, str]], caller: str) -> set[str]:
    return {to for frm, to in edges if frm == caller}


# ---------------------------------------------------------------------------
# Go
# ---------------------------------------------------------------------------


GO_REPO = {
    "go.mod": "module example.com/app\n\ngo 1.22\n",
    "server.go": """package main

import (
    "fmt"
    "example.com/app/util"
    "github.com/lib/client"
)

type Server struct{}

func (s *Server) Run() error {
    s.handle()
    fmt.Println("running")
    util.Helper()
    client.Get()
    go func() { cleanup() }()
    return nil
}

func (s Server) handle() {}

func cleanup() {}
""",
    "util/util.go": """package util

func Helper() int { return 1 }

func Get() int { return 2 }
""",
}


class TestGo:
    def test_functions_and_methods_discovered(self):
        ids, _ = _graph(GO_REPO)
        assert "server.go::Server.Run" in ids
        assert "server.go::Server.handle" in ids
        assert "server.go::cleanup" in ids
        assert "util/util.go::Helper" in ids

    def test_calls_resolved(self):
        _, edges = _graph(GO_REPO)
        callees = _callees(edges, "server.go::Server.Run")
        assert "server.go::Server.handle" in callees
        # Internal package (matches go.mod module path) is followed
        assert "util/util.go::Helper" in callees
        # client.Get() is an external package, must not match util.Get
        assert "util/util.go::Get" not in callees
        # Calls inside a func literal are not attributed to the enclosing function
        assert "server.go::cleanup" not in callees

    def test_imports(self):
        root = _parse_code(
            'package x\nimport (\n  "fmt"\n  y "gopkg.in/yaml"\n  _ "embed"\n  "github.com/a/pkg/v2"\n)\n',
            "go",
        )
        assert _collect_file_imports(root, "go") == {
            "fmt": "fmt",
            "y": "gopkg.in/yaml",
            "pkg": "github.com/a/pkg/v2",
        }


# ---------------------------------------------------------------------------
# Java
# ---------------------------------------------------------------------------


JAVA_REPO = {
    "src/main/java/com/example/app/Service.java": """package com.example.app;

import java.util.List;
import com.example.app.data.Store;

public class Service {
    public Service() { init(); }

    private void init() {}

    public void run() {
        this.helper();
        Repo.save();
        List.of(1, 2);
        new Repo();
        Store.load();
        List.of(1).forEach(x -> cleanup());
    }

    private void helper() {}

    private void cleanup() {}
}
""",
    "src/main/java/com/example/app/Repo.java": """package com.example.app;

class Repo {
    Repo() {}
    static void save() {}
    void of() {}
}
""",
    "src/main/java/com/example/app/data/Store.java": """package com.example.app.data;

public class Store {
    public static void load() {}
}
""",
}

JAVA_SERVICE = "src/main/java/com/example/app/Service.java"
JAVA_REPO_FILE = "src/main/java/com/example/app/Repo.java"


class TestJava:
    def test_methods_and_constructors_discovered(self):
        ids, _ = _graph(JAVA_REPO)
        assert f"{JAVA_SERVICE}::Service.Service" in ids
        assert f"{JAVA_SERVICE}::Service.run" in ids
        assert f"{JAVA_REPO_FILE}::Repo.Repo" in ids

    def test_calls_resolved(self):
        _, edges = _graph(JAVA_REPO)
        callees = _callees(edges, f"{JAVA_SERVICE}::Service.run")
        assert f"{JAVA_SERVICE}::Service.helper" in callees
        assert f"{JAVA_REPO_FILE}::Repo.save" in callees
        # `new Repo()` is a constructor call
        assert f"{JAVA_REPO_FILE}::Repo.Repo" in callees
        # Import of the repo's own package is internal
        assert "src/main/java/com/example/app/data/Store.java::Store.load" in callees
        # java.util.List.of() must not match the internal Repo.of()
        assert f"{JAVA_REPO_FILE}::Repo.of" not in callees
        # Calls inside lambdas are not attributed to the enclosing method
        assert f"{JAVA_SERVICE}::Service.cleanup" not in callees

        assert f"{JAVA_SERVICE}::Service.init" in _callees(edges, f"{JAVA_SERVICE}::Service.Service")

    def test_imports(self):
        root = _parse_code(
            "import java.util.List;\nimport static java.lang.Math.max;\nimport com.foo.*;\n",
            "java",
        )
        assert _collect_file_imports(root, "java") == {
            "List": "java.util.List",
            "max": "java.lang.Math.max",
        }


# ---------------------------------------------------------------------------
# Rust
# ---------------------------------------------------------------------------


RUST_REPO = {
    "Cargo.toml": '[package]\nname = "my-app"\nversion = "0.1.0"\n\n[dependencies]\nserde = "1"\n',
    "src/main.rs": """use std::collections::HashMap;
use my_app::store::Store;

struct App;

impl App {
    fn new() -> Self {
        Self::setup();
        App
    }

    fn setup() {}

    fn run(&self) {
        self.step();
        let _m: HashMap<i32, i32> = HashMap::new();
        Store::open();
        let f = || cleanup();
        compute::<i32>();
        println!("done");
    }

    fn step(&self) {}
}

fn cleanup() {}

fn compute<T>() {}
""",
    "src/store.rs": """pub struct Store;

impl Store {
    pub fn open() {}
}
""",
}


class TestRust:
    def test_functions_discovered(self):
        ids, _ = _graph(RUST_REPO)
        assert "src/main.rs::App.new" in ids
        assert "src/main.rs::App.run" in ids
        assert "src/main.rs::compute" in ids
        assert "src/store.rs::Store.open" in ids

    def test_calls_resolved(self):
        _, edges = _graph(RUST_REPO)
        callees = _callees(edges, "src/main.rs::App.run")
        assert "src/main.rs::App.step" in callees
        # `use my_app::...` refers to this crate (name from Cargo.toml)
        assert "src/store.rs::Store.open" in callees
        # Turbofish call
        assert "src/main.rs::compute" in callees
        # HashMap::new() is from std, must not match App::new
        assert "src/main.rs::App.new" not in callees
        # Calls inside closures are not attributed to the enclosing function
        assert "src/main.rs::cleanup" not in callees

        assert "src/main.rs::App.setup" in _callees(edges, "src/main.rs::App.new")

    def test_imports(self):
        root = _parse_code(
            "use std::collections::HashMap;\n"
            "use serde::{Serialize, Deserialize as De};\n"
            "use crate::local::thing;\n"
            "use super::other;\n",
            "rust",
        )
        assert _collect_file_imports(root, "rust") == {
            "HashMap": "std",
            "Serialize": "serde",
            "De": "serde",
        }


# ---------------------------------------------------------------------------
# C#
# ---------------------------------------------------------------------------


CSHARP_REPO = {
    "Service.cs": """using System;
using IO = System.IO;

namespace App.Core
{
    public class Service
    {
        public Service() { Init(); }

        private void Init() {}

        public int Run()
        {
            this.Step();
            var r = new Repo();
            Console.WriteLine("x");
            IO.File.Exists("a");
            Repo.Save();
            Action a = () => Cleanup();
            return Compute();
        }

        private void Step() {}

        private int Compute() => Helper();

        private int Helper()
        {
            int Local() => 1;
            return Local();
        }

        void Cleanup() {}
    }
}
""",
    "Repo.cs": """namespace App.Core;

class Repo
{
    public Repo() {}
    public static void Save() {}
    public static bool Exists(string path) => true;
}
""",
}


class TestCSharp:
    def test_methods_discovered(self):
        ids, _ = _graph(CSHARP_REPO)
        assert "Service.cs::Service.Service" in ids
        assert "Service.cs::Service.Run" in ids
        assert "Service.cs::Service.Local" in ids
        assert "Repo.cs::Repo.Save" in ids

    def test_calls_resolved(self):
        _, edges = _graph(CSHARP_REPO)
        callees = _callees(edges, "Service.cs::Service.Run")
        assert "Service.cs::Service.Step" in callees
        assert "Service.cs::Service.Compute" in callees
        assert "Repo.cs::Repo.Repo" in callees
        assert "Repo.cs::Repo.Save" in callees
        # IO is an alias for System.IO, so IO.File.Exists() must not match Repo.Exists
        assert "Repo.cs::Repo.Exists" not in callees
        # Calls inside lambdas are not attributed to the enclosing method
        assert "Service.cs::Service.Cleanup" not in callees

        # Expression-bodied method and local function
        assert "Service.cs::Service.Helper" in _callees(edges, "Service.cs::Service.Compute")
        assert "Service.cs::Service.Local" in _callees(edges, "Service.cs::Service.Helper")


# ---------------------------------------------------------------------------
# Behaviour shared with Python/JS, exercised in the new languages
# ---------------------------------------------------------------------------


class TestReceiverNarrowing:
    def test_rust_type_receiver_picks_that_types_method(self):
        """Foo::new() links to Foo::new only, not every `new` in the repo."""
        _, edges = _graph({
            "src/lib.rs": """struct Foo;
struct Bar;
impl Foo { fn new() -> Self { Foo } }
impl Bar { fn new() -> Self { Bar } }
fn make() { Foo::new(); }
""",
        })
        assert _callees(edges, "src/lib.rs::make") == {"src/lib.rs::Foo.new"}

    def test_java_unknown_receiver_keeps_all_candidates(self):
        """A variable receiver can't be typed, so all same-named methods remain."""
        _, edges = _graph({
            "Shapes.java": """class Circle { int area() { return 1; } }
class Square { int area() { return 2; } }
class Report { int total(Object shape) { return shape.area(); } }
""",
        })
        assert _callees(edges, "Shapes.java::Report.total") == {
            "Shapes.java::Circle.area", "Shapes.java::Square.area",
        }


class TestSkipDirs:
    def test_vendor_dir_skipped(self):
        ids, _ = _graph({
            "main.go": "package main\n\nfunc main() {}\n",
            "vendor/lib/lib.go": "package lib\n\nfunc Vendored() {}\n",
        })
        assert "main.go::main" in ids
        assert not any("Vendored" in i for i in ids)
