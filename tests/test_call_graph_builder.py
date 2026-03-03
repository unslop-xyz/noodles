"""Tests for the call graph builder's external library filtering."""

import tempfile
from pathlib import Path

import pytest

from noodles.agents.repo_analyzer.call_graph_builder.call_graph_builder import (
    _collect_file_imports,
    _extract_callee_info,
    _get_full_receiver_name,
    _is_external_receiver,
    _is_imported_function,
    build_call_graph,
)


# ---------------------------------------------------------------------------
# Helper to parse code
# ---------------------------------------------------------------------------


def _parse_code(code: str, lang: str):
    """Parse code and return the root node."""
    from tree_sitter_language_pack import get_parser

    parser = get_parser(lang)
    tree = parser.parse(code.encode("utf-8"))
    return tree.root_node


# ---------------------------------------------------------------------------
# Tests for _extract_callee_info
# ---------------------------------------------------------------------------


def _find_first_call(node, lang: str):
    """Recursively find the first call node in the AST."""
    call_type = "call" if lang == "python" else "call_expression"
    if node.type == call_type:
        return node
    for child in node.children:
        result = _find_first_call(child, lang)
        if result:
            return result
    return None


class TestExtractCalleeInfo:
    """Tests for extracting callee name and receiver."""

    def test_simple_function_call(self):
        """helper() should return ('helper', None)."""
        code = "helper()"
        root = _parse_code(code, "python")
        call_node = _find_first_call(root, "python")
        name, receiver = _extract_callee_info(call_node, "python")
        assert name == "helper"
        assert receiver is None

    def test_method_call_on_module(self):
        """requests.get() should return ('get', 'requests')."""
        code = "requests.get(url)"
        root = _parse_code(code, "python")
        call_node = _find_first_call(root, "python")
        name, receiver = _extract_callee_info(call_node, "python")
        assert name == "get"
        assert receiver == "requests"

    def test_method_call_on_self(self):
        """self.helper() should return ('helper', 'self')."""
        code = "self.helper()"
        root = _parse_code(code, "python")
        call_node = _find_first_call(root, "python")
        name, receiver = _extract_callee_info(call_node, "python")
        assert name == "helper"
        assert receiver == "self"

    def test_chained_attribute_call(self):
        """os.path.join() should return ('join', 'os.path')."""
        code = "os.path.join(a, b)"
        root = _parse_code(code, "python")
        call_node = _find_first_call(root, "python")
        name, receiver = _extract_callee_info(call_node, "python")
        assert name == "join"
        assert receiver == "os.path"

    def test_js_method_call(self):
        """axios.get() in JS should return ('get', 'axios')."""
        code = "axios.get(url);"
        root = _parse_code(code, "javascript")
        call_node = root.children[0].children[0]  # expression_statement -> call_expression
        name, receiver = _extract_callee_info(call_node, "javascript")
        assert name == "get"
        assert receiver == "axios"

    def test_js_this_call(self):
        """this.helper() in JS should return ('helper', 'this')."""
        code = "this.helper();"
        root = _parse_code(code, "javascript")
        call_node = root.children[0].children[0]
        name, receiver = _extract_callee_info(call_node, "javascript")
        assert name == "helper"
        assert receiver == "this"


# ---------------------------------------------------------------------------
# Tests for _collect_file_imports
# ---------------------------------------------------------------------------


class TestCollectFileImports:
    """Tests for import collection."""

    def test_python_simple_import(self):
        """import requests should add requests to imports."""
        code = "import requests"
        root = _parse_code(code, "python")
        imports = _collect_file_imports(root, "python")
        assert "requests" in imports
        assert imports["requests"] == "requests"

    def test_python_from_import(self):
        """from subprocess import run should add run to imports."""
        code = "from subprocess import run"
        root = _parse_code(code, "python")
        imports = _collect_file_imports(root, "python")
        assert "run" in imports
        assert imports["run"] == "subprocess"

    def test_python_aliased_import(self):
        """import pandas as pd should add pd to imports."""
        code = "import pandas as pd"
        root = _parse_code(code, "python")
        imports = _collect_file_imports(root, "python")
        assert "pd" in imports
        assert imports["pd"] == "pandas"

    def test_python_relative_import_skipped(self):
        """from . import utils should not add to imports."""
        code = "from . import utils"
        root = _parse_code(code, "python")
        imports = _collect_file_imports(root, "python")
        assert "utils" not in imports

    def test_python_relative_from_import_skipped(self):
        """from ..models import User should not add to imports."""
        code = "from ..models import User"
        root = _parse_code(code, "python")
        imports = _collect_file_imports(root, "python")
        assert "User" not in imports

    def test_js_default_import(self):
        """import axios from 'axios' should add axios to imports."""
        code = "import axios from 'axios';"
        root = _parse_code(code, "javascript")
        imports = _collect_file_imports(root, "javascript")
        assert "axios" in imports
        assert imports["axios"] == "axios"

    def test_js_named_import(self):
        """import { get } from 'lodash' should add get to imports."""
        code = "import { get } from 'lodash';"
        root = _parse_code(code, "javascript")
        imports = _collect_file_imports(root, "javascript")
        assert "get" in imports
        assert imports["get"] == "lodash"

    def test_js_namespace_import(self):
        """import * as path from 'path' should add path to imports."""
        code = "import * as path from 'path';"
        root = _parse_code(code, "javascript")
        imports = _collect_file_imports(root, "javascript")
        assert "path" in imports
        assert imports["path"] == "path"

    def test_js_require(self):
        """const fs = require('fs') should add fs to imports."""
        code = "const fs = require('fs');"
        root = _parse_code(code, "javascript")
        imports = _collect_file_imports(root, "javascript")
        assert "fs" in imports
        assert imports["fs"] == "fs"

    def test_js_relative_import_skipped(self):
        """import util from './util' should not add to imports."""
        code = "import util from './util';"
        root = _parse_code(code, "javascript")
        imports = _collect_file_imports(root, "javascript")
        assert "util" not in imports


# ---------------------------------------------------------------------------
# Tests for _is_external_receiver
# ---------------------------------------------------------------------------


class TestIsExternalReceiver:
    """Tests for external receiver detection."""

    def test_self_is_not_external(self):
        """self should not be considered external."""
        imports = {"requests": "requests"}
        assert _is_external_receiver("self", imports, "python") is False

    def test_cls_is_not_external(self):
        """cls should not be considered external."""
        imports = {"requests": "requests"}
        assert _is_external_receiver("cls", imports, "python") is False

    def test_this_is_not_external(self):
        """this should not be considered external in JS."""
        imports = {"axios": "axios"}
        assert _is_external_receiver("this", imports, "javascript") is False

    def test_imported_module_is_external(self):
        """requests should be external when imported."""
        imports = {"requests": "requests"}
        assert _is_external_receiver("requests", imports, "python") is True

    def test_chained_import_is_external(self):
        """os.path should be external when os is imported."""
        imports = {"os": "os"}
        assert _is_external_receiver("os.path", imports, "python") is True

    def test_unknown_receiver_is_not_external(self):
        """Unknown receiver should not be considered external."""
        imports = {"requests": "requests"}
        assert _is_external_receiver("helper", imports, "python") is False

    def test_none_receiver_is_not_external(self):
        """None receiver should not be considered external."""
        imports = {"requests": "requests"}
        assert _is_external_receiver(None, imports, "python") is False


# ---------------------------------------------------------------------------
# Tests for _is_imported_function
# ---------------------------------------------------------------------------


class TestIsImportedFunction:
    """Tests for imported function detection."""

    def test_imported_function_with_no_receiver(self):
        """run() should be detected as imported when in imports."""
        imports = {"run": "subprocess"}
        assert _is_imported_function("run", None, imports) is True

    def test_local_function_not_imported(self):
        """helper() should not be detected as imported."""
        imports = {"run": "subprocess"}
        assert _is_imported_function("helper", None, imports) is False

    def test_function_with_receiver_not_imported_function(self):
        """self.run() should not be detected as imported function."""
        imports = {"run": "subprocess"}
        assert _is_imported_function("run", "self", imports) is False

    def test_function_with_module_receiver_not_imported_function(self):
        """subprocess.run() - handled by _is_external_receiver, not this."""
        imports = {"subprocess": "subprocess"}
        # Even though "run" might be in imports, if there's a receiver,
        # this function should return False
        assert _is_imported_function("run", "subprocess", imports) is False


# ---------------------------------------------------------------------------
# Integration tests with build_call_graph
# ---------------------------------------------------------------------------


class TestExternalCallFiltering:
    """Integration tests for external library call filtering."""

    def test_external_library_calls_not_matched(self):
        """requests.get() should NOT match internal APIClient.get()."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_path = Path(tmpdir)

            # Create a file with an internal get method
            api_client = repo_path / "api_client.py"
            api_client.write_text("""
class APIClient:
    def get(self, url):
        return {"data": "response"}
""")

            # Create a file that uses requests.get (external) - should NOT match
            caller = repo_path / "caller.py"
            caller.write_text("""
import requests

def fetch_data():
    return requests.get("http://example.com")
""")

            call_graph, start_points, end_points, orphans = build_call_graph(repo_path)

            # Find the fetch_data node
            fetch_data_node = None
            for node in call_graph["nodes"]:
                if "fetch_data" in node["id"]:
                    fetch_data_node = node
                    break

            # fetch_data should NOT have APIClient.get as a callee
            assert fetch_data_node is not None
            assert len(fetch_data_node["callees"]) == 0

    def test_self_method_calls_matched(self):
        """self.helper() SHOULD match internal methods."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_path = Path(tmpdir)

            # Create a file with internal method calls
            service = repo_path / "service.py"
            service.write_text("""
class Service:
    def helper(self):
        return "help"

    def process(self):
        return self.helper()
""")

            call_graph, start_points, end_points, orphans = build_call_graph(repo_path)

            # Find the process node
            process_node = None
            for node in call_graph["nodes"]:
                if "process" in node["id"]:
                    process_node = node
                    break

            # process should have helper as a callee
            assert process_node is not None
            assert len(process_node["callees"]) == 1
            assert "helper" in process_node["callees"][0]

    def test_imported_function_not_matched(self):
        """from subprocess import run - run() should NOT match internal run()."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_path = Path(tmpdir)

            # Create a file with an internal run function
            internal = repo_path / "internal.py"
            internal.write_text("""
def run(command):
    print(f"Running: {command}")
""")

            # Create a file that uses subprocess.run (external) - should NOT match
            caller = repo_path / "caller.py"
            caller.write_text("""
from subprocess import run

def execute():
    run("ls -la")
""")

            call_graph, start_points, end_points, orphans = build_call_graph(repo_path)

            # Find the execute node
            execute_node = None
            for node in call_graph["nodes"]:
                if "execute" in node["id"]:
                    execute_node = node
                    break

            # execute should NOT have internal.run as a callee
            assert execute_node is not None
            assert len(execute_node["callees"]) == 0

    def test_js_external_calls_not_matched(self):
        """axios.get() should NOT match internal DataService.get()."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_path = Path(tmpdir)

            # Create a file with an internal get method
            data_service = repo_path / "data_service.js"
            data_service.write_text("""
class DataService {
    get(url) {
        return { data: "response" };
    }
}
""")

            # Create a file that uses axios.get (external) - should NOT match
            caller = repo_path / "caller.js"
            caller.write_text("""
import axios from 'axios';

function fetchData() {
    return axios.get("http://example.com");
}
""")

            call_graph, start_points, end_points, orphans = build_call_graph(repo_path)

            # Find the fetchData node
            fetch_data_node = None
            for node in call_graph["nodes"]:
                if "fetchData" in node["id"]:
                    fetch_data_node = node
                    break

            # fetchData should NOT have DataService.get as a callee
            assert fetch_data_node is not None
            assert len(fetch_data_node["callees"]) == 0

    def test_internal_calls_still_work(self):
        """Direct internal function calls should still work."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_path = Path(tmpdir)

            # Create a file with internal function calls
            utils = repo_path / "utils.py"
            utils.write_text("""
def helper():
    return "help"

def process():
    return helper()
""")

            call_graph, start_points, end_points, orphans = build_call_graph(repo_path)

            # Find the process node
            process_node = None
            for node in call_graph["nodes"]:
                if "process" in node["id"]:
                    process_node = node
                    break

            # process should have helper as a callee
            assert process_node is not None
            assert len(process_node["callees"]) == 1
            assert "helper" in process_node["callees"][0]
