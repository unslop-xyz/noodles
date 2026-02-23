# Noodles

## TL;DR

Your codebase was probably AI-generated. Get a better handle on it.

Noodles creates interactive diagrams that visualize how your code actually works, so you can understand what the AI built without reading every line.

![noodles demo](assets/demo.gif)

## What it does

- Builds function call graphs using tree-sitter AST parsing
- Generates mermaid diagrams showing code flow
- Interactive viewer (pan, zoom, drill down into sub-diagrams)
- **Repo analyzer** - Analyze an entire repository
- **PR analyzer** - Analyze a GitHub PR to see what changed

## Setup

```bash
uv sync
export ANTHROPIC_API_KEY=<your-key> # Optional to enable AI-powered enrichment of node descriptions and edge labels.
```

## Usage

### Analyze a repository

```bash
python src/agents/repo_analyzer/repo_analyzer.py <repo-url>
```

### Analyze a PR

```bash
python src/agents/pr_analyzer/pr_analyzer.py <github-pr-url>
```

Options: `--output-dir <dir>`, `--no-view`

Results saved to `src/agents/*/result_<id>/`.

### Viewer controls

- **Pan** - Click and drag
- **Zoom** - Scroll wheel
- **Drill down** - Click `[+]` nodes
- **Back** - Escape

> **Note**: Previous code archived under `src/archive/`

## Supported languages

The call graph builder uses tree-sitter for AST parsing. Currently supported:

| Extension | Language   |
|-----------|------------|
| `.py`     | Python     |
| `.js`     | JavaScript |
| `.jsx`    | JavaScript |
| `.ts`     | TypeScript |
| `.tsx`    | TSX        |

React/Next.js JSX component usage (`<MyComponent />`) is detected as function calls, so component hierarchies appear correctly in the call graph.

### Not yet supported

- **Other languages** - Requires adding tree-sitter grammar and function detection logic

## Limitations

### Call detection

The call graph is built by detecting function calls in the AST. This works well for:
- Direct function calls: `foo()`, `obj.method()`
- Imported function calls
- JSX component usage: `<MyComponent />`, `<UI.Card />`

This does **not** detect:
- **Dynamic calls** - `getattr(obj, 'method')()`, `obj[key]()`
- **Callbacks passed to frameworks** - e.g., route handlers registered via decorators

### PR analyzer

The PR analyzer prunes the call graph to functions affected by a PR. This works best when:
- Changed functions call or are called by other functions
- The codebase has interconnected function calls

It produces limited results when:
- Changes are to isolated/standalone functions
- The codebase uses patterns not detected by AST analysis (decorators, dynamic dispatch)
