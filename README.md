# Noodles

## TL;DR

Your codebase was probably AI-generated. Get a better handle on it.

Noodles creates interactive diagrams that visualize how your code actually works, so you can understand what the AI built without reading every line.

![noodles demo](assets/demo.gif)

Hosted version available at https://unslop.xyz.

## What it does

- Builds function call graphs using tree-sitter AST parsing
- Generates mermaid diagrams showing code flow
- Interactive viewer (pan, zoom, drill down into sub-diagrams)
- **Repo analyzer** - Analyze an entire repository
- **PR analyzer** - Analyze a GitHub PR to see what changed

## Installation

```bash
pip install git+https://github.com/unslop-xyz/noodles.git
```

Or for development:

```bash
git clone https://github.com/unslop-xyz/noodles.git
cd noodles
pip install -e .
```

## Configuration

Set your LLM API key as an environment variable:

```bash
export ANTHROPIC_API_KEY=your-key-here
```

Supported providers: `anthropic` (default), `openai`, `gemini`, `groq`, `huggingface`. To use a different provider:

```bash
export LLM_PROVIDER=openai
export OPENAI_API_KEY=your-key-here
```

Alternatively, create a `.env` file in your working directory (see `.env.example` if you cloned the repo).

An API key enables AI-powered enrichment of node descriptions and edge labels. Without it, the call graph will still be generated but nodes and edges won't have human-readable descriptions.

Noodles uses `gh` to fetch your repo/PR. To set it up, run:
```bash
brew install gh
gh auth login
```

## Usage

### Analyze a repository

```bash
noodles repo <repo-url>
```

### Analyze a PR

```bash
noodles pr <github-pr-url>
```

### Open viewer for existing results

```bash
noodles viewer <result-dir>
```

### Options

- `--output-dir <dir>` - Custom output directory for results
- `--no-view` - Don't open the viewer after analysis

## MCP Server (Claude Code Integration)

Noodles can run as an MCP server, allowing Claude Code to analyze PRs and repositories directly.

### Setup

1. Install with MCP dependencies:
   ```bash
   pip install "noodles[mcp]"
   ```

2. Add to your Claude Code settings (`~/.claude/settings.json`):
   ```json
   {
     "mcpServers": {
       "noodles": {
         "command": "noodles-mcp"
       }
     }
   }
   ```

3. Restart Claude Code

4. Verify with `/mcp` - you should see the noodles server listed

### Available Tools

Once configured, Claude Code can use these tools:

- `analyze_pr` - Analyze a GitHub PR
- `analyze_repo` - Analyze an entire repository
- `analyze_local_repo` - Analyze a local repository (no cloning)
- `analyze_changes` - Analyze uncommitted local changes
- `get_diagram` - Get Mermaid diagram from analysis
- `get_call_graph` - Get full call graph JSON
- `find_path` - Find call path between two functions
- `get_callers` - Get functions that call a given function
- `get_callees` - Get functions called by a given function
- `filter_graph` - Filter nodes by type (entry_points, endpoints, new, updated, orphans)
- `get_summary` - Get plain English summary of analysis impact
- `get_changes` - Get detailed change info for modified functions

### Viewer controls

- **Pan** - Click and drag
- **Zoom** - Scroll wheel
- **Drill down** - Click `[+]` nodes
- **Back** - Escape

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
