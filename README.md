# Noodles

## TL;DR

Your codebase was probably AI-generated. Get a better handle on it.

Noodles creates interactive diagrams that visualize how your code actually works, so you can understand what the AI built without reading every line.

![noodles demo](assets/demo.gif)

## What it does

- Scans a folder and builds a manifest of your code
- Uses OpenAI to identify user-facing entry points (CLI commands, routes, UI components)
- Generates D2 diagrams showing how code flows from entry to outcome
- Renders an interactive overlay to explore the diagrams
- Tracks changes and updates diagrams incrementally when code changes

## Setup

### Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip
- [d2](https://d2lang.com/) CLI for diagram rendering
- OpenAI API key

### Install dependencies

```bash
# Using uv (recommended)
uv sync

# Or using pip
pip install -e '.[dev]'
```

### Install d2

```bash
# macOS
brew install d2

# Or download from https://d2lang.com
```

### Configure OpenAI API key

**Option 1: `.env` file (recommended)**

Create a `.env` file in your project root:
```
OPENAI_API_KEY=your-key
```
The CLI automatically loads `.env` files on startup.

**Option 2: Environment variable**
```bash
# macOS/Linux
export OPENAI_API_KEY="your-key"

# Windows (PowerShell)
$env:OPENAI_API_KEY="your-key"

# Windows (CMD)
set OPENAI_API_KEY=your-key
```

### Optional configuration

```bash
# Point to d2 binary if not on PATH
export UNSLOP_D2_BIN=/opt/homebrew/bin/d2

# Set log level (DEBUG, INFO, WARNING, ERROR)
export UNSLOP_LOG_LEVEL=INFO
```

## Usage

```bash
unslop run
```

- Use the overlay to select folders to analyze
- Diagrams and manifests are stored in `<folder>/.unslop/`
- Close the overlay to exit

### Overlay controls

- **Choose folder** - Select a codebase to analyze
- **Update** - Regenerate diagram only if code changed (incremental)
- **Rerun** - Full rebuild from scratch
- **?** - Hover for keyboard shortcuts and icon meanings
- Click any node to drill into details; hover for tooltips

## Known rough edges

- Speed of diagram generation; works best on projects with fewer than 100 files for now
- UI is intentionally verbose for debugging; simplification planned
- Diagram quality varies; prompt tuning ongoing
