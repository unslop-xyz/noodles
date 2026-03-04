# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.4.2] - 2026-03-03

### Fixed
- Clone repos to temp directory instead of package directory to avoid empty call graphs when installed via `uv tool` on Windows (fixes #20)

## [0.4.1] - 2026-03-03

### Fixed
- External library calls (e.g., `requests.get()`) no longer incorrectly match internal methods (e.g., `APIClient.get()`)
- Direct imported functions (e.g., `from subprocess import run`) no longer match internal functions with the same name

## [0.4.0] - 2026-03-03

### Added
- MCP server for Claude Code integration
- MCP query tools with caching
- Gitignore file check before building call graph

### Changed
- Large in-degree end nodes now moved to sub-diagrams for cleaner visualization

### Fixed
- fuzzy_match to support short_id format matching

## [0.3.1] - 2026-03-01

### Added
- Local-path analysis functions for embedded use

### Fixed
- Empty diagram for merged PRs
- Version number alignment (pyproject.toml now matches git tag)

## [0.3.0] - 2026-02-24

_Note: This release was tagged as v0.3.0 but pyproject.toml contained version 0.2.0. Fixed in 0.3.1._

### Added
- Self-contained viewer.html generation
- XSS protection for generated output

### Changed
- Updated README with unslop.xyz branding

### Fixed
- Git fetch auth for private repos
- TDZ error in generated viewer.html

## [0.1.1] - 2026-02-23

### Changed
- Use importlib.metadata for version detection

### Fixed
- Private repo fetch

## [0.1.0] - 2026-02-23

### Added
- Initial release as Python package with `noodles` CLI
- LLM provider abstraction (OpenAI, Anthropic, Gemini, Groq, HuggingFace)
- React/JSX component detection in call graphs
- Exponential backoff retry for rate limit errors
- Interactive viewer with tooltips and navigation
