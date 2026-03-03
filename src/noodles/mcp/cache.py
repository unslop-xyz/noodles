"""Persistent cache for MCP analysis results.

Caches PR and repo analysis results to avoid re-analyzing the same commit.
"""

import json
import re
import shutil
import subprocess
from pathlib import Path
from urllib.parse import urlparse

CACHE_DIR = Path.home() / ".cache" / "noodles"
INDEX_FILE = CACHE_DIR / "index.json"


def _load_index() -> dict:
    """Load the cache index from disk."""
    if not INDEX_FILE.exists():
        return {}
    try:
        return json.loads(INDEX_FILE.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def _save_index(index: dict) -> None:
    """Save the cache index to disk."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    INDEX_FILE.write_text(json.dumps(index, indent=2))


def make_cache_key(
    repo_url: str,
    pr_number: int | None = None,
    head_sha: str | None = None,
) -> str:
    """Generate cache key from repo URL and optional PR/SHA info.

    Format: "github.com/owner/repo:pr:123:abc123" or "github.com/owner/repo:sha:abc123"

    Args:
        repo_url: GitHub repo or PR URL
        pr_number: PR number if analyzing a PR
        head_sha: Commit SHA to include in key

    Returns:
        Cache key string.
    """
    # Parse repo URL to get normalized identifier
    parsed = urlparse(repo_url)
    path = parsed.path.strip("/")

    # Remove .git suffix if present
    if path.endswith(".git"):
        path = path[:-4]

    # Remove /pull/N if present (for PR URLs)
    match = re.match(r"(.+?)/pull/\d+", path)
    if match:
        path = match.group(1)

    host = parsed.netloc or "github.com"
    repo_id = f"{host}/{path}"

    if pr_number is not None:
        key = f"{repo_id}:pr:{pr_number}"
    else:
        key = f"{repo_id}:repo"

    if head_sha:
        key += f":{head_sha[:12]}"

    return key


def get_pr_head_sha(owner: str, repo: str, pr_number: int) -> str | None:
    """Get the HEAD SHA for a PR using gh CLI.

    Args:
        owner: Repository owner
        repo: Repository name
        pr_number: PR number

    Returns:
        SHA string or None if not found.
    """
    try:
        result = subprocess.run(
            [
                "gh", "api",
                f"repos/{owner}/{repo}/pulls/{pr_number}",
                "--jq", ".head.sha",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


def get_repo_head_sha(repo_url: str) -> str | None:
    """Get the HEAD SHA for a repo using gh CLI.

    Args:
        repo_url: GitHub repo URL

    Returns:
        SHA string or None if not found.
    """
    # Parse owner/repo from URL
    parsed = urlparse(repo_url)
    path = parsed.path.strip("/")
    if path.endswith(".git"):
        path = path[:-4]

    try:
        result = subprocess.run(
            [
                "gh", "api",
                f"repos/{path}/commits/HEAD",
                "--jq", ".sha",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


def check_cache(cache_key: str) -> dict | None:
    """Check if a cache entry exists and is valid.

    Args:
        cache_key: Cache key from make_cache_key()

    Returns:
        Dict with "analysis_id" and "result_dir" if cached, None otherwise.
    """
    index = _load_index()
    entry = index.get(cache_key)

    if entry is None:
        return None

    # Verify the result directory still exists
    result_dir = Path(entry["result_dir"])
    if not result_dir.exists():
        # Stale entry - remove from index
        del index[cache_key]
        _save_index(index)
        return None

    # Verify call_graph.json exists (basic validity check)
    if not (result_dir / "call_graph.json").exists():
        del index[cache_key]
        _save_index(index)
        return None

    return entry


def store_in_cache(
    cache_key: str,
    analysis_id: str,
    result_dir: Path,
) -> None:
    """Store analysis result in persistent cache.

    Args:
        cache_key: Cache key from make_cache_key()
        analysis_id: The analysis ID
        result_dir: Path to the result directory
    """
    index = _load_index()
    index[cache_key] = {
        "analysis_id": analysis_id,
        "result_dir": str(result_dir.resolve()),
    }
    _save_index(index)


def copy_cached_result(
    cached_entry: dict,
    new_analysis_id: str,
    new_output_dir: Path,
) -> Path:
    """Copy cached result to a new location with new analysis_id.

    Args:
        cached_entry: Entry from check_cache()
        new_analysis_id: New analysis ID
        new_output_dir: Base directory for the new result

    Returns:
        Path to the new result directory.
    """
    cached_dir = Path(cached_entry["result_dir"])
    new_result_dir = new_output_dir / f"result_{new_analysis_id}"

    # Copy result files (not the cloned repo to save space)
    new_result_dir.mkdir(parents=True, exist_ok=True)

    files_to_copy = [
        "call_graph.json",
        "classification.json",
        "start_points.json",
        "end_points.json",
        "orphans.json",
        "viewer.html",
        "viewer.js",
    ]

    for fname in files_to_copy:
        src = cached_dir / fname
        if src.exists():
            shutil.copy2(src, new_result_dir / fname)

    # Copy diagrams
    for diagram in cached_dir.glob("diagram_*.mmd"):
        shutil.copy2(diagram, new_result_dir / diagram.name)

    return new_result_dir


def clear_cache() -> int:
    """Clear all cached entries.

    Returns:
        Number of entries cleared.
    """
    index = _load_index()
    count = len(index)
    _save_index({})
    return count


def list_cache() -> list[dict]:
    """List all cache entries.

    Returns:
        List of cache entries with key, analysis_id, and result_dir.
    """
    index = _load_index()
    entries = []
    for key, entry in index.items():
        entries.append({
            "cache_key": key,
            "analysis_id": entry["analysis_id"],
            "result_dir": entry["result_dir"],
        })
    return entries
