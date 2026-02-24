"""Noodles - AI-powered code visualization and analysis."""

from importlib.metadata import version

__version__ = version("noodles")

from noodles.llm import (
    get_provider,
    LLMProvider,
    LLMResponse,
    calculate_cost,
    get_pricing,
)

from noodles.agents.repo_analyzer.repo_analyzer import (
    analyze_repo,
    analyze_local_repo,
)

from noodles.agents.pr_analyzer.pr_analyzer import (
    analyze_pr,
    analyze_local_changes,
)

__all__ = [
    "__version__",
    # LLM utilities
    "get_provider",
    "LLMProvider",
    "LLMResponse",
    "calculate_cost",
    "get_pricing",
    # Repo analysis
    "analyze_repo",
    "analyze_local_repo",
    # PR/change analysis
    "analyze_pr",
    "analyze_local_changes",
]
