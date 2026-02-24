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

__all__ = [
    "__version__",
    "get_provider",
    "LLMProvider",
    "LLMResponse",
    "calculate_cost",
    "get_pricing",
]
