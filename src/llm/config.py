"""LLM provider configuration and factory.

Environment variables:
    LLM_PROVIDER: Provider name (anthropic, openai, gemini). Default: anthropic
    LLM_MODEL: Model name override. Default: provider-specific default
    LLM_BASE_URL: Base URL for OpenAI-compatible endpoints (openai provider only)
"""

import os
import sys
from typing import Literal

from .anthropic import AnthropicProvider
from .gemini import GeminiProvider
from .openai import OpenAIProvider
from .provider import LLMProvider

ProviderName = Literal["anthropic", "openai", "gemini"]

DEFAULT_PROVIDER: ProviderName = "anthropic"

_logged_provider = False


def get_provider(
    provider: ProviderName | None = None,
    model: str | None = None,
    base_url: str | None = None,
) -> LLMProvider:
    """Create an LLM provider based on configuration.

    Args:
        provider: Provider name. Falls back to LLM_PROVIDER env var, then default.
        model: Model name. Falls back to LLM_MODEL env var, then provider default.
        base_url: Base URL for OpenAI-compatible endpoints.
                  Falls back to LLM_BASE_URL env var.

    Returns:
        An LLMProvider instance.

    Raises:
        ValueError: If provider name is invalid.
    """
    # Resolve provider
    provider_name = provider or os.environ.get("LLM_PROVIDER", DEFAULT_PROVIDER)
    provider_name = provider_name.lower()

    # Resolve model
    model_name = model or os.environ.get("LLM_MODEL")

    # Resolve base_url (for openai provider)
    resolved_base_url = base_url or os.environ.get("LLM_BASE_URL")

    if provider_name == "anthropic":
        instance = AnthropicProvider(model=model_name)
    elif provider_name == "openai":
        instance = OpenAIProvider(model=model_name, base_url=resolved_base_url)
    elif provider_name == "gemini":
        instance = GeminiProvider(model=model_name)
    else:
        valid_providers = ["anthropic", "openai", "gemini"]
        raise ValueError(
            f"Unknown provider: {provider_name}. Valid providers: {valid_providers}"
        )

    # Log provider info once
    global _logged_provider
    if not _logged_provider:
        print(f"LLM: {provider_name} / {instance.model}", file=sys.stderr)
        _logged_provider = True

    return instance
