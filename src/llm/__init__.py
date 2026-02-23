"""LLM provider abstraction for multiple AI providers.

Usage:
    from llm import get_provider, LLMResponse

    provider = get_provider()  # Uses env vars for configuration
    response = await provider.complete(system_prompt, user_prompt)
    print(response.text, response.cost_usd)

Environment variables:
    LLM_PROVIDER: Provider name (anthropic, openai, gemini, groq, huggingface). Default: anthropic
    LLM_MODEL: Model name override. Default: provider-specific default
    LLM_BASE_URL: Base URL for OpenAI-compatible endpoints (openai provider only)
"""

from .config import get_provider
from .pricing import calculate_cost, get_pricing
from .provider import LLMProvider, LLMResponse

__all__ = [
    "get_provider",
    "LLMProvider",
    "LLMResponse",
    "calculate_cost",
    "get_pricing",
]
