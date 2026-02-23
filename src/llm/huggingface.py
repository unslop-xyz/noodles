"""HuggingFace Inference LLM provider implementation.

Uses the HuggingFace OpenAI-compatible router endpoint to access
Kimi K2 and other open-source models on HuggingFace.
"""

import os

import openai

from .pricing import calculate_cost
from .provider import LLMProviderError, LLMResponse, extract_openai_usage

DEFAULT_MODEL = "moonshotai/Kimi-K2-Instruct"
DEFAULT_BASE_URL = "https://router.huggingface.co/v1"

_client: openai.AsyncOpenAI | None = None
_client_base_url: str | None = None


def _get_client(base_url: str | None = None) -> openai.AsyncOpenAI:
    """Lazily create a shared async client for HuggingFace Inference API."""
    global _client, _client_base_url

    effective_base_url = base_url or DEFAULT_BASE_URL

    # Recreate client if base_url changed
    if _client is None or _client_base_url != effective_base_url:
        api_key = os.environ.get("HF_TOKEN")
        if not api_key:
            raise ValueError(
                "HF_TOKEN environment variable is required for HuggingFace provider"
            )
        _client = openai.AsyncOpenAI(
            api_key=api_key,
            base_url=effective_base_url,
        )
        _client_base_url = effective_base_url
    return _client


class HuggingFaceProvider:
    """HuggingFace Inference LLM provider."""

    def __init__(self, model: str | None = None, base_url: str | None = None):
        """Initialize the provider.

        Args:
            model: Model name to use (default: moonshotai/Kimi-K2-Instruct)
            base_url: Optional base URL for HuggingFace-compatible endpoints.
                      Can also be set via HF_BASE_URL env var.
        """
        self._model = model or DEFAULT_MODEL
        self._base_url = base_url or os.environ.get("HF_BASE_URL")

    @property
    def model(self) -> str:
        """Return the model name."""
        return self._model

    async def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        """Complete a prompt using HuggingFace Inference API."""
        client = _get_client(self._base_url)

        try:
            response = await client.chat.completions.create(
                model=self._model,
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
        except openai.AuthenticationError as e:
            raise LLMProviderError(
                "Authentication failed. Check your HF_TOKEN.",
                "HuggingFace",
                e,
            ) from e
        except openai.RateLimitError as e:
            raise LLMProviderError(
                "Rate limit exceeded. Please try again later.",
                "HuggingFace",
                e,
                retryable=True,
            ) from e
        except openai.APIConnectionError as e:
            raise LLMProviderError(
                f"Failed to connect to HuggingFace API: {e}",
                "HuggingFace",
                e,
                retryable=True,
            ) from e
        except openai.APIError as e:
            raise LLMProviderError(
                f"API error: {e}",
                "HuggingFace",
                e,
            ) from e

        text = response.choices[0].message.content or ""
        input_tokens, output_tokens = extract_openai_usage(response)
        cost_usd = calculate_cost(self._model, input_tokens, output_tokens)

        return LLMResponse(
            text=text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd,
            model=self._model,
        )
