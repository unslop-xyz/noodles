"""HuggingFace Inference LLM provider implementation.

Uses the HuggingFace OpenAI-compatible router endpoint to access
Kimi K2 and other open-source models on HuggingFace.
"""

import os

import openai

from .pricing import calculate_cost
from .provider import LLMResponse

DEFAULT_MODEL = "moonshotai/Kimi-K2-Instruct"
DEFAULT_BASE_URL = "https://router.huggingface.co/v1"

_client: openai.AsyncOpenAI | None = None


def _get_client() -> openai.AsyncOpenAI:
    """Lazily create a shared async client for HuggingFace Inference API."""
    global _client

    if _client is None:
        api_key = os.environ.get("HF_TOKEN")
        if not api_key:
            raise ValueError(
                "HF_TOKEN environment variable is required for HuggingFace provider"
            )
        _client = openai.AsyncOpenAI(
            api_key=api_key,
            base_url=DEFAULT_BASE_URL,
        )
    return _client


class HuggingFaceProvider:
    """HuggingFace Inference LLM provider."""

    def __init__(self, model: str | None = None):
        """Initialize the provider.

        Args:
            model: Model name to use (default: moonshotai/Kimi-K2-Instruct)
        """
        self._model = model or DEFAULT_MODEL

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
        client = _get_client()
        response = await client.chat.completions.create(
            model=self._model,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )

        text = response.choices[0].message.content or ""
        input_tokens = response.usage.prompt_tokens if response.usage else 0
        output_tokens = response.usage.completion_tokens if response.usage else 0
        cost_usd = calculate_cost(self._model, input_tokens, output_tokens)

        return LLMResponse(
            text=text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd,
            model=self._model,
        )
