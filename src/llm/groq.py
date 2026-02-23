"""Groq LLM provider implementation.

Uses Groq's OpenAI-compatible API for fast, affordable inference
on open source models like Llama and Mistral.
"""

import os

import openai

from .pricing import calculate_cost
from .provider import LLMResponse

DEFAULT_MODEL = "llama-3.3-70b-versatile"
DEFAULT_BASE_URL = "https://api.groq.com/openai/v1"

_client: openai.AsyncOpenAI | None = None


def _get_client() -> openai.AsyncOpenAI:
    """Lazily create a shared async client for Groq API."""
    global _client

    if _client is None:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise ValueError(
                "GROQ_API_KEY environment variable is required for Groq provider"
            )
        _client = openai.AsyncOpenAI(
            api_key=api_key,
            base_url=DEFAULT_BASE_URL,
        )
    return _client


class GroqProvider:
    """Groq LLM provider."""

    def __init__(self, model: str | None = None):
        """Initialize the provider.

        Args:
            model: Model name to use (default: llama-3.3-70b-versatile)
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
        """Complete a prompt using Groq API."""
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
