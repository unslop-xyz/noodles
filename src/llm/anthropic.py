"""Anthropic LLM provider implementation."""

import anthropic

from .pricing import calculate_cost
from .provider import LLMResponse

DEFAULT_MODEL = "claude-haiku-4-5-20251001"

_client: anthropic.AsyncAnthropic | None = None


def _get_client() -> anthropic.AsyncAnthropic:
    """Lazily create a shared async client."""
    global _client
    if _client is None:
        _client = anthropic.AsyncAnthropic()
    return _client


class AnthropicProvider:
    """Anthropic LLM provider."""

    def __init__(self, model: str | None = None):
        """Initialize the provider with an optional model override."""
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
        """Complete a prompt using Anthropic API."""
        client = _get_client()
        response = await client.messages.create(
            model=self._model,
            max_tokens=max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )

        text = "".join(
            block.text for block in response.content if block.type == "text"
        )
        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens
        cost_usd = calculate_cost(self._model, input_tokens, output_tokens)

        return LLMResponse(
            text=text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd,
            model=self._model,
        )
