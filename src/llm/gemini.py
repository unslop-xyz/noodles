"""Google Gemini LLM provider implementation."""

import google.genai as genai
from google.genai import types

from .pricing import calculate_cost
from .provider import LLMResponse

DEFAULT_MODEL = "gemini-2.0-flash"

_client: genai.Client | None = None


def _get_client() -> genai.Client:
    """Lazily create a shared client."""
    global _client
    if _client is None:
        _client = genai.Client()
    return _client


class GeminiProvider:
    """Google Gemini LLM provider."""

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
        """Complete a prompt using Gemini API."""
        client = _get_client()

        response = await client.aio.models.generate_content(
            model=self._model,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                max_output_tokens=max_tokens,
            ),
        )

        text = response.text or ""

        # Extract token counts from usage metadata
        input_tokens = 0
        output_tokens = 0
        if response.usage_metadata:
            input_tokens = response.usage_metadata.prompt_token_count or 0
            output_tokens = response.usage_metadata.candidates_token_count or 0

        cost_usd = calculate_cost(self._model, input_tokens, output_tokens)

        return LLMResponse(
            text=text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd,
            model=self._model,
        )
