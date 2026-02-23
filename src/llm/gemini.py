"""Google Gemini LLM provider implementation."""

import os

import google.genai as genai
from google.genai import types

from .pricing import calculate_cost
from .provider import LLMProviderError, LLMResponse

DEFAULT_MODEL = "gemini-2.0-flash"

_client: genai.Client | None = None


def _get_client() -> genai.Client:
    """Lazily create a shared client."""
    global _client
    if _client is None:
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError(
                "GOOGLE_API_KEY environment variable is required for Gemini provider"
            )
        _client = genai.Client(api_key=api_key)
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

        try:
            response = await client.aio.models.generate_content(
                model=self._model,
                contents=user_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    max_output_tokens=max_tokens,
                ),
            )
        except genai.errors.AuthenticationError as e:
            raise LLMProviderError(
                "Authentication failed. Check your GOOGLE_API_KEY.",
                "Gemini",
                e,
            ) from e
        except genai.errors.ClientError as e:
            error_msg = str(e)
            if "429" in error_msg or "rate" in error_msg.lower():
                raise LLMProviderError(
                    "Rate limit exceeded. Please try again later.",
                    "Gemini",
                    e,
                    retryable=True,
                ) from e
            raise LLMProviderError(
                f"API error: {e}",
                "Gemini",
                e,
            ) from e
        except genai.errors.ServerError as e:
            raise LLMProviderError(
                f"Server error: {e}",
                "Gemini",
                e,
                retryable=True,
            ) from e
        except Exception as e:
            # Catch connection errors and other unexpected errors
            if "connect" in str(e).lower() or "timeout" in str(e).lower():
                raise LLMProviderError(
                    f"Failed to connect to Gemini API: {e}",
                    "Gemini",
                    e,
                    retryable=True,
                ) from e
            raise

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
