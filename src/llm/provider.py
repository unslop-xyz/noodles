"""LLM provider protocol and response dataclass."""

from dataclasses import dataclass
from typing import Protocol


class LLMProviderError(Exception):
    """Exception raised when an LLM provider operation fails.

    Wraps underlying API errors with user-friendly messages.
    """

    def __init__(self, message: str, provider: str, cause: Exception | None = None):
        """Initialize the error.

        Args:
            message: User-friendly error message.
            provider: Name of the provider that failed.
            cause: The underlying exception that caused this error.
        """
        self.provider = provider
        self.cause = cause
        super().__init__(f"{provider}: {message}")


@dataclass
class LLMResponse:
    """Response from an LLM provider."""

    text: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    model: str


class LLMProvider(Protocol):
    """Protocol for LLM providers."""

    @property
    def model(self) -> str:
        """Return the model name."""
        ...

    async def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        """Complete a prompt and return the response."""
        ...


def extract_openai_usage(response) -> tuple[int, int]:
    """Extract token counts from an OpenAI-compatible response.

    Works with responses from OpenAI, Groq, and HuggingFace providers.

    Args:
        response: An OpenAI-compatible chat completion response.

    Returns:
        Tuple of (input_tokens, output_tokens). Returns (0, 0) if usage
        information is not available.
    """
    if response.usage is None:
        return 0, 0
    return (
        response.usage.prompt_tokens or 0,
        response.usage.completion_tokens or 0,
    )
