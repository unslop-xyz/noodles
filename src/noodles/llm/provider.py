"""LLM provider protocol and response dataclass."""

import asyncio
from dataclasses import dataclass
from typing import Protocol

MAX_RETRIES = 3
INITIAL_BACKOFF_SECONDS = 2.0


class LLMProviderError(Exception):
    """Exception raised when an LLM provider operation fails.

    Wraps underlying API errors with user-friendly messages.
    """

    def __init__(
        self,
        message: str,
        provider: str,
        cause: Exception | None = None,
        retryable: bool = False,
    ):
        """Initialize the error.

        Args:
            message: User-friendly error message.
            provider: Name of the provider that failed.
            cause: The underlying exception that caused this error.
            retryable: Whether this error can be retried (e.g., rate limits).
        """
        self.provider = provider
        self.cause = cause
        self.retryable = retryable
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


class RetryingProvider:
    """Wrapper that adds exponential backoff retry to any LLM provider."""

    def __init__(self, provider: LLMProvider):
        """Wrap a provider with retry logic.

        Args:
            provider: The underlying LLM provider to wrap.
        """
        self._provider = provider

    @property
    def model(self) -> str:
        """Return the model name from the underlying provider."""
        return self._provider.model

    async def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        """Complete a prompt with exponential backoff retry on retryable errors."""
        last_error: LLMProviderError | None = None

        for attempt in range(MAX_RETRIES + 1):
            try:
                return await self._provider.complete(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    max_tokens=max_tokens,
                )
            except LLMProviderError as e:
                if not e.retryable or attempt >= MAX_RETRIES:
                    raise
                last_error = e
                backoff = INITIAL_BACKOFF_SECONDS * (2**attempt)
                await asyncio.sleep(backoff)

        # Should not reach here, but handle just in case
        raise last_error or LLMProviderError(
            f"Failed after {MAX_RETRIES + 1} attempts",
            "unknown",
        )


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
