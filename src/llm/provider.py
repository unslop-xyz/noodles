"""LLM provider protocol and response dataclass."""

from dataclasses import dataclass
from typing import Protocol


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
