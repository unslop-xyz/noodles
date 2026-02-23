"""Pricing information for LLM models.

Prices are in USD per million tokens.
"""

from dataclasses import dataclass


@dataclass
class ModelPricing:
    """Pricing for input and output tokens."""

    input_per_million: float
    output_per_million: float

    def calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate total cost in USD."""
        return (
            input_tokens * self.input_per_million + output_tokens * self.output_per_million
        ) / 1_000_000


# Anthropic models
ANTHROPIC_PRICING: dict[str, ModelPricing] = {
    # Claude 4.5 (Nov 2025)
    "claude-opus-4-5-20251101": ModelPricing(5.00, 25.00),
    "claude-sonnet-4-5-20251101": ModelPricing(3.00, 15.00),
    "claude-haiku-4-5-20251001": ModelPricing(1.00, 5.00),
    # Claude 4 (May 2025)
    "claude-sonnet-4-20250514": ModelPricing(3.00, 15.00),
    "claude-opus-4-20250514": ModelPricing(15.00, 75.00),
    # Claude 3.5
    "claude-3-5-sonnet-20241022": ModelPricing(3.00, 15.00),
    "claude-3-5-haiku-20241022": ModelPricing(0.80, 4.00),
}

# OpenAI models
OPENAI_PRICING: dict[str, ModelPricing] = {
    # GPT-4o
    "gpt-4o": ModelPricing(2.50, 10.00),
    "gpt-4o-mini": ModelPricing(0.15, 0.60),
    # o-series reasoning models
    "o1": ModelPricing(10.00, 40.00),
    "o3": ModelPricing(2.00, 8.00),
    "o3-mini": ModelPricing(0.55, 2.20),
    "o4-mini": ModelPricing(1.10, 4.40),
}

# Google Gemini models
GEMINI_PRICING: dict[str, ModelPricing] = {
    # Gemini 2.5
    "gemini-2.5-pro": ModelPricing(1.25, 10.00),
    "gemini-2.5-flash": ModelPricing(0.15, 0.60),
    # Gemini 2.0
    "gemini-2.0-flash": ModelPricing(0.10, 0.40),
    "gemini-2.0-flash-lite": ModelPricing(0.10, 0.40),
}

# Groq models
GROQ_PRICING: dict[str, ModelPricing] = {
    # Llama models
    "llama-3.1-8b-instant": ModelPricing(0.05, 0.08),
    "llama-3.3-70b-versatile": ModelPricing(0.59, 0.79),
    "llama-4-scout-17b-16e-instruct": ModelPricing(0.11, 0.34),
    # Qwen
    "qwen-qwq-32b": ModelPricing(0.29, 0.39),
    # DeepSeek
    "deepseek-r1-distill-llama-70b": ModelPricing(0.75, 0.99),
}

# Combined pricing lookup
ALL_PRICING: dict[str, ModelPricing] = {
    **ANTHROPIC_PRICING,
    **OPENAI_PRICING,
    **GEMINI_PRICING,
    **GROQ_PRICING,
}

# Default pricing for unknown models (conservative estimate)
DEFAULT_PRICING = ModelPricing(1.00, 4.00)


def get_pricing(model: str) -> ModelPricing:
    """Get pricing for a model, with fallback to default."""
    return ALL_PRICING.get(model, DEFAULT_PRICING)


def calculate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Calculate cost for a model and token counts."""
    pricing = get_pricing(model)
    return pricing.calculate_cost(input_tokens, output_tokens)
