"""Tests for the LLM provider abstraction."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llm import LLMResponse, calculate_cost, get_pricing, get_provider
from llm.anthropic import AnthropicProvider
from llm.gemini import GeminiProvider
from llm.groq import GroqProvider
from llm.huggingface import HuggingFaceProvider
from llm.openai import OpenAIProvider
from llm.pricing import DEFAULT_PRICING, ModelPricing


# ---------------------------------------------------------------------------
# Pricing tests
# ---------------------------------------------------------------------------


class TestPricing:
    """Tests for pricing calculation."""

    def test_calculate_cost_known_model(self):
        """Cost calculation for a known model uses its pricing."""
        # Claude Haiku 4.5: $1.00/M input, $5.00/M output
        cost = calculate_cost("claude-haiku-4-5-20251001", 1_000_000, 1_000_000)
        assert cost == pytest.approx(6.00)

    def test_calculate_cost_gpt4o_mini(self):
        """Cost calculation for GPT-4o-mini."""
        # GPT-4o-mini: $0.15/M input, $0.60/M output
        cost = calculate_cost("gpt-4o-mini", 1_000_000, 1_000_000)
        assert cost == pytest.approx(0.75)

    def test_calculate_cost_gemini_flash(self):
        """Cost calculation for Gemini 2.0 Flash."""
        # Gemini 2.0 Flash: $0.10/M input, $0.40/M output
        cost = calculate_cost("gemini-2.0-flash", 1_000_000, 1_000_000)
        assert cost == pytest.approx(0.50)

    def test_calculate_cost_groq_llama(self):
        """Cost calculation for Groq Llama 3.3 70B."""
        # Llama 3.3 70B: $0.59/M input, $0.79/M output
        cost = calculate_cost("llama-3.3-70b-versatile", 1_000_000, 1_000_000)
        assert cost == pytest.approx(1.38)

    def test_calculate_cost_huggingface_uses_default(self):
        """HuggingFace models use default pricing (provider pricing varies)."""
        # HuggingFace pricing depends on the inference provider, so we use default
        cost = calculate_cost("moonshotai/Kimi-K2-Instruct", 1_000_000, 1_000_000)
        expected = DEFAULT_PRICING.input_per_million + DEFAULT_PRICING.output_per_million
        assert cost == pytest.approx(expected)

    def test_calculate_cost_unknown_model_uses_default(self):
        """Unknown models use default pricing."""
        cost = calculate_cost("unknown-model-xyz", 1_000_000, 1_000_000)
        expected = (
            DEFAULT_PRICING.input_per_million + DEFAULT_PRICING.output_per_million
        )
        assert cost == pytest.approx(expected)

    def test_get_pricing_known_model(self):
        """Get pricing for a known model."""
        pricing = get_pricing("claude-sonnet-4-20250514")
        assert pricing.input_per_million == 3.00
        assert pricing.output_per_million == 15.00

    def test_get_pricing_unknown_model(self):
        """Get pricing for an unknown model returns default."""
        pricing = get_pricing("some-future-model")
        assert pricing == DEFAULT_PRICING

    def test_model_pricing_calculate_cost(self):
        """Test ModelPricing.calculate_cost method."""
        pricing = ModelPricing(1.00, 4.00)
        cost = pricing.calculate_cost(500_000, 250_000)
        # (500_000 * 1.00 + 250_000 * 4.00) / 1_000_000 = 1.5
        assert cost == pytest.approx(1.5)


# ---------------------------------------------------------------------------
# Provider factory tests
# ---------------------------------------------------------------------------


class TestProviderFactory:
    """Tests for get_provider factory function."""

    def test_default_provider_is_anthropic(self, monkeypatch):
        """Default provider should be Anthropic."""
        monkeypatch.delenv("LLM_PROVIDER", raising=False)
        monkeypatch.delenv("LLM_MODEL", raising=False)
        provider = get_provider()
        assert isinstance(provider, AnthropicProvider)

    def test_provider_from_env_var(self, monkeypatch):
        """Provider can be set via LLM_PROVIDER env var."""
        monkeypatch.setenv("LLM_PROVIDER", "openai")
        monkeypatch.delenv("LLM_MODEL", raising=False)
        provider = get_provider()
        assert isinstance(provider, OpenAIProvider)

    def test_model_from_env_var(self, monkeypatch):
        """Model can be set via LLM_MODEL env var."""
        monkeypatch.setenv("LLM_PROVIDER", "anthropic")
        monkeypatch.setenv("LLM_MODEL", "claude-sonnet-4-20250514")
        provider = get_provider()
        assert provider.model == "claude-sonnet-4-20250514"

    def test_explicit_provider_overrides_env(self, monkeypatch):
        """Explicit provider argument overrides env var."""
        monkeypatch.setenv("LLM_PROVIDER", "openai")
        provider = get_provider(provider="gemini")
        assert isinstance(provider, GeminiProvider)

    def test_explicit_model_overrides_env(self, monkeypatch):
        """Explicit model argument overrides env var."""
        monkeypatch.setenv("LLM_MODEL", "some-model")
        provider = get_provider(model="custom-model")
        assert provider.model == "custom-model"

    def test_base_url_for_openai_provider(self, monkeypatch):
        """OpenAI provider accepts base_url parameter."""
        monkeypatch.delenv("LLM_BASE_URL", raising=False)
        provider = get_provider(
            provider="openai", base_url="http://localhost:8080/v1"
        )
        assert isinstance(provider, OpenAIProvider)
        assert provider._base_url == "http://localhost:8080/v1"

    def test_base_url_from_env_var(self, monkeypatch):
        """Base URL can be set via LLM_BASE_URL env var."""
        monkeypatch.setenv("LLM_BASE_URL", "http://localhost:11434/v1")
        provider = get_provider(provider="openai")
        assert provider._base_url == "http://localhost:11434/v1"

    def test_invalid_provider_raises_error(self, monkeypatch):
        """Invalid provider name raises ValueError."""
        monkeypatch.delenv("LLM_PROVIDER", raising=False)
        with pytest.raises(ValueError, match="Unknown provider"):
            get_provider(provider="invalid_provider")

    def test_provider_name_case_insensitive(self, monkeypatch):
        """Provider names are case-insensitive."""
        monkeypatch.setenv("LLM_PROVIDER", "ANTHROPIC")
        provider = get_provider()
        assert isinstance(provider, AnthropicProvider)

    def test_gemini_provider_creation(self, monkeypatch):
        """Gemini provider can be created."""
        monkeypatch.delenv("LLM_MODEL", raising=False)
        provider = get_provider(provider="gemini")
        assert isinstance(provider, GeminiProvider)

    def test_groq_provider_creation(self, monkeypatch):
        """Groq provider can be created."""
        monkeypatch.delenv("LLM_MODEL", raising=False)
        monkeypatch.setenv("GROQ_API_KEY", "test-key")
        provider = get_provider(provider="groq")
        assert isinstance(provider, GroqProvider)
        assert provider.model == "llama-3.3-70b-versatile"

    def test_huggingface_provider_creation(self, monkeypatch):
        """HuggingFace provider can be created."""
        monkeypatch.delenv("LLM_MODEL", raising=False)
        monkeypatch.setenv("HF_TOKEN", "test-token")
        provider = get_provider(provider="huggingface")
        assert isinstance(provider, HuggingFaceProvider)
        assert provider.model == "moonshotai/Kimi-K2-Instruct"


# ---------------------------------------------------------------------------
# Mocked API call tests
# ---------------------------------------------------------------------------


class TestAnthropicProviderComplete:
    """Tests for AnthropicProvider.complete with mocked API."""

    @pytest.mark.asyncio
    async def test_complete_returns_response(self):
        """complete() should return LLMResponse with correct fields."""
        # Mock the anthropic client
        mock_response = MagicMock()
        mock_response.content = [MagicMock(type="text", text="Test response")]
        mock_response.usage.input_tokens = 100
        mock_response.usage.output_tokens = 50

        mock_client = MagicMock()
        mock_client.messages.create = AsyncMock(return_value=mock_response)

        with patch("llm.anthropic._get_client", return_value=mock_client):
            provider = AnthropicProvider(model="claude-haiku-4-5-20251001")
            response = await provider.complete(
                system_prompt="You are a helpful assistant.",
                user_prompt="Hello!",
            )

        assert isinstance(response, LLMResponse)
        assert response.text == "Test response"
        assert response.input_tokens == 100
        assert response.output_tokens == 50
        assert response.model == "claude-haiku-4-5-20251001"
        assert response.cost_usd > 0


class TestOpenAIProviderComplete:
    """Tests for OpenAIProvider.complete with mocked API."""

    @pytest.mark.asyncio
    async def test_complete_returns_response(self):
        """complete() should return LLMResponse with correct fields."""
        # Mock the openai client
        mock_message = MagicMock()
        mock_message.content = "Test response from OpenAI"

        mock_choice = MagicMock()
        mock_choice.message = mock_message

        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 100
        mock_usage.completion_tokens = 50

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage = mock_usage

        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        with patch("llm.openai._get_client", return_value=mock_client):
            provider = OpenAIProvider(model="gpt-4o-mini")
            response = await provider.complete(
                system_prompt="You are a helpful assistant.",
                user_prompt="Hello!",
            )

        assert isinstance(response, LLMResponse)
        assert response.text == "Test response from OpenAI"
        assert response.input_tokens == 100
        assert response.output_tokens == 50
        assert response.model == "gpt-4o-mini"
        assert response.cost_usd > 0

    @pytest.mark.asyncio
    async def test_complete_handles_none_usage(self):
        """complete() should handle None usage gracefully."""
        mock_message = MagicMock()
        mock_message.content = "Response"

        mock_choice = MagicMock()
        mock_choice.message = mock_message

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage = None

        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        with patch("llm.openai._get_client", return_value=mock_client):
            provider = OpenAIProvider()
            response = await provider.complete("System", "User")

        assert response.input_tokens == 0
        assert response.output_tokens == 0


class TestGeminiProviderComplete:
    """Tests for GeminiProvider.complete with mocked API."""

    @pytest.mark.asyncio
    async def test_complete_returns_response(self):
        """complete() should return LLMResponse with correct fields."""
        # Mock the usage metadata
        mock_usage = MagicMock()
        mock_usage.prompt_token_count = 100
        mock_usage.candidates_token_count = 50

        # Mock the response
        mock_response = MagicMock()
        mock_response.text = "Test response from Gemini"
        mock_response.usage_metadata = mock_usage

        # Mock the client
        mock_client = MagicMock()
        mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)

        with patch("llm.gemini._get_client", return_value=mock_client):
            provider = GeminiProvider(model="gemini-2.0-flash")
            response = await provider.complete(
                system_prompt="You are a helpful assistant.",
                user_prompt="Hello!",
            )

        assert isinstance(response, LLMResponse)
        assert response.text == "Test response from Gemini"
        assert response.input_tokens == 100
        assert response.output_tokens == 50
        assert response.model == "gemini-2.0-flash"
        assert response.cost_usd > 0

    @pytest.mark.asyncio
    async def test_complete_handles_none_usage_metadata(self):
        """complete() should handle None usage_metadata gracefully."""
        mock_response = MagicMock()
        mock_response.text = "Response"
        mock_response.usage_metadata = None

        mock_client = MagicMock()
        mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)

        with patch("llm.gemini._get_client", return_value=mock_client):
            provider = GeminiProvider()
            response = await provider.complete("System", "User")

        assert response.input_tokens == 0
        assert response.output_tokens == 0


class TestGroqProviderComplete:
    """Tests for GroqProvider.complete with mocked API."""

    @pytest.mark.asyncio
    async def test_complete_returns_response(self):
        """complete() should return LLMResponse with correct fields."""
        # Mock the openai client (Groq uses OpenAI-compatible API)
        mock_message = MagicMock()
        mock_message.content = "Test response from Groq"

        mock_choice = MagicMock()
        mock_choice.message = mock_message

        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 100
        mock_usage.completion_tokens = 50

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage = mock_usage

        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        with patch("llm.groq._get_client", return_value=mock_client):
            provider = GroqProvider(model="llama-3.3-70b-versatile")
            response = await provider.complete(
                system_prompt="You are a helpful assistant.",
                user_prompt="Hello!",
            )

        assert isinstance(response, LLMResponse)
        assert response.text == "Test response from Groq"
        assert response.input_tokens == 100
        assert response.output_tokens == 50
        assert response.model == "llama-3.3-70b-versatile"
        assert response.cost_usd > 0


class TestHuggingFaceProviderComplete:
    """Tests for HuggingFaceProvider.complete with mocked API."""

    @pytest.mark.asyncio
    async def test_complete_returns_response(self):
        """complete() should return LLMResponse with correct fields."""
        # Mock the openai client (HuggingFace uses OpenAI-compatible API)
        mock_message = MagicMock()
        mock_message.content = "Test response from HuggingFace"

        mock_choice = MagicMock()
        mock_choice.message = mock_message

        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 100
        mock_usage.completion_tokens = 50

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage = mock_usage

        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        with patch("llm.huggingface._get_client", return_value=mock_client):
            provider = HuggingFaceProvider(model="moonshotai/Kimi-K2-Instruct")
            response = await provider.complete(
                system_prompt="You are a helpful assistant.",
                user_prompt="Hello!",
            )

        assert isinstance(response, LLMResponse)
        assert response.text == "Test response from HuggingFace"
        assert response.input_tokens == 100
        assert response.output_tokens == 50
        assert response.model == "moonshotai/Kimi-K2-Instruct"
        assert response.cost_usd > 0
