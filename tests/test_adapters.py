"""Tests for LLM adapters (using mocked HTTP clients)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from hallucimap.models.base import CompletionRequest, CompletionResponse
from hallucimap.testing import MockAdapter


# ------------------------------------------------------------------ #
# BaseLLMAdapter / MockAdapter                                         #
# ------------------------------------------------------------------ #


class TestMockAdapter:
    @pytest.mark.asyncio
    async def test_complete_returns_string(self) -> None:
        adapter = MockAdapter(response="hello")
        result = await adapter.complete("test prompt")
        assert result == "hello"

    @pytest.mark.asyncio
    async def test_complete_structured_returns_completion_response(self) -> None:
        adapter = MockAdapter(response="world")
        resp = await adapter.complete_structured(CompletionRequest(prompt="test"))
        assert isinstance(resp, CompletionResponse)
        assert resp.text == "world"
        assert resp.model_id == "mock-model-v1"

    @pytest.mark.asyncio
    async def test_model_id_set(self) -> None:
        adapter = MockAdapter()
        assert adapter.model_id == "mock-model-v1"


# ------------------------------------------------------------------ #
# OpenAIAdapter                                                        #
# ------------------------------------------------------------------ #


class TestOpenAIAdapter:
    @pytest.mark.asyncio
    async def test_complete_delegates_to_openai_client(self) -> None:
        from hallucimap.models.openai_adapter import OpenAIAdapter

        mock_choice = MagicMock()
        mock_choice.message.content = "The capital is Paris."
        mock_choice.finish_reason = "stop"

        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 10
        mock_usage.completion_tokens = 5

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage = mock_usage

        with patch("hallucimap.models.openai_adapter.AsyncOpenAI") as MockOpenAI:
            instance = MockOpenAI.return_value
            instance.chat.completions.create = AsyncMock(return_value=mock_response)

            adapter = OpenAIAdapter(model="gpt-4o", api_key="test-key")
            adapter._client = instance
            text = await adapter.complete("What is the capital of France?")

        assert text == "The capital is Paris."

    @pytest.mark.asyncio
    async def test_complete_structured_populates_token_counts(self) -> None:
        from hallucimap.models.openai_adapter import OpenAIAdapter

        mock_choice = MagicMock()
        mock_choice.message.content = "42"
        mock_choice.finish_reason = "stop"

        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 8
        mock_usage.completion_tokens = 2

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage = mock_usage

        with patch("hallucimap.models.openai_adapter.AsyncOpenAI") as MockOpenAI:
            instance = MockOpenAI.return_value
            instance.chat.completions.create = AsyncMock(return_value=mock_response)

            adapter = OpenAIAdapter(model="gpt-4o", api_key="test-key")
            adapter._client = instance
            resp = await adapter.complete_structured(CompletionRequest(prompt="What is 6*7?"))

        assert resp.prompt_tokens == 8
        assert resp.completion_tokens == 2
        assert resp.finish_reason == "stop"


# ------------------------------------------------------------------ #
# AnthropicAdapter                                                     #
# ------------------------------------------------------------------ #


class TestAnthropicAdapter:
    @pytest.mark.asyncio
    async def test_complete_extracts_text_from_content_block(self) -> None:
        from hallucimap.models.anthropic_adapter import AnthropicAdapter

        mock_block = MagicMock()
        mock_block.text = "London"

        mock_usage = MagicMock()
        mock_usage.input_tokens = 12
        mock_usage.output_tokens = 3

        mock_response = MagicMock()
        mock_response.content = [mock_block]
        mock_response.usage = mock_usage
        mock_response.stop_reason = "end_turn"

        with patch("hallucimap.models.anthropic_adapter.AsyncAnthropic") as MockAnthropic:
            instance = MockAnthropic.return_value
            instance.messages.create = AsyncMock(return_value=mock_response)

            adapter = AnthropicAdapter(
                model="claude-3-5-sonnet-20241022", api_key="test-key"
            )
            adapter._client = instance
            text = await adapter.complete("Capital of UK?")

        assert text == "London"

    @pytest.mark.asyncio
    async def test_system_prompt_passed_as_keyword(self) -> None:
        from hallucimap.models.anthropic_adapter import AnthropicAdapter

        mock_block = MagicMock()
        mock_block.text = "OK"

        mock_response = MagicMock()
        mock_response.content = [mock_block]
        mock_response.usage = MagicMock(input_tokens=5, output_tokens=1)
        mock_response.stop_reason = "end_turn"

        with patch("hallucimap.models.anthropic_adapter.AsyncAnthropic") as MockAnthropic:
            instance = MockAnthropic.return_value
            create_mock = AsyncMock(return_value=mock_response)
            instance.messages.create = create_mock

            adapter = AnthropicAdapter(model="claude-3-5-sonnet-20241022", api_key="test-key")
            adapter._client = instance
            await adapter.complete("Hello", system_prompt="Be helpful")

        call_kwargs = create_mock.call_args.kwargs
        assert "system" in call_kwargs
        assert call_kwargs["system"] == "Be helpful"
