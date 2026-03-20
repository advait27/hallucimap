"""Test helpers for hallucimap — usable in downstream tests too.

Provides :class:`MockAdapter`, a deterministic :class:`BaseLLMAdapter`
that always returns a fixed response without making any network calls.

Examples
--------
>>> from hallucimap.testing import MockAdapter
>>> adapter = MockAdapter(response="42")
>>> import asyncio
>>> asyncio.run(adapter.complete("What is 6 * 7?"))
'42'
"""

from __future__ import annotations

from hallucimap.models.base import BaseLLMAdapter, CompletionRequest, CompletionResponse


class MockAdapter(BaseLLMAdapter):
    """Deterministic LLM adapter for unit tests.

    Parameters
    ----------
    response : str
        Fixed text returned for every completion request.

    Examples
    --------
    >>> adapter = MockAdapter(response="Paris")
    >>> import asyncio
    >>> asyncio.run(adapter.complete("Capital of France?"))
    'Paris'
    """

    def __init__(self, response: str = "Paris") -> None:
        super().__init__(model_id="mock-model-v1")
        self._response = response

    async def complete(
        self,
        prompt: str,
        *,
        temperature: float = 0.7,
        max_tokens: int = 512,
        system_prompt: str | None = None,
    ) -> str:
        return self._response

    async def complete_structured(self, request: CompletionRequest) -> CompletionResponse:
        return CompletionResponse(
            text=self._response,
            model_id=self.model_id,
            prompt_tokens=10,
            completion_tokens=5,
            finish_reason="stop",
        )
