"""Anthropic adapter for hallucimap.

Wraps the ``anthropic`` async client with retry logic via ``tenacity``.

Examples
--------
>>> import asyncio
>>> from hallucimap.models.anthropic_adapter import AnthropicAdapter
>>>
>>> adapter = AnthropicAdapter(model="claude-3-5-sonnet-20241022")
>>> text = asyncio.run(adapter.complete("What is the capital of France?"))
>>> print(text)
Paris
"""

from __future__ import annotations

import os

from tenacity import retry, stop_after_attempt, wait_exponential

from hallucimap.models.base import BaseLLMAdapter, CompletionRequest, CompletionResponse

try:
    from anthropic import AsyncAnthropic
except ImportError as e:
    raise ImportError("anthropic package required: pip install anthropic") from e


class AnthropicAdapter(BaseLLMAdapter):
    """Async adapter for Anthropic Claude models.

    Parameters
    ----------
    model : str
        Claude model identifier, e.g. ``"claude-3-5-sonnet-20241022"``.
    api_key : str | None
        API key.  Falls back to ``ANTHROPIC_API_KEY`` environment variable.
    max_tokens : int
        Default maximum tokens for completions.

    Examples
    --------
    >>> adapter = AnthropicAdapter(model="claude-3-5-sonnet-20241022")
    >>> text = asyncio.run(adapter.complete("Explain photosynthesis briefly."))
    """

    def __init__(
        self,
        model: str = "claude-3-5-sonnet-20241022",
        api_key: str | None = None,
        max_tokens: int = 512,
    ) -> None:
        super().__init__(model_id=model)
        self._client = AsyncAnthropic(
            api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"),
        )
        self._default_max_tokens = max_tokens

    async def complete(
        self,
        prompt: str,
        *,
        temperature: float = 0.7,
        max_tokens: int = 512,
        system_prompt: str | None = None,
    ) -> str:
        """Generate a completion via the Anthropic Messages API.

        Parameters
        ----------
        prompt : str
            The user prompt.
        temperature : float
            Sampling temperature.
        max_tokens : int
            Max tokens to generate.
        system_prompt : str | None
            Optional system message prepended to the conversation.

        Returns
        -------
        str
            Generated text.

        Examples
        --------
        >>> text = asyncio.run(adapter.complete("What year did WWII end?"))
        >>> "1945" in text
        True
        """
        resp = await self.complete_structured(
            CompletionRequest(
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                system_prompt=system_prompt,
            )
        )
        return resp.text

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    async def complete_structured(self, request: CompletionRequest) -> CompletionResponse:
        """Generate a structured completion with retry logic.

        Parameters
        ----------
        request : CompletionRequest
            Fully specified request parameters.

        Returns
        -------
        CompletionResponse
            Response including token counts and finish reason.

        Examples
        --------
        >>> req = CompletionRequest(prompt="Hello", temperature=0.5)
        >>> resp = asyncio.run(adapter.complete_structured(req))
        """
        kwargs: dict = dict(
            model=self.model_id,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            messages=[{"role": "user", "content": request.prompt}],
        )
        if request.system_prompt:
            kwargs["system"] = request.system_prompt

        # TODO Phase 2: add tool-use for structured grounding checks
        raw = await self._client.messages.create(**kwargs)

        text = ""
        if raw.content:
            block = raw.content[0]
            if hasattr(block, "text"):
                text = block.text

        return CompletionResponse(
            text=text,
            model_id=self.model_id,
            prompt_tokens=raw.usage.input_tokens if raw.usage else None,
            completion_tokens=raw.usage.output_tokens if raw.usage else None,
            finish_reason=raw.stop_reason,
        )
