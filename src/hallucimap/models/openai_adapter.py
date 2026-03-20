"""OpenAI adapter for hallucimap.

Wraps the ``openai`` async client with retry logic via ``tenacity``.

Examples
--------
>>> import asyncio
>>> from hallucimap.models.openai_adapter import OpenAIAdapter
>>>
>>> adapter = OpenAIAdapter(model="gpt-4o")
>>> text = asyncio.run(adapter.complete("What is the capital of France?"))
>>> print(text)
Paris
"""

from __future__ import annotations

import os

from tenacity import retry, stop_after_attempt, wait_exponential

from hallucimap.models.base import BaseLLMAdapter, CompletionRequest, CompletionResponse

try:
    from openai import AsyncOpenAI
except ImportError as e:
    raise ImportError("openai package required: pip install openai") from e


class OpenAIAdapter(BaseLLMAdapter):
    """Async adapter for OpenAI chat completion models.

    Parameters
    ----------
    model : str
        OpenAI model name, e.g. ``"gpt-4o"`` or ``"gpt-4-turbo"``.
    api_key : str | None
        API key.  Falls back to ``OPENAI_API_KEY`` environment variable.
    base_url : str | None
        Custom base URL for API-compatible endpoints (e.g. Azure).
    max_retries : int
        Number of retries on transient errors.

    Examples
    --------
    >>> adapter = OpenAIAdapter(model="gpt-4o")
    >>> text = asyncio.run(adapter.complete("Explain the photoelectric effect."))
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: str | None = None,
        base_url: str | None = None,
        max_retries: int = 3,
    ) -> None:
        super().__init__(model_id=model)
        self._client = AsyncOpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY"),
            base_url=base_url,
        )
        self._max_retries = max_retries

    async def complete(
        self,
        prompt: str,
        *,
        temperature: float = 0.7,
        max_tokens: int = 512,
        system_prompt: str | None = None,
    ) -> str:
        """Generate a completion via the OpenAI chat API.

        Parameters
        ----------
        prompt : str
            The user prompt.
        temperature : float
            Sampling temperature.
        max_tokens : int
            Max tokens to generate.
        system_prompt : str | None
            Optional system message.

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
        messages: list[dict[str, str]] = []
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})
        messages.append({"role": "user", "content": request.prompt})

        # TODO Phase 2: add structured output mode for grounding checks
        raw = await self._client.chat.completions.create(
            model=self.model_id,
            messages=messages,  # type: ignore[arg-type]
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        )
        choice = raw.choices[0]
        return CompletionResponse(
            text=choice.message.content or "",
            model_id=self.model_id,
            prompt_tokens=raw.usage.prompt_tokens if raw.usage else None,
            completion_tokens=raw.usage.completion_tokens if raw.usage else None,
            finish_reason=choice.finish_reason,
        )
