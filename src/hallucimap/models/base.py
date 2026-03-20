"""BaseLLMAdapter — the abstract interface every model adapter must implement.

All adapters must expose an async :meth:`complete` method and carry a
``model_id`` string.  The scorer and probes depend only on this interface,
so adapters are fully swappable.

Examples
--------
>>> class MyAdapter(BaseLLMAdapter):
...     model_id = "my-model-v1"
...
...     async def complete(self, prompt, *, temperature=0.7, max_tokens=512):
...         return "mocked response"
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from pydantic import BaseModel


class CompletionRequest(BaseModel):
    """Structured request to an LLM.

    Parameters
    ----------
    prompt : str
        The user/system prompt to send.
    temperature : float
        Sampling temperature.
    max_tokens : int
        Maximum tokens to generate.
    system_prompt : str | None
        Optional system-level instruction.

    Examples
    --------
    >>> req = CompletionRequest(prompt="Hello", temperature=0.5)
    """

    prompt: str
    temperature: float = 0.7
    max_tokens: int = 512
    system_prompt: str | None = None


class CompletionResponse(BaseModel):
    """Structured response from an LLM.

    Parameters
    ----------
    text : str
        The generated text.
    model_id : str
        The model that produced this response.
    prompt_tokens : int | None
        Number of tokens in the prompt, if reported.
    completion_tokens : int | None
        Number of tokens generated, if reported.
    finish_reason : str | None
        Stop reason (e.g. ``"stop"``, ``"length"``).

    Examples
    --------
    >>> resp = CompletionResponse(text="Paris", model_id="gpt-4o")
    """

    text: str
    model_id: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    finish_reason: str | None = None


class BaseLLMAdapter(ABC):
    """Abstract base class for all LLM adapters.

    Parameters
    ----------
    model_id : str
        The identifier of the model being wrapped.

    Examples
    --------
    >>> adapter = MyAdapter(model_id="custom-v1")
    >>> text = await adapter.complete("What is the capital of France?")
    >>> print(text)
    Paris
    """

    def __init__(self, model_id: str) -> None:
        self.model_id = model_id

    @abstractmethod
    async def complete(
        self,
        prompt: str,
        *,
        temperature: float = 0.7,
        max_tokens: int = 512,
        system_prompt: str | None = None,
    ) -> str:
        """Generate a single completion for the given prompt.

        Parameters
        ----------
        prompt : str
            The prompt to complete.
        temperature : float
            Sampling temperature (``0`` = greedy).
        max_tokens : int
            Maximum tokens in the completion.
        system_prompt : str | None
            Optional system-level instruction.

        Returns
        -------
        str
            Raw completion text.

        Examples
        --------
        >>> text = await adapter.complete("Who wrote Hamlet?")
        >>> "Shakespeare" in text
        True
        """
        ...

    @abstractmethod
    async def complete_structured(self, request: CompletionRequest) -> CompletionResponse:
        """Generate a completion from a :class:`CompletionRequest`.

        Parameters
        ----------
        request : CompletionRequest
            Fully specified completion request.

        Returns
        -------
        CompletionResponse
            Structured response including token counts and finish reason.

        Examples
        --------
        >>> req = CompletionRequest(prompt="Hello", temperature=0.5)
        >>> resp = await adapter.complete_structured(req)
        >>> resp.text
        'Hello! How can I help you today?'
        """
        ...
