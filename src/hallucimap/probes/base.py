"""BaseProbe — abstract base class for all probe types.

Every probe must:
1. Declare a ``domain`` and ``subdomain``.
2. Implement :meth:`generate_questions` as an async generator.
3. Implement :meth:`score_response` to produce a :class:`ProbeResult`.

The :meth:`run_all` convenience method streams all questions through the
adapter and returns a list of :class:`ProbeResult` objects ready for the
scorer.

Examples
--------
>>> class MyProbe(BaseProbe):
...     domain = "science"
...     subdomain = "physics"
...
...     async def generate_questions(self):
...         yield "What is the speed of light?"
...
...     async def score_response(self, question, response):
...         return ProbeResult(question=question, response=response,
...                            domain=self.domain, subdomain=self.subdomain)
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

from pydantic import BaseModel

if TYPE_CHECKING:
    from hallucimap.models.base import BaseLLMAdapter


class ProbeResult(BaseModel):
    """Raw (un-scored) probe result.

    Parameters
    ----------
    question : str
        The question that was asked.
    response : str
        The model's raw response.
    domain : str
        Top-level domain label.
    subdomain : str
        Fine-grained subdomain label.
    reference : str | None
        Ground-truth answer if known, else ``None``.

    Examples
    --------
    >>> result = ProbeResult(
    ...     question="What is H2O?",
    ...     response="Water",
    ...     domain="science",
    ...     subdomain="chemistry",
    ... )
    """

    question: str
    response: str
    domain: str
    subdomain: str
    reference: str | None = None


class BaseProbe(ABC):
    """Abstract base class for all hallucimap probes.

    Subclasses must set the class-level ``domain`` and ``subdomain``
    attributes and implement :meth:`generate_questions` and
    :meth:`score_response`.

    Parameters
    ----------
    max_questions : int
        Cap on how many questions to generate per run.  ``0`` means no limit.

    Examples
    --------
    >>> probe = MyProbe(max_questions=20)
    >>> results = asyncio.run(probe.run_all(adapter))
    """

    domain: str = "generic"
    subdomain: str = "general"

    def __init__(self, max_questions: int = 0) -> None:
        self.max_questions = max_questions

    @abstractmethod
    async def generate_questions(self) -> AsyncIterator[str]:
        """Yield probe questions one by one.

        Yields
        ------
        str
            A single probe question string.

        Examples
        --------
        >>> async for q in probe.generate_questions():
        ...     print(q)
        """
        # mypy requires this pattern for abstract async generators
        raise NotImplementedError
        # This line is unreachable but satisfies the return type annotation
        yield  # type: ignore[misc]

    @abstractmethod
    async def score_response(self, question: str, response: str) -> ProbeResult:
        """Parse a model response and return a :class:`ProbeResult`.

        Parameters
        ----------
        question : str
            The original probe question.
        response : str
            The model's raw response text.

        Returns
        -------
        ProbeResult
            Populated result including any reference answer if known.

        Examples
        --------
        >>> result = await probe.score_response("What is H2O?", "Water")
        """
        ...

    async def run_all(
        self,
        adapter: BaseLLMAdapter,
        concurrency: int = 5,
    ) -> list[ProbeResult]:
        """Run every probe question through the adapter and return results.

        Parameters
        ----------
        adapter : BaseLLMAdapter
            The LLM adapter to use for completions.
        concurrency : int
            Maximum number of simultaneous adapter calls.

        Returns
        -------
        list[ProbeResult]
            One result per question generated.

        Examples
        --------
        >>> results = await probe.run_all(adapter, concurrency=10)
        >>> len(results)
        20
        """
        sem = asyncio.Semaphore(concurrency)
        results: list[ProbeResult] = []
        count = 0

        async def _run_one(question: str) -> ProbeResult:
            async with sem:
                response = await adapter.complete(question)
                return await self.score_response(question, response)

        tasks = []
        async for question in self.generate_questions():
            if self.max_questions > 0 and count >= self.max_questions:
                break
            tasks.append(asyncio.create_task(_run_one(question)))
            count += 1

        results = list(await asyncio.gather(*tasks))
        return results
