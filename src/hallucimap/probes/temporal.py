"""TemporalProbe — probe post-cutoff date facts.

Tests the model's knowledge of events that occurred after its training
cutoff.  High risk scores here indicate the model is confabulating
"recent" events it cannot actually know.

Examples
--------
>>> import asyncio
>>> from hallucimap.probes.temporal import TemporalProbe
>>> from hallucimap.models import OpenAIAdapter
>>>
>>> adapter = OpenAIAdapter(model="gpt-4o")
>>> probe = TemporalProbe(cutoff_year=2024)
>>> results = asyncio.run(probe.run_all(adapter))
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from hallucimap.probes.base import BaseProbe, ProbeResult

# TODO Phase 4: load this from a curated dataset of post-cutoff events.
_POST_CUTOFF_TEMPLATES: list[str] = [
    "Who won the Nobel Prize in {field} in {year}?",
    "What was the outcome of the {year} {event} election?",
    "What major AI models were released in {year}?",
    "Who is the current president/prime minister of {country} as of {year}?",
    "What were the biggest technology acquisitions of {year}?",
]


class TemporalProbe(BaseProbe):
    """Probe post-training-cutoff factual knowledge.

    Generates questions about events that post-date the model's training
    cutoff.  A well-calibrated model should hedge rather than confabulate.

    Parameters
    ----------
    cutoff_year : int
        The year after which events are considered post-cutoff.
    target_years : list[int] | None
        Specific years to probe.  Defaults to ``[cutoff_year, cutoff_year + 1]``.
    max_questions : int
        Cap on questions per run.

    Examples
    --------
    >>> probe = TemporalProbe(cutoff_year=2023, max_questions=10)
    >>> results = asyncio.run(probe.run_all(adapter))
    """

    domain = "temporal"
    subdomain = "post_cutoff"

    def __init__(
        self,
        cutoff_year: int = 2023,
        target_years: list[int] | None = None,
        max_questions: int = 0,
    ) -> None:
        super().__init__(max_questions=max_questions)
        self.cutoff_year = cutoff_year
        self.target_years = target_years or [cutoff_year, cutoff_year + 1]

    async def generate_questions(self) -> AsyncIterator[str]:
        """Yield temporal probe questions.

        Yields
        ------
        str
            A question about a post-cutoff event.

        Examples
        --------
        >>> async for q in probe.generate_questions():
        ...     print(q)
        """
        # TODO Phase 4: expand from a rich curated corpus of post-cutoff events.
        for year in self.target_years:
            for template in _POST_CUTOFF_TEMPLATES:
                # Simple placeholder fill — real version queries a dataset
                question = template.format(
                    year=year,
                    field="Physics",
                    event="US Presidential",
                    country="the United States",
                )
                yield question

    async def score_response(self, question: str, response: str) -> ProbeResult:
        """Parse and package a temporal probe response.

        Parameters
        ----------
        question : str
            The probe question.
        response : str
            The model's raw response.

        Returns
        -------
        ProbeResult
            Result with ``domain="temporal"`` and ``subdomain="post_cutoff"``.

        Examples
        --------
        >>> result = await probe.score_response("Who won Nobel 2024?", "I don't know.")
        """
        # TODO Phase 4: cross-reference response against a known-facts database.
        return ProbeResult(
            question=question,
            response=response,
            domain=self.domain,
            subdomain=self.subdomain,
            reference=None,  # TODO Phase 4: fill from curated dataset
        )
