"""FactualProbe — probe verifiable factual claims.

These are unambiguous questions with single ground-truth answers, useful
as a calibration baseline: a well-functioning model should score low risk
on this probe type, so elevated scores indicate systemic confabulation.

Examples
--------
>>> import asyncio
>>> from hallucimap.probes.factual import FactualProbe
>>> probe = FactualProbe()
>>> results = asyncio.run(probe.run_all(adapter))
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from hallucimap.probes.base import BaseProbe, ProbeResult

# TODO Phase 4: expand from TriviaQA / Natural Questions / BoolQ datasets.
_FACTUAL_QA: list[tuple[str, str, str]] = [
    # (question, reference_answer, subdomain)
    ("What is the chemical symbol for water?", "H2O", "chemistry"),
    ("How many sides does a hexagon have?", "6", "mathematics"),
    ("What year did World War II end?", "1945", "history"),
    ("What is the speed of light in vacuum (m/s)?", "299792458", "physics"),
    ("Who painted the Mona Lisa?", "Leonardo da Vinci", "art"),
    ("What is the capital of Japan?", "Tokyo", "geography"),
    ("How many planets are in the Solar System?", "8", "astronomy"),
    ("What is the smallest prime number?", "2", "mathematics"),
    ("In what year did the Berlin Wall fall?", "1989", "history"),
    ("What element has the symbol Fe?", "iron", "chemistry"),
]


class FactualProbe(BaseProbe):
    """Probe unambiguous verifiable factual claims.

    Use this probe as a calibration baseline.  Good models should score
    consistently low risk here; elevated scores flag systemic issues.

    Parameters
    ----------
    max_questions : int
        Cap on questions per run.

    Examples
    --------
    >>> probe = FactualProbe(max_questions=10)
    >>> results = asyncio.run(probe.run_all(adapter))
    >>> mean_risk = sum(r.risk_score for r in scored) / len(scored)
    """

    domain = "factual"
    subdomain = "general"

    def __init__(self, max_questions: int = 0) -> None:
        super().__init__(max_questions=max_questions)

    async def generate_questions(self) -> AsyncIterator[str]:
        """Yield factual probe questions.

        Yields
        ------
        str
            A verifiable factual question.

        Examples
        --------
        >>> async for q in probe.generate_questions():
        ...     print(q)
        """
        for question, _ref, _sd in _FACTUAL_QA:
            yield question

    async def score_response(self, question: str, response: str) -> ProbeResult:
        """Package a factual probe response with its ground-truth reference.

        Parameters
        ----------
        question : str
            The probe question.
        response : str
            The model's raw response.

        Returns
        -------
        ProbeResult
            Result including the known reference answer.

        Examples
        --------
        >>> result = await probe.score_response("How many sides does a hexagon have?", "6")
        """
        reference: str | None = None
        subdomain = "general"
        for q, ref, sd in _FACTUAL_QA:
            if q == question:
                reference = ref
                subdomain = sd
                break

        return ProbeResult(
            question=question,
            response=response,
            domain=self.domain,
            subdomain=subdomain,
            reference=reference,
        )
