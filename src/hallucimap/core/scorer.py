"""HallucinationScorer — consistency sampling + factual grounding.

The scorer asks the model the same question ``n_samples`` times with
non-zero temperature and measures *consistency* across answers.  Low
consistency implies the model is uncertain (high hallucination risk).
An optional *grounding* step cross-checks a sampled answer against a
verifiable fact source.

Examples
--------
>>> import asyncio
>>> from hallucimap.core.scorer import HallucinationScorer
>>> from hallucimap.models import OpenAIAdapter
>>>
>>> adapter = OpenAIAdapter(model="gpt-4o")
>>> scorer = HallucinationScorer(adapter=adapter, n_samples=5)
>>> result = asyncio.run(scorer.score("What is the capital of France?", "geography", "europe"))
>>> print(result.risk_score)  # ~0.05 — very consistent
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import numpy as np
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from hallucimap.models.base import BaseLLMAdapter


class ScoredResponse(BaseModel):
    """The output of :class:`HallucinationScorer` for a single question.

    Parameters
    ----------
    question : str
        The probe question that was asked.
    domain : str
        Top-level domain label.
    subdomain : str
        Fine-grained subdomain label.
    risk_score : float
        Hallucination risk estimate in ``[0, 1]``.  Higher = riskier.
    consistency_score : float
        Inter-sample consistency in ``[0, 1]``.  ``1 - consistency_score``
        contributes to ``risk_score``.
    grounding_score : float | None
        Factual grounding score ``[0, 1]`` if a ground-truth was available,
        else ``None``.
    samples : list[str]
        The raw responses collected during consistency sampling.
    model_id : str
        Which model produced these responses.

    Examples
    --------
    >>> resp = ScoredResponse(
    ...     question="Who wrote Hamlet?",
    ...     domain="literature",
    ...     subdomain="shakespeare",
    ...     risk_score=0.05,
    ...     consistency_score=0.95,
    ...     samples=["Shakespeare", "William Shakespeare", "Shakespeare"],
    ...     model_id="gpt-4o",
    ... )
    """

    question: str
    domain: str
    subdomain: str
    risk_score: float
    consistency_score: float
    grounding_score: float | None = None
    samples: list[str] = Field(default_factory=list)
    model_id: str = ""


class HallucinationScorer:
    """Score hallucination risk for a single (question, domain) pair.

    Strategy
    --------
    1. **Consistency sampling**: ask the same question ``n_samples`` times
       at ``temperature > 0``.  Compute pairwise semantic similarity across
       answers; low mean similarity → high inconsistency → higher risk.
    2. **Grounding** (optional): if a reference answer is provided, compare
       the modal sample against it; divergence raises risk further.
    3. **Composite score**: ``risk = α * (1 - consistency) + β * (1 - grounding)``
       where ``β = 0`` when no grounding is available.

    Parameters
    ----------
    adapter : BaseLLMAdapter
        Any model adapter (OpenAI, Anthropic, HuggingFace).
    n_samples : int
        Number of independent completions to draw per question.
    temperature : float
        Sampling temperature.  Must be ``> 0`` for consistency sampling.
    alpha : float
        Weight for consistency term in the composite risk score.
    beta : float
        Weight for grounding term in the composite risk score.

    Examples
    --------
    >>> scorer = HallucinationScorer(adapter=adapter, n_samples=5)
    >>> result = asyncio.run(scorer.score("What year did WWII end?", "history", "wwii"))
    >>> print(result.risk_score)
    """

    def __init__(
        self,
        adapter: BaseLLMAdapter,
        n_samples: int = 5,
        temperature: float = 0.9,
        alpha: float = 0.7,
        beta: float = 0.3,
    ) -> None:
        if temperature <= 0:
            raise ValueError("temperature must be > 0 for consistency sampling")
        self.adapter = adapter
        self.n_samples = n_samples
        self.temperature = temperature
        self.alpha = alpha
        self.beta = beta

    async def score(
        self,
        question: str,
        domain: str,
        subdomain: str,
        reference: str | None = None,
    ) -> ScoredResponse:
        """Score hallucination risk for one question.

        Parameters
        ----------
        question : str
            The probe question.
        domain : str
            Domain label for atlas bucketing.
        subdomain : str
            Subdomain label for atlas bucketing.
        reference : str | None
            Known ground-truth answer, used for grounding.  Optional.

        Returns
        -------
        ScoredResponse
            Fully populated risk score record.

        Examples
        --------
        >>> result = await scorer.score(
        ...     "What is the atomic number of gold?",
        ...     domain="science",
        ...     subdomain="chemistry",
        ...     reference="79",
        ... )
        >>> result.risk_score < 0.2  # expect low risk for well-known fact
        True
        """
        # --- 1. Consistency sampling ---
        samples = await self._draw_samples(question)
        consistency = await self._compute_consistency(samples)

        # --- 2. Grounding (optional) ---
        grounding: float | None = None
        if reference is not None:
            grounding = await self._compute_grounding(samples[0], reference)

        # --- 3. Composite risk ---
        risk = self._composite_risk(consistency, grounding)

        return ScoredResponse(
            question=question,
            domain=domain,
            subdomain=subdomain,
            risk_score=risk,
            consistency_score=consistency,
            grounding_score=grounding,
            samples=samples,
            model_id=self.adapter.model_id,
        )

    async def score_batch(
        self,
        questions: list[tuple[str, str, str]],
        references: list[str | None] | None = None,
        concurrency: int = 10,
    ) -> list[ScoredResponse]:
        """Score a batch of questions concurrently.

        Parameters
        ----------
        questions : list[tuple[str, str, str]]
            List of ``(question, domain, subdomain)`` triples.
        references : list[str | None] | None
            Optional parallel list of reference answers.
        concurrency : int
            Maximum concurrent scoring tasks.

        Returns
        -------
        list[ScoredResponse]
            One scored response per input question, in order.

        Examples
        --------
        >>> qs = [("Q1", "science", "physics"), ("Q2", "history", "wwii")]
        >>> results = await scorer.score_batch(qs)
        """
        refs: list[str | None] = references or [None] * len(questions)
        sem = asyncio.Semaphore(concurrency)

        async def _guarded(q: tuple[str, str, str], ref: str | None) -> ScoredResponse:
            async with sem:
                return await self.score(q[0], q[1], q[2], reference=ref)

        return await asyncio.gather(*[_guarded(q, r) for q, r in zip(questions, refs, strict=False)])

    # ------------------------------------------------------------------ #
    # Private helpers — algorithms are TODO stubs for Phase 2             #
    # ------------------------------------------------------------------ #

    async def _draw_samples(self, question: str) -> list[str]:
        """Draw ``n_samples`` independent completions from the adapter.

        TODO Phase 2: pass temperature to adapter call.
        """
        tasks = [
            self.adapter.complete(question, temperature=self.temperature)
            for _ in range(self.n_samples)
        ]
        return list(await asyncio.gather(*tasks))

    async def _compute_consistency(self, samples: list[str]) -> float:
        """Measure semantic consistency across samples.

        Returns a score in ``[0, 1]`` where ``1`` = perfectly consistent.

        TODO Phase 2: implement embedding-based pairwise cosine similarity.
        Currently returns a naive lexical overlap heuristic.
        """
        if len(samples) <= 1:
            return 1.0
        # TODO Phase 2: replace with proper semantic similarity
        unique_ratio = len(set(s.strip().lower() for s in samples)) / len(samples)
        # Fewer unique → more consistent; invert so 1 = perfectly consistent
        return float(np.clip(1.0 - (unique_ratio - 1.0 / len(samples)), 0.0, 1.0))

    async def _compute_grounding(self, response: str, reference: str) -> float:
        """Measure how well ``response`` matches the reference answer.

        Returns a score in ``[0, 1]`` where ``1`` = perfect match.

        TODO Phase 2: implement semantic textual similarity (STS) scoring.
        Currently falls back to token-level F1.
        """
        # TODO Phase 2: replace with STS model
        resp_tokens = set(response.lower().split())
        ref_tokens = set(reference.lower().split())
        if not ref_tokens:
            return 1.0
        overlap = resp_tokens & ref_tokens
        precision = len(overlap) / len(resp_tokens) if resp_tokens else 0.0
        recall = len(overlap) / len(ref_tokens)
        if precision + recall == 0:
            return 0.0
        f1 = 2 * precision * recall / (precision + recall)
        return float(f1)

    def _composite_risk(self, consistency: float, grounding: float | None) -> float:
        """Combine consistency and grounding into a final risk score.

        TODO Phase 2: calibrate alpha/beta weights empirically.
        """
        inconsistency = 1.0 - consistency
        if grounding is None:
            return float(np.clip(inconsistency, 0.0, 1.0))
        factual_error = 1.0 - grounding
        raw = self.alpha * inconsistency + self.beta * factual_error
        return float(np.clip(raw, 0.0, 1.0))
