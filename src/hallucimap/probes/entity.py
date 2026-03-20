"""EntityProbe — probe named-entity knowledge (people, orgs, places).

Named-entity questions are a common hallucination flashpoint: models
invent plausible-sounding but wrong biographical details, merge distinct
people, or fabricate org histories.

Examples
--------
>>> import asyncio
>>> from hallucimap.probes.entity import EntityProbe
>>> probe = EntityProbe(entity_type="person", max_questions=20)
>>> results = asyncio.run(probe.run_all(adapter))
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Literal

from hallucimap.probes.base import BaseProbe, ProbeResult

EntityType = Literal["person", "organization", "place"]

# TODO Phase 4: replace with a proper entity knowledge base / Wikidata queries.
_ENTITY_QUESTIONS: dict[EntityType, list[tuple[str, str | None]]] = {
    "person": [
        ("What year was Marie Curie born?", "1867"),
        ("What university did Albert Einstein attend?", "ETH Zurich"),
        ("What is Elon Musk's middle name?", None),
        ("Who founded Apple Computer?", "Steve Jobs, Steve Wozniak, Ronald Wayne"),
        ("What is the nationality of Sundar Pichai?", "Indian-American"),
    ],
    "organization": [
        ("When was Google founded?", "1998"),
        ("Where is the headquarters of the United Nations?", "New York City"),
        ("What does NASA stand for?", "National Aeronautics and Space Administration"),
        ("Who founded OpenAI?", None),
        ("What country is CERN located in?", "Switzerland"),
    ],
    "place": [
        ("What is the capital of Australia?", "Canberra"),
        ("How tall is Mount Everest in meters?", "8849"),
        ("Which river is the longest in the world?", "Nile"),
        ("What is the population of Tokyo?", None),
        ("In which continent is Kazakhstan located?", "Asia"),
    ],
}


class EntityProbe(BaseProbe):
    """Probe named-entity factual knowledge.

    Parameters
    ----------
    entity_type : EntityType
        One of ``"person"``, ``"organization"``, or ``"place"``.
    max_questions : int
        Cap on questions per run.

    Examples
    --------
    >>> probe = EntityProbe(entity_type="organization")
    >>> results = asyncio.run(probe.run_all(adapter))
    """

    domain = "entity"

    def __init__(
        self,
        entity_type: EntityType = "person",
        max_questions: int = 0,
    ) -> None:
        super().__init__(max_questions=max_questions)
        self.entity_type = entity_type
        self.subdomain = entity_type  # set instance subdomain

    async def generate_questions(self) -> AsyncIterator[str]:
        """Yield entity probe questions.

        Yields
        ------
        str
            A question about a named entity.

        Examples
        --------
        >>> async for q in probe.generate_questions():
        ...     print(q)
        """
        for question, _ref in _ENTITY_QUESTIONS.get(self.entity_type, []):
            yield question

    async def score_response(self, question: str, response: str) -> ProbeResult:
        """Package an entity probe response with reference if available.

        Parameters
        ----------
        question : str
            The probe question.
        response : str
            The model's raw response.

        Returns
        -------
        ProbeResult
            Populated result, with ``reference`` set when known.

        Examples
        --------
        >>> result = await probe.score_response("What year was Marie Curie born?", "1867")
        """
        # Look up reference answer
        reference: str | None = None
        for q, ref in _ENTITY_QUESTIONS.get(self.entity_type, []):
            if q == question:
                reference = ref
                break

        return ProbeResult(
            question=question,
            response=response,
            domain=self.domain,
            subdomain=self.entity_type,
            reference=reference,
        )
