"""DomainProbe — probe deep domain knowledge (science, law, medicine, finance).

Domain probes target specialized knowledge that models frequently
confabulate: drug interactions, legal citations, financial regulations,
and cutting-edge research findings.

Examples
--------
>>> import asyncio
>>> from hallucimap.probes.domain import DomainProbe
>>> probe = DomainProbe(domain="medicine", subdomain="pharmacology")
>>> results = asyncio.run(probe.run_all(adapter))
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from hallucimap.probes.base import BaseProbe, ProbeResult

# TODO Phase 4: expand each domain with a rich curated question bank,
#   ideally loaded from a versioned external dataset.
_DOMAIN_QUESTIONS: dict[str, dict[str, list[tuple[str, str | None]]]] = {
    "science": {
        "physics": [
            ("What is the Planck constant in SI units?", "6.626e-34 J·s"),
            ("What does the Schrödinger equation describe?", None),
            ("What is dark matter?", None),
        ],
        "chemistry": [
            ("What is the atomic number of gold?", "79"),
            ("What is the pH of pure water at 25°C?", "7"),
            ("What is the molecular formula of glucose?", "C6H12O6"),
        ],
        "biology": [
            ("What is CRISPR-Cas9 used for?", None),
            ("How many chromosomes does a human cell have?", "46"),
            ("What organelle produces ATP in eukaryotic cells?", "mitochondria"),
        ],
    },
    "law": {
        "constitutional": [
            ("How many amendments does the US Constitution have?", "27"),
            ("What does the First Amendment protect?", None),
            ("What year was the US Constitution ratified?", "1788"),
        ],
        "intellectual_property": [
            ("How long does a US utility patent last?", "20 years"),
            ("What is the difference between copyright and trademark?", None),
            ("What does DMCA stand for?", "Digital Millennium Copyright Act"),
        ],
    },
    "medicine": {
        "pharmacology": [
            ("What is the mechanism of action of aspirin?", None),
            ("What are SSRIs used to treat?", "depression, anxiety"),
            ("What is the antidote for acetaminophen overdose?", "N-acetylcysteine"),
        ],
        "anatomy": [
            ("How many bones are in the adult human body?", "206"),
            ("What is the largest organ in the human body?", "skin"),
            ("Where is the pituitary gland located?", "base of the brain"),
        ],
    },
    "finance": {
        "markets": [
            ("What does P/E ratio stand for?", "price-to-earnings"),
            ("What is a short squeeze?", None),
            ("What does the Federal Reserve control?", None),
        ],
        "instruments": [
            ("What is a derivative in finance?", None),
            ("What does ETF stand for?", "Exchange-Traded Fund"),
            ("What is the difference between a bond and a stock?", None),
        ],
    },
}


class DomainProbe(BaseProbe):
    """Probe deep domain knowledge within a specific field.

    Parameters
    ----------
    domain : str
        Top-level domain.  Supported: ``"science"``, ``"law"``,
        ``"medicine"``, ``"finance"``.
    subdomain : str | None
        If provided, restricts questions to this subdomain only.
        If ``None``, all subdomains within ``domain`` are probed.
    max_questions : int
        Cap on questions per run.

    Examples
    --------
    >>> probe = DomainProbe(domain="science", subdomain="chemistry")
    >>> results = asyncio.run(probe.run_all(adapter))
    """

    def __init__(
        self,
        domain: str = "science",
        subdomain: str | None = None,
        max_questions: int = 0,
    ) -> None:
        super().__init__(max_questions=max_questions)
        self.domain = domain
        self.subdomain = subdomain or "general"
        self._subdomain_filter = subdomain

    async def generate_questions(self) -> AsyncIterator[str]:
        """Yield domain probe questions.

        Yields
        ------
        str
            A specialized domain question.

        Examples
        --------
        >>> async for q in probe.generate_questions():
        ...     print(q)
        """
        subdomains = _DOMAIN_QUESTIONS.get(self.domain, {})
        for sd_name, questions in subdomains.items():
            if self._subdomain_filter and sd_name != self._subdomain_filter:
                continue
            for question, _ref in questions:
                yield question

    async def score_response(self, question: str, response: str) -> ProbeResult:
        """Package a domain probe response.

        Parameters
        ----------
        question : str
            The probe question.
        response : str
            The model's raw response.

        Returns
        -------
        ProbeResult
            Populated result with reference if available.

        Examples
        --------
        >>> result = await probe.score_response("What is the pH of pure water?", "7")
        """
        reference: str | None = None
        subdomain_hit = "general"
        subdomains = _DOMAIN_QUESTIONS.get(self.domain, {})
        for sd_name, questions in subdomains.items():
            if self._subdomain_filter and sd_name != self._subdomain_filter:
                continue
            for q, ref in questions:
                if q == question:
                    reference = ref
                    subdomain_hit = sd_name
                    break

        return ProbeResult(
            question=question,
            response=response,
            domain=self.domain,
            subdomain=subdomain_hit,
            reference=reference,
        )
