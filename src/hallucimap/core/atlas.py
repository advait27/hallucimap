"""RiskAtlas — the persistent danger map for a single model.

The atlas stores one :class:`AtlasCell` per (domain, subdomain) pair.  Each
cell accumulates scored probe results over time.  You can run partial scans,
load an existing atlas, update it, and save it back — enabling incremental
cartography without re-probing cells you already know.

Examples
--------
>>> atlas = RiskAtlas(model_id="gpt-4o")
>>> atlas.update(scored_results)
>>> atlas.save("gpt4o_atlas.json")
>>> loaded = RiskAtlas.load("gpt4o_atlas.json")
>>> print(loaded.hottest_cells(n=5))
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np
from pydantic import BaseModel, Field

from hallucimap.core.scorer import ScoredResponse


class AtlasCell(BaseModel):
    """One cell in the risk atlas, corresponding to a (domain, subdomain) pair.

    Parameters
    ----------
    domain : str
        Top-level domain label, e.g. ``"science"``.
    subdomain : str
        Finer-grained subdomain, e.g. ``"quantum_physics"``.
    risk_score : float
        Mean hallucination risk in ``[0, 1]``.  Higher is riskier.
    confidence : float
        Confidence in the risk estimate based on sample count.  ``[0, 1]``.
    sample_count : int
        Number of scored responses that inform this cell.
    last_updated : datetime
        UTC timestamp of last update.

    Examples
    --------
    >>> cell = AtlasCell(domain="science", subdomain="quantum_physics")
    >>> cell.risk_score
    0.0
    """

    domain: str
    subdomain: str
    risk_score: float = 0.0
    confidence: float = 0.0
    sample_count: int = 0
    last_updated: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # Raw scores for recomputation
    raw_scores: list[float] = Field(default_factory=list)

    def absorb(self, response: ScoredResponse) -> None:
        """Incorporate a new scored response into this cell.

        Parameters
        ----------
        response : ScoredResponse
            A freshly scored probe response.
        """
        self.raw_scores.append(response.risk_score)
        self.sample_count = len(self.raw_scores)
        self.risk_score = float(np.mean(self.raw_scores))
        # Confidence grows with sample count (simple asymptotic formula)
        # TODO Phase 3: replace with proper Bayesian confidence interval
        self.confidence = 1.0 - 1.0 / (1.0 + self.sample_count * 0.2)
        self.last_updated = datetime.now(timezone.utc)


class ScanSession(BaseModel):
    """Metadata for a single scan run.

    Parameters
    ----------
    session_id : str
        Auto-generated UUID for this scan session.
    model_id : str
        Identifier of the model that was probed.
    started_at : datetime
        UTC start time.
    finished_at : datetime | None
        UTC end time, set when the session completes.
    probe_count : int
        Total number of probes run.

    Examples
    --------
    >>> session = ScanSession(model_id="gpt-4o")
    >>> session.session_id  # auto-generated UUID string
    """

    session_id: str = Field(default_factory=lambda: str(uuid4()))
    model_id: str
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    finished_at: datetime | None = None
    probe_count: int = 0


class RiskAtlas(BaseModel):
    """The persistent danger map for a single LLM.

    A ``RiskAtlas`` accumulates :class:`AtlasCell` entries over one or more
    scan sessions.  It is the canonical output artifact of a hallucimap run
    and can be serialized / deserialized to JSON.

    Parameters
    ----------
    model_id : str
        Identifier string for the model (e.g. ``"gpt-4o"``).
    cells : dict[str, AtlasCell]
        Mapping of ``"<domain>/<subdomain>"`` to its cell.
    sessions : list[ScanSession]
        Ordered list of all scan sessions that contributed to this atlas.

    Examples
    --------
    >>> atlas = RiskAtlas(model_id="gpt-4o")
    >>> atlas.update(scored_results)
    >>> atlas.save("gpt4o_atlas.json")
    >>> loaded = RiskAtlas.load("gpt4o_atlas.json")
    """

    model_id: str
    cells: dict[str, AtlasCell] = Field(default_factory=dict)
    sessions: list[ScanSession] = Field(default_factory=list)

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def update(
        self,
        responses: Sequence[ScoredResponse],
        session: ScanSession | None = None,
    ) -> None:
        """Incorporate a batch of scored responses into the atlas.

        Parameters
        ----------
        responses : Sequence[ScoredResponse]
            Scored probe responses to absorb.
        session : ScanSession | None
            If provided, recorded in ``self.sessions`` with final probe count.

        Examples
        --------
        >>> atlas.update(scored_responses, session=my_session)
        """
        for resp in responses:
            key = f"{resp.domain}/{resp.subdomain}"
            if key not in self.cells:
                self.cells[key] = AtlasCell(domain=resp.domain, subdomain=resp.subdomain)
            self.cells[key].absorb(resp)

        if session is not None:
            session.finished_at = datetime.now(timezone.utc)
            session.probe_count = len(responses)
            self.sessions.append(session)

    def hottest_cells(self, n: int = 10) -> list[AtlasCell]:
        """Return the ``n`` highest-risk cells, sorted descending.

        Parameters
        ----------
        n : int
            Number of cells to return.

        Returns
        -------
        list[AtlasCell]
            Cells with the highest ``risk_score``.

        Examples
        --------
        >>> hot = atlas.hottest_cells(n=5)
        >>> for c in hot:
        ...     print(c.domain, c.subdomain, f"{c.risk_score:.2f}")
        """
        return sorted(self.cells.values(), key=lambda c: c.risk_score, reverse=True)[:n]

    def risk_matrix(self) -> tuple[list[str], list[str], np.ndarray]:
        """Build a 2-D risk matrix (domains × subdomains).

        Returns
        -------
        domains : list[str]
            Sorted unique domain labels (rows).
        subdomains : list[str]
            Sorted unique subdomain labels (columns).
        matrix : np.ndarray
            Shape ``(len(domains), len(subdomains))``.  Cells with no data
            contain ``NaN``.

        Examples
        --------
        >>> domains, subdomains, mat = atlas.risk_matrix()
        >>> import numpy as np
        >>> np.nanmean(mat)  # overall mean risk
        """
        domains = sorted({c.domain for c in self.cells.values()})
        subdomains = sorted({c.subdomain for c in self.cells.values()})
        matrix = np.full((len(domains), len(subdomains)), np.nan)
        d_idx = {d: i for i, d in enumerate(domains)}
        s_idx = {s: i for i, s in enumerate(subdomains)}
        for cell in self.cells.values():
            matrix[d_idx[cell.domain], s_idx[cell.subdomain]] = cell.risk_score
        return domains, subdomains, matrix

    # ------------------------------------------------------------------ #
    # Persistence                                                          #
    # ------------------------------------------------------------------ #

    def save(self, path: str | Path) -> None:
        """Serialize the atlas to a JSON file.

        Parameters
        ----------
        path : str | Path
            Destination file path (created or overwritten).

        Examples
        --------
        >>> atlas.save("gpt4o_atlas.json")
        """
        Path(path).write_text(self.model_dump_json(indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> RiskAtlas:
        """Deserialize a ``RiskAtlas`` from a JSON file.

        Parameters
        ----------
        path : str | Path
            Path to a previously saved atlas JSON.

        Returns
        -------
        RiskAtlas
            Fully reconstructed atlas.

        Examples
        --------
        >>> atlas = RiskAtlas.load("gpt4o_atlas.json")
        """
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.model_validate(data)

    def summary(self) -> str:
        """Return a human-readable one-liner summary.

        Examples
        --------
        >>> print(atlas.summary())
        RiskAtlas(model=gpt-4o, cells=42, mean_risk=0.37, sessions=3)
        """
        if not self.cells:
            return f"RiskAtlas(model={self.model_id}, cells=0)"
        mean_risk = np.mean([c.risk_score for c in self.cells.values()])
        return (
            f"RiskAtlas(model={self.model_id}, cells={len(self.cells)}, "
            f"mean_risk={mean_risk:.3f}, sessions={len(self.sessions)})"
        )
