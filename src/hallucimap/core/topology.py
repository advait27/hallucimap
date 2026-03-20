"""KnowledgeTopology — 2-D embedding of the domain knowledge space.

The topology projects each (domain, subdomain) cell into a 2-D coordinate
so that semantically similar knowledge areas cluster together.  This
coordinate is used as the spatial axis in the heatmap visualization.

Examples
--------
>>> from hallucimap.core.topology import KnowledgeTopology
>>> topo = KnowledgeTopology()
>>> coords = topo.fit(atlas.cells)
>>> # coords: dict mapping cell key → (x, y) tuple
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from pydantic import BaseModel

if TYPE_CHECKING:
    from hallucimap.core.atlas import AtlasCell


class TopologyCell(BaseModel):
    """A cell in the 2-D knowledge topology.

    Parameters
    ----------
    key : str
        The ``"<domain>/<subdomain>"`` identifier.
    x : float
        Horizontal coordinate in the embedded space.
    y : float
        Vertical coordinate in the embedded space.
    risk_score : float
        Risk score from the corresponding :class:`~hallucimap.core.atlas.AtlasCell`.

    Examples
    --------
    >>> cell = TopologyCell(key="science/quantum_physics", x=0.3, y=-0.7, risk_score=0.62)
    """

    key: str
    x: float
    y: float
    risk_score: float
    domain: str = ""
    subdomain: str = ""

    def model_post_init(self, __context: object) -> None:
        if "/" in self.key and not self.domain:
            parts = self.key.split("/", 1)
            self.domain = parts[0]
            self.subdomain = parts[1]


class KnowledgeTopology:
    """Build and query a 2-D map of the knowledge space.

    The topology is computed from domain/subdomain string embeddings.
    Dimensionality reduction is applied so that similar domains appear
    adjacent in the output coordinate space.

    Parameters
    ----------
    method : str
        Dimensionality reduction method.  Supported: ``"pca"`` (default),
        ``"umap"`` (requires ``umap-learn`` package).
    random_state : int
        Random seed for reproducibility.

    Examples
    --------
    >>> topo = KnowledgeTopology(method="pca")
    >>> layout = topo.fit(atlas.cells)
    >>> for tc in layout:
    ...     print(tc.key, tc.x, tc.y)
    """

    def __init__(self, method: str = "pca", random_state: int = 42) -> None:
        if method not in ("pca", "umap"):
            raise ValueError(f"Unknown topology method: {method!r}. Choose 'pca' or 'umap'.")
        self.method = method
        self.random_state = random_state

    def fit(self, cells: dict[str, AtlasCell]) -> list[TopologyCell]:
        """Embed all atlas cells into 2-D topology coordinates.

        Parameters
        ----------
        cells : dict[str, AtlasCell]
            The ``RiskAtlas.cells`` mapping.

        Returns
        -------
        list[TopologyCell]
            One topology cell per atlas cell, with ``(x, y)`` coordinates
            derived from the chosen dimensionality reduction method.

        Examples
        --------
        >>> layout = topo.fit(atlas.cells)
        >>> len(layout) == len(atlas.cells)
        True
        """
        if not cells:
            return []

        keys = list(cells.keys())
        # TODO Phase 3: encode domain/subdomain strings via a sentence-transformer
        #   and replace this placeholder with real embeddings.
        feature_matrix = self._placeholder_features(keys)
        coords_2d = self._reduce(feature_matrix)

        result = []
        for key, (x, y) in zip(keys, coords_2d, strict=False):
            cell = cells[key]
            result.append(
                TopologyCell(
                    key=key,
                    x=float(x),
                    y=float(y),
                    risk_score=cell.risk_score,
                )
            )
        return result

    # ------------------------------------------------------------------ #
    # Private helpers                                                      #
    # ------------------------------------------------------------------ #

    def _placeholder_features(self, keys: list[str]) -> np.ndarray:
        """Build a simple bag-of-chars feature matrix.

        TODO Phase 3: replace with sentence-transformer embeddings
        (e.g. ``all-MiniLM-L6-v2``).
        """
        # Use character n-gram hashing as a stand-in for real embeddings
        vocab: dict[str, int] = {}
        rows = []
        for key in keys:
            ngrams = [key[i : i + 3] for i in range(len(key) - 2)]
            for ng in ngrams:
                if ng not in vocab:
                    vocab[ng] = len(vocab)
            rows.append(ngrams)

        n_features = max(len(vocab), 1)
        matrix = np.zeros((len(keys), n_features))
        for i, ngrams in enumerate(rows):
            for ng in ngrams:
                matrix[i, vocab[ng]] += 1

        # L2-normalize
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1
        return matrix / norms

    def _reduce(self, features: np.ndarray) -> np.ndarray:
        """Reduce feature matrix to 2-D.

        TODO Phase 3: wire in UMAP when method == "umap".
        """
        n = features.shape[0]
        if n == 1:
            return np.array([[0.0, 0.0]])

        if self.method == "pca":
            return self._pca_2d(features)
        else:
            # TODO Phase 3: implement UMAP path
            return self._pca_2d(features)

    def _pca_2d(self, features: np.ndarray) -> np.ndarray:
        """Manual 2-component PCA (no sklearn dependency)."""
        centered = features - features.mean(axis=0)
        cov = np.cov(centered.T)
        if cov.ndim == 0:
            return np.zeros((features.shape[0], 2))
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        # Take the two eigenvectors with the largest eigenvalues
        top_idx = np.argsort(eigenvalues)[::-1][:2]
        components = eigenvectors[:, top_idx].T
        projected = centered @ components.T
        if projected.shape[1] < 2:
            projected = np.hstack([projected, np.zeros((projected.shape[0], 2 - projected.shape[1]))])
        return projected[:, :2]
