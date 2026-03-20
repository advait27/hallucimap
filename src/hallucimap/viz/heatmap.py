"""HeatmapRenderer — interactive Plotly heatmap for a RiskAtlas.

Renders the atlas risk matrix as an interactive HTML heatmap where:
- rows = domains
- columns = subdomains
- color = risk score (0 = safe / blue, 1 = dangerous / red)

Hovering over a cell shows domain, subdomain, risk score, confidence,
and sample count.

Examples
--------
>>> from hallucimap import RiskAtlas
>>> from hallucimap.viz import HeatmapRenderer
>>>
>>> atlas = RiskAtlas.load("gpt4o_atlas.json")
>>> renderer = HeatmapRenderer(atlas)
>>> renderer.show()           # opens browser
>>> renderer.save("map.html") # save standalone HTML
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

try:
    import plotly.graph_objects as go
    from plotly.graph_objects import Figure
except ImportError as e:
    raise ImportError("plotly required: pip install plotly") from e

if TYPE_CHECKING:
    from hallucimap.core.atlas import RiskAtlas


class HeatmapRenderer:
    """Render a :class:`RiskAtlas` as an interactive Plotly heatmap.

    Parameters
    ----------
    atlas : RiskAtlas
        The atlas to visualize.
    colorscale : str
        Plotly colorscale name.  Default: ``"RdYlGn_r"`` (red = high risk).
    title : str | None
        Chart title.  Defaults to ``"Hallucination Risk: <model_id>"``.
    width : int
        Figure width in pixels.
    height : int
        Figure height in pixels.

    Examples
    --------
    >>> renderer = HeatmapRenderer(atlas, colorscale="Hot_r")
    >>> renderer.show()
    """

    def __init__(
        self,
        atlas: RiskAtlas,
        colorscale: str = "RdYlGn_r",
        title: str | None = None,
        width: int = 1000,
        height: int = 600,
    ) -> None:
        self.atlas = atlas
        self.colorscale = colorscale
        self.title = title or f"Hallucination Risk: {atlas.model_id}"
        self.width = width
        self.height = height

    def build_figure(self) -> Figure:
        """Construct the Plotly figure.

        Returns
        -------
        Figure
            A Plotly figure ready for ``show()`` or ``write_html()``.

        Examples
        --------
        >>> fig = renderer.build_figure()
        >>> fig.show()
        """
        domains, subdomains, matrix = self.atlas.risk_matrix()

        if not domains:
            # Empty atlas — return a placeholder figure
            fig = go.Figure()
            fig.update_layout(title="No data — run a scan first.")
            return fig

        # Build custom hover text
        hover = self._build_hover_text(domains, subdomains, matrix)

        # TODO Phase 5: overlay topology coordinates from KnowledgeTopology
        #   so semantically similar cells appear adjacent regardless of
        #   alphabetical domain/subdomain sort order.

        fig = go.Figure(
            data=go.Heatmap(
                z=matrix,
                x=subdomains,
                y=domains,
                colorscale=self.colorscale,
                zmin=0.0,
                zmax=1.0,
                text=hover,
                hovertemplate=(
                    "<b>%{y} / %{x}</b><br>"
                    "Risk: %{z:.3f}<br>"
                    "%{text}<extra></extra>"
                ),
                colorbar=dict(
                    title="Risk Score",
                    tickvals=[0, 0.25, 0.5, 0.75, 1.0],
                    ticktext=["0 (safe)", "0.25", "0.5", "0.75", "1 (danger)"],
                ),
            )
        )

        fig.update_layout(
            title=dict(text=self.title, font=dict(size=18)),
            xaxis=dict(title="Subdomain", tickangle=-30),
            yaxis=dict(title="Domain"),
            width=self.width,
            height=self.height,
            plot_bgcolor="white",
            paper_bgcolor="white",
        )
        return fig

    def show(self) -> None:
        """Open the heatmap in the default browser.

        Examples
        --------
        >>> renderer.show()
        """
        self.build_figure().show()

    def save(self, path: str | Path) -> None:
        """Save the heatmap as a standalone HTML file.

        Parameters
        ----------
        path : str | Path
            Destination ``.html`` file path.

        Examples
        --------
        >>> renderer.save("atlas_gpt4o.html")
        """
        self.build_figure().write_html(str(path), include_plotlyjs="cdn")

    # ------------------------------------------------------------------ #
    # Private helpers                                                      #
    # ------------------------------------------------------------------ #

    def _build_hover_text(
        self,
        domains: list[str],
        subdomains: list[str],
        matrix: np.ndarray,
    ) -> list[list[str]]:
        """Build the hover annotation matrix."""
        cells = self.atlas.cells
        hover: list[list[str]] = []
        for domain in domains:
            row: list[str] = []
            for subdomain in subdomains:
                key = f"{domain}/{subdomain}"
                cell = cells.get(key)
                if cell is None:
                    row.append("No data")
                else:
                    row.append(
                        f"Confidence: {cell.confidence:.2f} | "
                        f"Samples: {cell.sample_count}"
                    )
            hover.append(row)
        return hover
