"""Example: scan Claude and build a RiskAtlas.

Run:
    ANTHROPIC_API_KEY=sk-ant-... python examples/scan_claude.py
"""

from __future__ import annotations

import asyncio

from hallucimap import RiskAtlas
from hallucimap.core.atlas import ScanSession
from hallucimap.core.scorer import HallucinationScorer
from hallucimap.models import AnthropicAdapter
from hallucimap.probes import DomainProbe, FactualProbe, TemporalProbe
from hallucimap.viz import HeatmapRenderer


async def main() -> None:
    model_id = "claude-3-5-sonnet-20241022"
    adapter = AnthropicAdapter(model=model_id)
    scorer = HallucinationScorer(adapter=adapter, n_samples=5, temperature=0.9)

    atlas = RiskAtlas(model_id=model_id)
    session = ScanSession(model_id=model_id)

    all_results = []

    # 1. Calibration
    print("Running factual calibration probes…")
    all_results.extend(await FactualProbe().run_all(adapter, concurrency=5))

    # 2. Temporal probes — Claude's knowledge cutoff is early 2024
    print("Running temporal probes (post-cutoff events)…")
    temporal = TemporalProbe(cutoff_year=2024, target_years=[2024, 2025])
    all_results.extend(await temporal.run_all(adapter, concurrency=5))

    # 3. Domain probes — areas where Claude may confabulate
    print("Running domain probes…")
    for domain in ("science", "law", "medicine"):
        all_results.extend(await DomainProbe(domain=domain).run_all(adapter, concurrency=5))

    # Score
    print(f"Scoring {len(all_results)} probe responses…")
    questions = [(r.question, r.domain, r.subdomain) for r in all_results]
    references = [r.reference for r in all_results]
    scored = await scorer.score_batch(questions, references=references, concurrency=10)

    atlas.update(scored, session=session)

    output_path = "atlas_claude.json"
    atlas.save(output_path)
    print(atlas.summary())
    print(f"Atlas saved to {output_path}")

    hot = atlas.hottest_cells(n=5)
    print("\nTop 5 highest-risk cells:")
    for cell in hot:
        print(f"  {cell.domain}/{cell.subdomain}  risk={cell.risk_score:.3f}  "
              f"conf={cell.confidence:.2f}  n={cell.sample_count}")

    renderer = HeatmapRenderer(atlas)
    renderer.save("atlas_claude.html")
    print("Heatmap saved to atlas_claude.html")


if __name__ == "__main__":
    asyncio.run(main())
