"""Example: scan GPT-4o and build a RiskAtlas.

Run:
    OPENAI_API_KEY=sk-... python examples/scan_gpt4o.py
"""

from __future__ import annotations

import asyncio

from hallucimap import RiskAtlas
from hallucimap.core.atlas import ScanSession
from hallucimap.core.scorer import HallucinationScorer
from hallucimap.models import OpenAIAdapter
from hallucimap.probes import DomainProbe, EntityProbe, FactualProbe
from hallucimap.viz import HeatmapRenderer


async def main() -> None:
    model_id = "gpt-4o"
    adapter = OpenAIAdapter(model=model_id)
    scorer = HallucinationScorer(adapter=adapter, n_samples=5, temperature=0.9)

    atlas = RiskAtlas(model_id=model_id)
    session = ScanSession(model_id=model_id)

    all_results = []

    # 1. Calibration: factual probes (should score low risk)
    print("Running factual calibration probes…")
    factual = FactualProbe()
    all_results.extend(await factual.run_all(adapter, concurrency=5))

    # 2. Entity probes
    print("Running entity probes…")
    for etype in ("person", "organization", "place"):
        probe = EntityProbe(entity_type=etype)  # type: ignore[arg-type]
        all_results.extend(await probe.run_all(adapter, concurrency=5))

    # 3. Domain probes
    print("Running domain probes…")
    for domain in ("science", "law", "medicine", "finance"):
        probe = DomainProbe(domain=domain)
        all_results.extend(await probe.run_all(adapter, concurrency=5))

    # Score everything
    print(f"Scoring {len(all_results)} probe responses…")
    questions = [(r.question, r.domain, r.subdomain) for r in all_results]
    references = [r.reference for r in all_results]
    scored = await scorer.score_batch(questions, references=references, concurrency=10)

    atlas.update(scored, session=session)

    output_path = "atlas_gpt4o.json"
    atlas.save(output_path)
    print(atlas.summary())
    print(f"Atlas saved to {output_path}")

    # Visualize
    renderer = HeatmapRenderer(atlas)
    renderer.save("atlas_gpt4o.html")
    print("Heatmap saved to atlas_gpt4o.html")
    # renderer.show()  # uncomment to open in browser


if __name__ == "__main__":
    asyncio.run(main())
