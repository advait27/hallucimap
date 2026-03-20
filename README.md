# hallucimap

**Cartograph hallucination risk across an LLM's knowledge space.**

hallucimap probes a model systematically across domains, time periods, and entity types, then builds a persistent *RiskAtlas* — a 2-D danger map showing **where** the model confabulates, not just whether it hallucinated on a single output.

```
science/physics      ████████░░  0.82  ← high risk
history/wwii         ████░░░░░░  0.41
medicine/anatomy     ██░░░░░░░░  0.23
factual/mathematics  █░░░░░░░░░  0.09  ← well-calibrated
```

## Features

- **Multi-model support** — OpenAI (GPT-4o), Anthropic (Claude), HuggingFace (local)
- **Consistency sampling** — ask the same question N times; low consistency = high risk
- **Factual grounding** — cross-check answers against known references
- **Persistent RiskAtlas** — JSON-serializable; run incremental scans over time
- **Interactive heatmap** — Plotly HTML output, hover for confidence + sample counts
- **Async-first** — fully non-blocking; scans run with tunable concurrency
- **CLI** — `hallucimap scan` and `hallucimap show`

## Quickstart

```bash
pip install hallucimap
```

```bash
export OPENAI_API_KEY=sk-...

# Run a scan
hallucimap scan --model gpt-4o --domains science,history,medicine --samples 5

# Visualize the result
hallucimap show atlas_gpt-4o.json --browser
```

## Python API

```python
import asyncio
from hallucimap import RiskAtlas, HallucinationScorer
from hallucimap.models import AnthropicAdapter
from hallucimap.probes import DomainProbe, FactualProbe
from hallucimap.viz import HeatmapRenderer

async def main():
    adapter = AnthropicAdapter(model="claude-3-5-sonnet-20241022")
    scorer = HallucinationScorer(adapter=adapter, n_samples=5)
    atlas = RiskAtlas(model_id="claude-3-5-sonnet-20241022")

    for domain in ("science", "law", "medicine"):
        probe = DomainProbe(domain=domain)
        results = await probe.run_all(adapter)
        questions = [(r.question, r.domain, r.subdomain) for r in results]
        scored = await scorer.score_batch(questions)
        atlas.update(scored)

    atlas.save("atlas.json")
    HeatmapRenderer(atlas).save("atlas.html")
    print(atlas.summary())
    for cell in atlas.hottest_cells(n=5):
        print(f"  {cell.domain}/{cell.subdomain}  {cell.risk_score:.3f}")

asyncio.run(main())
```

## Architecture

```
Probes ──► LLM Adapter ──► HallucinationScorer ──► RiskAtlas ──► Heatmap
            (async)          (consistency+grounding)  (persistent)  (Plotly)
```

| Module | Responsibility |
|---|---|
| `core/atlas.py` | RiskAtlas — load, update, save, query |
| `core/scorer.py` | Consistency sampling + grounding scoring |
| `core/topology.py` | 2-D embedding of knowledge space (PCA / UMAP) |
| `probes/` | Probe generators: temporal, entity, domain, factual |
| `models/` | Async adapters: OpenAI, Anthropic, HuggingFace |
| `viz/heatmap.py` | Interactive Plotly heatmap renderer |
| `cli.py` | `hallucimap scan` / `hallucimap show` |

## Probe types

| Probe | What it targets |
|---|---|
| `FactualProbe` | Unambiguous facts (calibration baseline) |
| `EntityProbe` | Named entities — people, organizations, places |
| `DomainProbe` | Specialized knowledge: science, law, medicine, finance |
| `TemporalProbe` | Post-training-cutoff events |

## Development

```bash
git clone https://github.com/your-org/hallucimap
cd hallucimap
pip install -e ".[dev]"

# Lint
ruff check src tests

# Type-check
mypy src

# Tests
pytest
```

## Implementation phases

- [x] Phase 1 — Scaffold, Pydantic models, adapter stubs
- [ ] Phase 2 — Implement `HallucinationScorer` (embedding-based consistency)
- [ ] Phase 3 — Implement `KnowledgeTopology` (UMAP embedding)
- [ ] Phase 4 — Rich probe datasets (TriviaQA, Wikidata, curated corpora)
- [ ] Phase 5 — Interactive heatmap with topology overlay
- [ ] Phase 6 — CLI polish, packaging, docs site

## License

MIT
