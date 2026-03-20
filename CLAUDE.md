# hallucimap

A Python package that cartographs hallucination risk across an LLM's knowledge 
space — producing persistent "danger maps" of WHERE a model confabulates.

## Architecture

### Core modules
- `core/atlas.py`      — RiskAtlas: persistent JSON/pickle store of risk scores per domain
- `core/scorer.py`     — HallucinationScorer: consistency sampling + factual grounding
- `core/topology.py`   — KnowledgeTopology: UMAP/t-SNE projection of risk scores to 2D

### Probe system
- `probes/base.py`     — BaseProbe: abstract probe with .generate_queries() + .score()
- `probes/temporal.py` — probes knowledge around LLM cutoff dates
- `probes/entity.py`   — probes named entity knowledge (people, orgs, locations)
- `probes/domain.py`   — probes domain knowledge (biomedical, legal, financial)
- `probes/factual.py`  — probes verifiable factual claims

### Model adapters
- `models/base.py`     — BaseLLMAdapter: .complete(), .complete_batch(), .embed()
- `models/openai_adapter.py`    — OpenAI GPT-4o/GPT-4-turbo
- `models/anthropic_adapter.py` — Anthropic Claude 3.5+
- `models/hf_adapter.py`        — HuggingFace local models

### Viz
- `viz/heatmap.py`     — Plotly-based interactive risk heatmap

## Key design decisions
- RiskAtlas is model-specific and persisted to disk (JSON + numpy arrays)
- Hallucination scoring uses N=5 consistency samples + optional web grounding
- Topology projection uses UMAP by default, falls back to t-SNE
- All LLM calls are async with tenacity retry logic
- CLI built with Typer: `hallucimap scan --model gpt-4o --probes temporal,entity`

## Build commands
pip install -e ".[dev]"
pytest tests/ -v --cov=src/hallucimap
ruff check src/
mypy src/hallucimap --ignore-missing-imports

## Phase tracker
- [x] Phase 0: Scaffold
- [ ] Phase 1: Model adapters + BaseProbe
- [ ] Phase 2: Scorer (consistency sampling)
- [ ] Phase 3: TemporalProbe + EntityProbe
- [ ] Phase 4: RiskAtlas persistence
- [ ] Phase 5: Topology projection + Heatmap viz
- [ ] Phase 6: CLI + examples
- [ ] Phase 7: DomainProbe + FactualProbe
- [ ] Phase 8: PyPI publish
```

---

## Step 3 — Phase-by-phase build prompts for Claude Code

Run these **one phase at a time** — don't skip ahead:

**Phase 1 — Model adapters**
```
Implement all three model adapters fully:
- BaseLLMAdapter with async .complete(prompt, n=1) and .complete_batch()
- OpenAIAdapter using the openai>=1.0 async client, model="gpt-4o" default
- AnthropicAdapter using anthropic>=0.25 async client, model="claude-sonnet-4-6"
- HuggingFaceAdapter using transformers pipeline, CPU/GPU autodetect
All adapters must handle rate limiting via tenacity (expo backoff, 3 retries).
Write tests for all three using pytest-asyncio with httpx mock fixtures.
```

**Phase 2 — Hallucination scorer**
```
Implement HallucinationScorer in core/scorer.py:
- .score(query, model_adapter, n_samples=5) -> HallucinationScore(pydantic)
- Consistency score: sample N responses, compute pairwise semantic similarity 
  using sentence-transformers all-MiniLM-L6-v2, return mean similarity
- Confidence score: ask model to rate its own certainty 0-1 on the same query
- Final risk_score = 1 - (0.6 * consistency + 0.4 * confidence)
- HallucinationScore fields: query, risk_score, consistency, confidence, 
  samples: list[str], timestamp
Write full tests with mock adapters.
```

**Phase 3 — Probes**
```
Implement TemporalProbe and EntityProbe:
TemporalProbe:
- Generates queries about events in 6-month windows relative to a given cutoff
- Default cutoff: "2023-04" (GPT-4 training cutoff)
- Query templates: recent appointments, releases, deaths, elections, discoveries

EntityProbe:
- Probes named entities from a configurable entity list
- Entity types: Person, Organization, Location, Product
- Query templates: biographical facts, founding dates, locations, key attributes

Both probes inherit BaseProbe and must implement:
- .generate_queries(n=20) -> list[ProbeQuery]
- .probe(model_adapter, scorer) -> list[HallucinationScore]
Write tests with small entity/temporal fixtures.
```

**Phase 4 — RiskAtlas**
```
Implement RiskAtlas in core/atlas.py:
- Stores HallucinationScores organized by (model_id, probe_type, domain)
- Persists to disk as atlas.json (metadata) + atlas.npy (score matrix)
- .add_scores(scores: list[HallucinationScore]) 
- .get_risk_map(probe_type=None) -> pd.DataFrame
- .top_risk_domains(n=10) -> list[tuple[str, float]]
- .summary() -> AtlasSummary(pydantic) with overall stats
- .save(path) / .load(path) classmethods
Write full tests including serialization round-trips.
```

**Phase 5 — Topology + Heatmap**
```
Implement KnowledgeTopology and heatmap viz:
KnowledgeTopology (core/topology.py):
- Takes a RiskAtlas and projects domains to 2D via UMAP (umap-learn)
- Falls back to t-SNE if umap not installed
- .fit(atlas) -> self
- .transform() -> TopologyMap(x, y, risk_scores, labels)

Heatmap (viz/heatmap.py):
- Takes a TopologyMap and renders an interactive Plotly scatter plot
- Color = risk_score (red=high, blue=low, RdBu_r colorscale)
- Hover shows: domain, probe_type, risk_score, n_samples
- .render() opens in browser
- .save(path) saves as standalone HTML
```

**Phase 6 — CLI**
```
Implement the Typer CLI in cli.py:
hallucimap scan:
  --model [gpt-4o|claude|hf:<model_id>]  (required)
  --probes [temporal,entity,domain,factual] (default: temporal,entity)
  --n-queries INT (default: 20 per probe)
  --output PATH (default: ./hallucimap_atlas/)
  --api-key TEXT (or reads from env: OPENAI_API_KEY, ANTHROPIC_API_KEY)
  Shows a Rich progress bar during scanning.

hallucimap show:
  --atlas PATH (default: ./hallucimap_atlas/)
  Opens the interactive Plotly heatmap in browser.

hallucimap summary:
  --atlas PATH
  Prints a Rich table: top 10 risk domains, overall stats, model info.

Wire up in pyproject.toml [project.scripts].