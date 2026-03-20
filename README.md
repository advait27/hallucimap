<div align="center">

# 🗺️ hallucimap

### *Cartograph hallucination risk across an LLM's knowledge space*

[![Python](https://img.shields.io/badge/python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![CI](https://img.shields.io/github/actions/workflow/status/advait27/hallucimap/ci.yml?branch=main&label=CI&logo=github)](https://github.com/advait27/hallucimap/actions)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Async](https://img.shields.io/badge/async--first-asyncio-purple)](https://docs.python.org/3/library/asyncio.html)
[![Models](https://img.shields.io/badge/models-OpenAI%20%7C%20Anthropic%20%7C%20HuggingFace-orange)](https://github.com/advait27/hallucimap)

<br/>

> **hallucimap** doesn't just ask *"did the model hallucinate?"* —
> it builds a persistent **danger map** showing *exactly where* a model confabulates
> across every domain, time period, and entity type it knows.

<br/>

```
  science/physics      ████████░░  0.82  ◄ high risk zone
  history/wwii         █████░░░░░  0.53
  medicine/anatomy     ███░░░░░░░  0.31
  finance/markets      ██░░░░░░░░  0.19
  factual/mathematics  █░░░░░░░░░  0.08  ◄ well-calibrated
```

</div>

---

## 🧭 What is this?

Most hallucination tools check a **single output** — one prompt, one verdict. That tells you almost nothing about the model's *systematic* failure modes.

**hallucimap** takes a different approach:

1. **Probe** the model across hundreds of questions spanning domains, time periods, and entity types
2. **Score** each answer using consistency sampling — ask the same question N times and measure how much the model contradicts itself
3. **Map** the scores into a persistent `RiskAtlas` — a 2-D grid of hallucination risk per knowledge cell
4. **Visualize** the atlas as an interactive heatmap so you can instantly see the danger zones

The result is a **reusable, persistent fingerprint** of where a model is unreliable — not just on today's test set, but structurally across its knowledge space.

---

## ✨ Features

| Feature | Description |
|---|---|
| 🔍 **Consistency Sampling** | Ask the same question N times at temperature > 0. Low agreement = high risk. |
| 📐 **Factual Grounding** | Cross-check answers against known references to catch confident confabulation. |
| 🗄️ **Persistent RiskAtlas** | JSON-serializable danger map that accumulates across multiple scan sessions. |
| 🌐 **Multi-Model** | OpenAI (GPT-4o), Anthropic (Claude 3.5+), or any local HuggingFace model. |
| ⚡ **Async-First** | Fully non-blocking — scans run concurrently with tunable parallelism. |
| 🗺️ **Interactive Heatmap** | Plotly HTML output — hover for domain, risk score, confidence, and sample count. |
| 🔁 **Incremental Scans** | Load an existing atlas and extend it — only probe what you haven't mapped yet. |
| 🖥️ **CLI** | `hallucimap scan` and `hallucimap show` — batteries included. |

---

## 🚀 Quickstart

### Install

```bash
pip install hallucimap
```

### Scan a model

```bash
# OpenAI
export OPENAI_API_KEY=sk-...
hallucimap scan --model gpt-4o --domains science,history,medicine --samples 5

# Anthropic
export ANTHROPIC_API_KEY=sk-ant-...
hallucimap scan --model claude-3-5-sonnet-20241022 --domains law,finance --samples 5
```

### Visualize the danger map

```bash
# Open interactive heatmap in browser
hallucimap show atlas_gpt-4o.json --browser

# Save as standalone HTML
hallucimap show atlas_gpt-4o.json --save map.html
```

### Print a summary table

```
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃               Risk Atlas: gpt-4o                             ┃
┡━━━━━━━━━━━━━━━┯━━━━━━━━━━━━━━━┯━━━━━━━┯━━━━━━━━━━━━┯━━━━━━━━┩
│ Domain        │ Subdomain     │  Risk │ Confidence │ Samples│
├───────────────┼───────────────┼───────┼────────────┼────────┤
│ science       │ physics       │ 0.821 │       0.91 │     25 │
│ temporal      │ post_cutoff   │ 0.764 │       0.88 │     20 │
│ entity        │ person        │ 0.612 │       0.85 │     15 │
│ history       │ wwii          │ 0.534 │       0.83 │     15 │
│ medicine      │ pharmacology  │ 0.487 │       0.82 │     15 │
└───────────────┴───────────────┴───────┴────────────┴────────┘
RiskAtlas(model=gpt-4o, cells=24, mean_risk=0.371, sessions=2)
```

---

## 🐍 Python API

```python
import asyncio
from hallucimap import RiskAtlas, HallucinationScorer
from hallucimap.models import AnthropicAdapter
from hallucimap.probes import DomainProbe, EntityProbe, TemporalProbe
from hallucimap.viz import HeatmapRenderer

async def main():
    # 1. Set up adapter + scorer
    adapter = AnthropicAdapter(model="claude-3-5-sonnet-20241022")
    scorer  = HallucinationScorer(adapter=adapter, n_samples=5, temperature=0.9)
    atlas   = RiskAtlas(model_id="claude-3-5-sonnet-20241022")

    # 2. Run probes across multiple domains
    probes = [
        DomainProbe(domain="science"),
        DomainProbe(domain="medicine"),
        EntityProbe(entity_type="person"),
        TemporalProbe(cutoff_year=2024),
    ]
    for probe in probes:
        results  = await probe.run_all(adapter, concurrency=10)
        questions = [(r.question, r.domain, r.subdomain) for r in results]
        references = [r.reference for r in results]
        scored   = await scorer.score_batch(questions, references=references)
        atlas.update(scored)

    # 3. Inspect the danger map
    print(atlas.summary())
    for cell in atlas.hottest_cells(n=5):
        print(f"  {cell.domain}/{cell.subdomain}  risk={cell.risk_score:.3f}")

    # 4. Persist + visualize
    atlas.save("atlas.json")
    HeatmapRenderer(atlas).save("atlas.html")   # standalone interactive HTML

asyncio.run(main())
```

---

## 🏗️ How It Works

```
┌─────────────┐    questions     ┌───────────────┐   N completions  ┌─────────────────────┐
│   Probes    │ ───────────────► │  LLM Adapter  │ ───────────────► │ HallucinationScorer │
│             │                  │  (async+retry)│                  │                     │
│ • Temporal  │                  └───────────────┘                  │ consistency score   │
│ • Entity    │                                                      │ + grounding score   │
│ • Domain    │                                                      │ → risk_score [0,1]  │
│ • Factual   │                                                      └──────────┬──────────┘
└─────────────┘                                                                 │
                                                                                ▼
                                                                     ┌─────────────────────┐
                                                                     │      RiskAtlas      │
                                                                     │                     │
                                                                     │  domain/subdomain   │
                                                                     │  → AtlasCell        │
                                                                     │  (risk, confidence, │
                                                                     │   sample_count)     │
                                                                     └──────────┬──────────┘
                                                                                │
                                                              ┌─────────────────┴──────────────┐
                                                              │                                │
                                                              ▼                                ▼
                                                   ┌──────────────────┐           ┌──────────────────┐
                                                   │  atlas.save()    │           │  HeatmapRenderer │
                                                   │  atlas.json      │           │  → atlas.html    │
                                                   │  (incremental)   │           │  (Plotly, hover) │
                                                   └──────────────────┘           └──────────────────┘
```

### Scoring algorithm

The risk score for any (question, domain, subdomain) is:

```
consistency  =  mean pairwise similarity across N samples
grounding    =  token-F1 between response and reference answer (if known)

risk_score   =  α × (1 − consistency)  +  β × (1 − grounding)
             where α = 0.7, β = 0.3  (grounding term omitted if no reference)
```

> **Phase 2** will replace the token-F1 heuristic with sentence-transformer embeddings (`all-MiniLM-L6-v2`) for semantic consistency scoring.

---

## 🔬 Probe Types

### `FactualProbe` — calibration baseline
Tests unambiguous facts with known answers (capitals, atomic numbers, historical dates). A well-functioning model should score near 0 here; elevated scores flag systemic issues.

```python
from hallucimap.probes import FactualProbe
probe = FactualProbe()
# → "What is the chemical symbol for water?"  ref: "H2O"
# → "How many sides does a hexagon have?"     ref: "6"
```

### `EntityProbe` — named entity knowledge
Probes biographical facts, founding dates, and key attributes of people, organizations, and places — a classic hallucination flashpoint where models invent plausible-but-wrong details.

```python
from hallucimap.probes import EntityProbe
probe = EntityProbe(entity_type="person")
# → "What year was Marie Curie born?"        ref: "1867"
# → "What university did Einstein attend?"   ref: "ETH Zurich"
```

### `DomainProbe` — deep domain knowledge
Targets specialized fields where overconfident confabulation is dangerous: biomedical, legal, financial, and scientific knowledge.

```python
from hallucimap.probes import DomainProbe
probe = DomainProbe(domain="medicine", subdomain="pharmacology")
# → "What is the antidote for acetaminophen overdose?"  ref: "N-acetylcysteine"
# → "What are SSRIs used to treat?"                     ref: "depression, anxiety"
```

### `TemporalProbe` — post-cutoff events
Tests knowledge of events after the model's training cutoff. A well-calibrated model should hedge; a hallucinating one will invent confident but fabricated details.

```python
from hallucimap.probes import TemporalProbe
probe = TemporalProbe(cutoff_year=2024, target_years=[2024, 2025])
# → "Who won the Nobel Prize in Physics in 2025?"
# → "What major AI models were released in 2025?"
```

---

## 🤖 Supported Models

| Provider | Models | Adapter |
|---|---|---|
| **OpenAI** | `gpt-4o`, `gpt-4-turbo`, `gpt-3.5-turbo` | `OpenAIAdapter` |
| **Anthropic** | `claude-3-5-sonnet-20241022`, `claude-opus-4-6`, `claude-3-5-haiku-20241022` | `AnthropicAdapter` |
| **HuggingFace** | Any local causal LM (Llama, Mistral, Phi…) | `HFAdapter` |

All adapters share the same async interface — swap models by changing one line.

```python
# Swap from OpenAI to Anthropic — nothing else changes
adapter = OpenAIAdapter(model="gpt-4o")
adapter = AnthropicAdapter(model="claude-3-5-sonnet-20241022")
adapter = HFAdapter(model="meta-llama/Llama-3-8B-Instruct", device="cuda")
```

---

## 📁 Project Structure

```
hallucimap/
├── src/hallucimap/
│   ├── core/
│   │   ├── atlas.py        ← RiskAtlas — load / update / save / query
│   │   ├── scorer.py       ← HallucinationScorer — consistency + grounding
│   │   └── topology.py     ← KnowledgeTopology — 2-D PCA/UMAP projection
│   ├── probes/
│   │   ├── base.py         ← BaseProbe (abstract)
│   │   ├── temporal.py     ← post-cutoff date facts
│   │   ├── entity.py       ← named entities (people, orgs, places)
│   │   ├── domain.py       ← domain knowledge (bio, law, finance)
│   │   └── factual.py      ← verifiable factual claims
│   ├── models/
│   │   ├── base.py         ← BaseLLMAdapter (abstract)
│   │   ├── openai_adapter.py
│   │   ├── anthropic_adapter.py
│   │   └── hf_adapter.py
│   ├── viz/
│   │   └── heatmap.py      ← Plotly interactive heatmap renderer
│   ├── testing.py          ← MockAdapter for downstream tests
│   └── cli.py              ← hallucimap scan / hallucimap show
├── tests/                  ← 53 tests, 63% coverage
├── examples/
│   ├── scan_gpt4o.py
│   └── scan_claude.py
└── .github/workflows/ci.yml
```

---

## 🛠️ Development

```bash
git clone https://github.com/advait27/hallucimap.git
cd hallucimap
pip install -e ".[dev]"
```

```bash
# Lint
ruff check src tests

# Type-check
mypy src

# Tests with coverage
pytest

# Full CI check (lint + types + tests)
ruff check src tests && mypy src && pytest
```

---

## 🗓️ Roadmap

| Phase | Status | Description |
|---|---|---|
| **0 — Scaffold** | ✅ Done | Package structure, Pydantic models, adapter stubs, CLI skeleton |
| **1 — Adapters** | ✅ Done | OpenAI, Anthropic, HuggingFace adapters with retry logic |
| **2 — Scorer** | 🔧 Next | Embedding-based consistency via `all-MiniLM-L6-v2` |
| **3 — Topology** | ⏳ Planned | UMAP projection of knowledge space; semantic clustering |
| **4 — Probe Datasets** | ⏳ Planned | TriviaQA, Wikidata, curated post-cutoff corpora |
| **5 — Heatmap v2** | ⏳ Planned | Topology-aware heatmap overlay; cluster annotations |
| **6 — CLI + Docs** | ⏳ Planned | Rich progress bars, `hallucimap summary`, hosted docs |
| **7 — PyPI** | ⏳ Planned | Publish to PyPI; versioned releases |

---

## 📄 License

MIT © [Advait Dharmadhikari](https://github.com/advait27) — see [LICENSE](LICENSE) for details.
