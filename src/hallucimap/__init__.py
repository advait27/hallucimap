"""hallucimap — cartograph hallucination risk across an LLM's knowledge space.

Quickstart
----------
>>> import asyncio
>>> from hallucimap import RiskAtlas, HallucinationScorer
>>> from hallucimap.models import AnthropicAdapter
>>> from hallucimap.probes import DomainProbe
>>>
>>> adapter = AnthropicAdapter(model="claude-3-5-sonnet-20241022")
>>> scorer = HallucinationScorer(adapter=adapter, n_samples=5)
>>> probe = DomainProbe(domain="science")
>>>
>>> atlas = RiskAtlas(model_id="claude-3-5-sonnet-20241022")
>>> results = asyncio.run(probe.run_all(adapter))
>>> atlas.update(results)
>>> atlas.save("my_atlas.json")
"""

from hallucimap.core.atlas import RiskAtlas
from hallucimap.core.scorer import HallucinationScorer
from hallucimap.core.topology import KnowledgeTopology

__all__ = [
    "RiskAtlas",
    "HallucinationScorer",
    "KnowledgeTopology",
]

__version__ = "0.1.0"
