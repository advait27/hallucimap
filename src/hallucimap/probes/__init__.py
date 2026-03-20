"""Built-in probe types for hallucimap.

Each probe class generates a set of questions targeting a specific
knowledge vulnerability and can be run against any :class:`BaseLLMAdapter`.
"""

from hallucimap.probes.base import BaseProbe, ProbeResult
from hallucimap.probes.domain import DomainProbe
from hallucimap.probes.entity import EntityProbe
from hallucimap.probes.factual import FactualProbe
from hallucimap.probes.temporal import TemporalProbe

__all__ = [
    "BaseProbe",
    "ProbeResult",
    "DomainProbe",
    "EntityProbe",
    "FactualProbe",
    "TemporalProbe",
]
