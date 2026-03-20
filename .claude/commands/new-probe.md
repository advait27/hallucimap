# /new-probe

Scaffold a new probe class in `src/hallucimap/probes/`.

## Usage

```
/new-probe <ProbeClassName> [domain_hint]
```

Examples:
- `/new-probe ChemistryProbe chemistry`
- `/new-probe LegalCaseProbe law`

## What this command does

1. Creates `src/hallucimap/probes/<snake_name>.py` with a subclass of `BaseProbe`.
2. Adds the class to `src/hallucimap/probes/__init__.py` exports.
3. Creates `tests/test_probe_<snake_name>.py` with a pytest stub.
4. Prints a reminder of which `ProbeResult` fields to populate.

## Template to follow

```python
from __future__ import annotations

import asyncio
from typing import AsyncIterator

from hallucimap.probes.base import BaseProbe, ProbeResult


class $ClassName(BaseProbe):
    """One-line description.

    Parameters
    ----------
    domain : str
        Domain label used for atlas bucketing.

    Examples
    --------
    >>> probe = $ClassName()
    >>> results = asyncio.run(probe.run_all(adapter))
    """

    domain: str = "$domain_hint"

    async def generate_questions(self) -> AsyncIterator[str]:
        # TODO: yield probe questions one by one
        raise NotImplementedError

    async def score_response(self, question: str, response: str) -> ProbeResult:
        # TODO: parse response and return a ProbeResult
        raise NotImplementedError
```
