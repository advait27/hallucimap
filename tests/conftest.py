"""Shared pytest fixtures for hallucimap tests."""

from __future__ import annotations

import pytest

from hallucimap.core.atlas import RiskAtlas
from hallucimap.core.scorer import ScoredResponse
from hallucimap.testing import MockAdapter


@pytest.fixture
def mock_adapter() -> MockAdapter:
    return MockAdapter()


@pytest.fixture
def varied_adapter() -> MockAdapter:
    """Adapter that returns slightly varied answers to simulate inconsistency."""
    return MockAdapter(response="Paris, France")


@pytest.fixture
def empty_atlas() -> RiskAtlas:
    return RiskAtlas(model_id="test-model")


@pytest.fixture
def populated_atlas() -> RiskAtlas:
    atlas = RiskAtlas(model_id="test-model")
    responses = [
        ScoredResponse(
            question=f"Q{i}",
            domain="science",
            subdomain="physics",
            risk_score=0.1 * i,
            consistency_score=1.0 - 0.1 * i,
            samples=["answer"],
            model_id="test-model",
        )
        for i in range(1, 6)
    ]
    atlas.update(responses)
    return atlas


@pytest.fixture
def multi_domain_atlas() -> RiskAtlas:
    atlas = RiskAtlas(model_id="test-model")
    pairs = [
        ("science", "physics", 0.2),
        ("science", "chemistry", 0.5),
        ("history", "wwii", 0.7),
        ("medicine", "anatomy", 0.3),
    ]
    responses = [
        ScoredResponse(
            question=f"Q-{d}-{s}",
            domain=d,
            subdomain=s,
            risk_score=r,
            consistency_score=1.0 - r,
            samples=["answer"],
            model_id="test-model",
        )
        for d, s, r in pairs
    ]
    atlas.update(responses)
    return atlas
