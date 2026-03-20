"""Tests for HallucinationScorer."""

from __future__ import annotations

import pytest

from hallucimap.core.scorer import HallucinationScorer, ScoredResponse
from hallucimap.testing import MockAdapter


class TestHallucinationScorer:
    def test_init_rejects_zero_temperature(self, mock_adapter: MockAdapter) -> None:
        with pytest.raises(ValueError, match="temperature"):
            HallucinationScorer(adapter=mock_adapter, temperature=0.0)

    @pytest.mark.asyncio
    async def test_score_returns_scored_response(self, mock_adapter: MockAdapter) -> None:
        scorer = HallucinationScorer(adapter=mock_adapter, n_samples=3)
        result = await scorer.score("What is the capital of France?", "geography", "europe")
        assert isinstance(result, ScoredResponse)
        assert result.domain == "geography"
        assert result.subdomain == "europe"
        assert 0.0 <= result.risk_score <= 1.0

    @pytest.mark.asyncio
    async def test_score_consistent_adapter_low_risk(self, mock_adapter: MockAdapter) -> None:
        """A mock adapter that always returns the same answer should score low risk."""
        scorer = HallucinationScorer(adapter=mock_adapter, n_samples=5)
        result = await scorer.score("Q?", "test", "test")
        # All samples identical → high consistency → low risk
        assert result.consistency_score > 0.5

    @pytest.mark.asyncio
    async def test_score_with_matching_reference(self, mock_adapter: MockAdapter) -> None:
        scorer = HallucinationScorer(adapter=mock_adapter, n_samples=3)
        # mock_adapter returns "Paris"; reference is also "Paris"
        result = await scorer.score("Capital of France?", "geo", "europe", reference="Paris")
        assert result.grounding_score is not None
        assert result.grounding_score > 0.0

    @pytest.mark.asyncio
    async def test_score_with_wrong_reference_raises_risk(self, mock_adapter: MockAdapter) -> None:
        scorer = HallucinationScorer(adapter=mock_adapter, n_samples=3)
        # mock returns "Paris" but reference is "London" → grounding penalty
        result_wrong = await scorer.score(
            "Capital of UK?", "geo", "europe", reference="London"
        )
        result_right = await scorer.score(
            "Capital of France?", "geo", "europe", reference="Paris"
        )
        assert result_wrong.risk_score >= result_right.risk_score

    @pytest.mark.asyncio
    async def test_score_batch_length(self, mock_adapter: MockAdapter) -> None:
        scorer = HallucinationScorer(adapter=mock_adapter, n_samples=2)
        questions = [("Q1", "science", "physics"), ("Q2", "history", "wwii")]
        results = await scorer.score_batch(questions)
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_score_batch_preserves_order(self, mock_adapter: MockAdapter) -> None:
        scorer = HallucinationScorer(adapter=mock_adapter, n_samples=2)
        questions = [(f"Q{i}", "domain", "sub") for i in range(10)]
        results = await scorer.score_batch(questions)
        for i, result in enumerate(results):
            assert result.question == f"Q{i}"

    @pytest.mark.asyncio
    async def test_samples_populated(self, mock_adapter: MockAdapter) -> None:
        scorer = HallucinationScorer(adapter=mock_adapter, n_samples=4)
        result = await scorer.score("Q?", "d", "s")
        assert len(result.samples) == 4

    @pytest.mark.asyncio
    async def test_model_id_propagated(self, mock_adapter: MockAdapter) -> None:
        scorer = HallucinationScorer(adapter=mock_adapter, n_samples=2)
        result = await scorer.score("Q?", "d", "s")
        assert result.model_id == mock_adapter.model_id

    def test_composite_risk_no_grounding(self, mock_adapter: MockAdapter) -> None:
        scorer = HallucinationScorer(adapter=mock_adapter)
        risk = scorer._composite_risk(consistency=0.8, grounding=None)
        assert risk == pytest.approx(0.2, abs=1e-6)

    def test_composite_risk_with_perfect_grounding(self, mock_adapter: MockAdapter) -> None:
        scorer = HallucinationScorer(adapter=mock_adapter, alpha=0.7, beta=0.3)
        risk = scorer._composite_risk(consistency=1.0, grounding=1.0)
        assert risk == pytest.approx(0.0, abs=1e-6)

    def test_composite_risk_clamped_to_unit_interval(self, mock_adapter: MockAdapter) -> None:
        scorer = HallucinationScorer(adapter=mock_adapter, alpha=1.0, beta=1.0)
        risk = scorer._composite_risk(consistency=0.0, grounding=0.0)
        assert 0.0 <= risk <= 1.0
