"""Tests for all probe types."""

from __future__ import annotations

import pytest

from hallucimap.probes import DomainProbe, EntityProbe, FactualProbe, TemporalProbe
from hallucimap.probes.base import ProbeResult
from hallucimap.testing import MockAdapter


# ------------------------------------------------------------------ #
# Helpers                                                              #
# ------------------------------------------------------------------ #


async def collect_questions(probe: object) -> list[str]:
    """Drain an async generator from a probe and return all questions."""
    from hallucimap.probes.base import BaseProbe

    assert isinstance(probe, BaseProbe)
    questions = []
    async for q in probe.generate_questions():
        questions.append(q)
    return questions


# ------------------------------------------------------------------ #
# FactualProbe                                                         #
# ------------------------------------------------------------------ #


class TestFactualProbe:
    @pytest.mark.asyncio
    async def test_generates_questions(self) -> None:
        probe = FactualProbe()
        questions = await collect_questions(probe)
        assert len(questions) > 0

    @pytest.mark.asyncio
    async def test_max_questions_respected(self) -> None:
        probe = FactualProbe(max_questions=3)
        results = await probe.run_all(MockAdapter(), concurrency=2)
        assert len(results) <= 3

    @pytest.mark.asyncio
    async def test_score_response_returns_probe_result(self) -> None:
        probe = FactualProbe()
        result = await probe.score_response("What year did World War II end?", "1945")
        assert isinstance(result, ProbeResult)
        assert result.domain == "factual"
        assert result.reference == "1945"

    @pytest.mark.asyncio
    async def test_run_all_length(self) -> None:
        probe = FactualProbe()
        all_qs = await collect_questions(probe)
        results = await probe.run_all(MockAdapter())
        assert len(results) == len(all_qs)


# ------------------------------------------------------------------ #
# EntityProbe                                                          #
# ------------------------------------------------------------------ #


class TestEntityProbe:
    @pytest.mark.asyncio
    async def test_person_domain(self) -> None:
        probe = EntityProbe(entity_type="person")
        questions = await collect_questions(probe)
        assert len(questions) > 0

    @pytest.mark.asyncio
    async def test_org_domain(self) -> None:
        probe = EntityProbe(entity_type="organization")
        questions = await collect_questions(probe)
        assert len(questions) > 0

    @pytest.mark.asyncio
    async def test_place_domain(self) -> None:
        probe = EntityProbe(entity_type="place")
        questions = await collect_questions(probe)
        assert len(questions) > 0

    @pytest.mark.asyncio
    async def test_score_response_has_reference_when_known(self) -> None:
        probe = EntityProbe(entity_type="person")
        result = await probe.score_response("What year was Marie Curie born?", "1867")
        assert result.reference == "1867"

    @pytest.mark.asyncio
    async def test_score_response_no_reference_when_unknown(self) -> None:
        probe = EntityProbe(entity_type="person")
        result = await probe.score_response("What is Elon Musk's middle name?", "Reeve")
        assert result.reference is None

    @pytest.mark.asyncio
    async def test_subdomain_matches_entity_type(self) -> None:
        for etype in ("person", "organization", "place"):
            probe = EntityProbe(entity_type=etype)  # type: ignore[arg-type]
            questions = await collect_questions(probe)
            if questions:
                result = await probe.score_response(questions[0], "answer")
                assert result.subdomain == etype


# ------------------------------------------------------------------ #
# DomainProbe                                                          #
# ------------------------------------------------------------------ #


class TestDomainProbe:
    @pytest.mark.asyncio
    async def test_generates_questions_for_known_domain(self) -> None:
        probe = DomainProbe(domain="science")
        questions = await collect_questions(probe)
        assert len(questions) > 0

    @pytest.mark.asyncio
    async def test_subdomain_filter(self) -> None:
        probe_all = DomainProbe(domain="science")
        probe_sub = DomainProbe(domain="science", subdomain="chemistry")
        q_all = await collect_questions(probe_all)
        q_sub = await collect_questions(probe_sub)
        assert len(q_sub) <= len(q_all)
        assert len(q_sub) > 0

    @pytest.mark.asyncio
    async def test_unknown_domain_yields_nothing(self) -> None:
        probe = DomainProbe(domain="foobar_nonexistent")
        questions = await collect_questions(probe)
        assert questions == []

    @pytest.mark.asyncio
    async def test_reference_populated_for_known_answer(self) -> None:
        probe = DomainProbe(domain="science", subdomain="chemistry")
        result = await probe.score_response("What is the atomic number of gold?", "79")
        assert result.reference == "79"

    @pytest.mark.asyncio
    async def test_run_all_returns_probe_results(self) -> None:
        probe = DomainProbe(domain="medicine")
        results = await probe.run_all(MockAdapter(), concurrency=3)
        assert all(isinstance(r, ProbeResult) for r in results)


# ------------------------------------------------------------------ #
# TemporalProbe                                                        #
# ------------------------------------------------------------------ #


class TestTemporalProbe:
    @pytest.mark.asyncio
    async def test_generates_questions(self) -> None:
        probe = TemporalProbe(cutoff_year=2023)
        questions = await collect_questions(probe)
        assert len(questions) > 0

    @pytest.mark.asyncio
    async def test_target_years_control_output(self) -> None:
        probe_one = TemporalProbe(cutoff_year=2023, target_years=[2024])
        probe_two = TemporalProbe(cutoff_year=2023, target_years=[2024, 2025])
        q_one = await collect_questions(probe_one)
        q_two = await collect_questions(probe_two)
        # Two years should produce double the questions
        assert len(q_two) == 2 * len(q_one)

    @pytest.mark.asyncio
    async def test_domain_is_temporal(self) -> None:
        probe = TemporalProbe()
        questions = await collect_questions(probe)
        result = await probe.score_response(questions[0], "I don't know.")
        assert result.domain == "temporal"
