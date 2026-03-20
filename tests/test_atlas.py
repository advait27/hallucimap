"""Tests for RiskAtlas, AtlasCell, and ScanSession."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from hallucimap.core.atlas import AtlasCell, RiskAtlas, ScanSession
from hallucimap.core.scorer import ScoredResponse


def make_response(domain: str, subdomain: str, risk: float) -> ScoredResponse:
    return ScoredResponse(
        question="Q",
        domain=domain,
        subdomain=subdomain,
        risk_score=risk,
        consistency_score=1.0 - risk,
        samples=["answer"],
        model_id="test",
    )


# ------------------------------------------------------------------ #
# AtlasCell                                                            #
# ------------------------------------------------------------------ #


class TestAtlasCell:
    def test_defaults(self) -> None:
        cell = AtlasCell(domain="science", subdomain="physics")
        assert cell.risk_score == 0.0
        assert cell.sample_count == 0

    def test_absorb_single(self) -> None:
        cell = AtlasCell(domain="science", subdomain="physics")
        resp = make_response("science", "physics", 0.6)
        cell.absorb(resp)
        assert cell.sample_count == 1
        assert cell.risk_score == pytest.approx(0.6)
        assert 0 < cell.confidence < 1

    def test_absorb_multiple_averages(self) -> None:
        cell = AtlasCell(domain="science", subdomain="physics")
        for risk in [0.2, 0.4, 0.6]:
            cell.absorb(make_response("science", "physics", risk))
        assert cell.sample_count == 3
        assert cell.risk_score == pytest.approx(0.4, abs=1e-6)

    def test_confidence_grows_with_samples(self) -> None:
        cell = AtlasCell(domain="science", subdomain="physics")
        prev_confidence = 0.0
        for i in range(1, 6):
            cell.absorb(make_response("science", "physics", 0.5))
            assert cell.confidence > prev_confidence
            prev_confidence = cell.confidence


# ------------------------------------------------------------------ #
# RiskAtlas                                                            #
# ------------------------------------------------------------------ #


class TestRiskAtlas:
    def test_empty_atlas(self, empty_atlas: RiskAtlas) -> None:
        assert len(empty_atlas.cells) == 0
        assert len(empty_atlas.sessions) == 0

    def test_update_creates_cells(self, empty_atlas: RiskAtlas) -> None:
        responses = [make_response("science", "physics", 0.3)]
        empty_atlas.update(responses)
        assert "science/physics" in empty_atlas.cells

    def test_update_with_session(self, empty_atlas: RiskAtlas) -> None:
        session = ScanSession(model_id="test")
        responses = [make_response("history", "wwii", 0.8)]
        empty_atlas.update(responses, session=session)
        assert len(empty_atlas.sessions) == 1
        assert empty_atlas.sessions[0].probe_count == 1
        assert empty_atlas.sessions[0].finished_at is not None

    def test_hottest_cells_sorted(self, multi_domain_atlas: RiskAtlas) -> None:
        hot = multi_domain_atlas.hottest_cells(n=2)
        assert len(hot) == 2
        assert hot[0].risk_score >= hot[1].risk_score

    def test_hottest_cells_less_than_n(self, populated_atlas: RiskAtlas) -> None:
        hot = populated_atlas.hottest_cells(n=100)
        assert len(hot) == len(populated_atlas.cells)

    def test_risk_matrix_shape(self, multi_domain_atlas: RiskAtlas) -> None:
        domains, subdomains, matrix = multi_domain_atlas.risk_matrix()
        assert matrix.shape == (len(domains), len(subdomains))

    def test_risk_matrix_values(self, multi_domain_atlas: RiskAtlas) -> None:
        domains, subdomains, matrix = multi_domain_atlas.risk_matrix()
        d_idx = {d: i for i, d in enumerate(domains)}
        s_idx = {s: i for i, s in enumerate(subdomains)}
        val = matrix[d_idx["history"], s_idx["wwii"]]
        assert val == pytest.approx(0.7, abs=1e-6)

    def test_risk_matrix_nan_for_missing(self, multi_domain_atlas: RiskAtlas) -> None:
        _, _, matrix = multi_domain_atlas.risk_matrix()
        # Not every (domain, subdomain) combo is filled
        assert np.any(np.isnan(matrix))

    def test_save_and_load_roundtrip(self, multi_domain_atlas: RiskAtlas) -> None:
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = Path(f.name)

        try:
            multi_domain_atlas.save(path)
            loaded = RiskAtlas.load(path)
            assert loaded.model_id == multi_domain_atlas.model_id
            assert set(loaded.cells.keys()) == set(multi_domain_atlas.cells.keys())
            for key, cell in loaded.cells.items():
                assert cell.risk_score == pytest.approx(
                    multi_domain_atlas.cells[key].risk_score, abs=1e-6
                )
        finally:
            path.unlink(missing_ok=True)

    def test_save_produces_valid_json(self, populated_atlas: RiskAtlas) -> None:
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = Path(f.name)
        try:
            populated_atlas.save(path)
            data = json.loads(path.read_text())
            assert "model_id" in data
            assert "cells" in data
        finally:
            path.unlink(missing_ok=True)

    def test_summary_non_empty(self, populated_atlas: RiskAtlas) -> None:
        s = populated_atlas.summary()
        assert populated_atlas.model_id in s
        assert "cells=" in s

    def test_summary_empty_atlas(self, empty_atlas: RiskAtlas) -> None:
        s = empty_atlas.summary()
        assert "cells=0" in s
