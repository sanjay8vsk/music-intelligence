"""Tests for the gated speed-tolerant cascade.

Orchestration only: which steps run, in what order, and which ones stop the
pipeline. The recognizer underneath is frozen and tested elsewhere.
"""

from __future__ import annotations

import numpy as np
import pytest

from musicintel.eval import degradation as dg
from musicintel.recognition.decision import Decision, DecisionConfig, decide
from musicintel.recognition.fingerprint import FingerprintConfig, fingerprint
from musicintel.recognition.gated_cascade import (
    GATED_RATE_GRID,
    PROBE_SECONDS,
    GatedCascadeConfig,
    gate_value_of,
    identify_gated,
)
from musicintel.recognition.index import build_index
from musicintel.recognition.matcher import match

SR = 11025
CFG = FingerprintConfig()


def _music(seed=0, seconds=8.0):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, seconds, int(SR * seconds), endpoint=False)
    wob = 600.0 + 200.0 * np.sin(2 * np.pi * 0.5 * t + seed)
    return (0.50 * np.sin(2 * np.pi * (440.0 + 7 * seed) * t)
            + 0.30 * np.sin(2 * np.pi * wob * t)
            + 0.20 * np.sin(2 * np.pi * 1500.0 * t)
            + 0.02 * rng.standard_normal(t.size)).astype(np.float32)


@pytest.fixture(scope="module")
def world():
    tracks = {f"t{i}": _music(seed=i, seconds=20.0) for i in range(3)}
    idx = build_index([(k, fingerprint(v, SR, CFG)) for k, v in tracks.items()], config=CFG)
    return idx, tracks


def _cfg(**kw):
    base = dict(stage1_threshold=0.05, gate_threshold=0.0,
                stage2_threshold=0.05, min_aligned_landmarks=5)
    base.update(kw)
    return GatedCascadeConfig(**base)


# 1 ------------------------------------------------------------------------
class TestStage1ShortCircuit:
    def test_stage1_match_stops_everything(self, world):
        idx, tracks = world
        res = identify_gated(tracks["t0"][: SR * 6], SR, idx, config=_cfg())
        assert res.is_match and res.stage == 1
        assert res.escalated is False
        assert res.gate_passed is False and res.probe_passed is False
        assert res.probe_hypotheses == ()
        assert res.timing.probe_ms == 0.0 and res.timing.confirm_ms == 0.0
        assert res.timing.hypotheses_evaluated == 1

    # 10 --------------------------------------------------------------------
    def test_stage1_behaviour_is_unchanged(self, world):
        """The wrapper must not alter what stage 1 decides or reports."""
        idx, tracks = world
        y = tracks["t2"][: SR * 6]
        plain = decide(match(fingerprint(y, SR, CFG), idx),
                       config=DecisionConfig(threshold=0.05, min_aligned_landmarks=5))
        res = identify_gated(y, SR, idx, config=_cfg())
        assert plain.is_match and res.is_match
        assert res.track_id == plain.track_id
        assert res.evidence_score == plain.evidence_score
        assert res.aligned_landmarks == plain.aligned_landmarks
        assert res.threshold == 0.05


# 2 ------------------------------------------------------------------------
class TestConcentrationGate:
    def test_failed_gate_skips_the_probe(self, world):
        """Out-of-catalog music is the population the gate exists to skip: many
        matched landmarks, few of them agreeing, so concentration sits well
        below 1.0 and a gate above it genuinely closes.

        (Pure noise is a poor fixture here -- it can match a single landmark
        that trivially "agrees", giving concentration exactly 1.0.)
        """
        idx, _ = world
        outsider = _music(seed=99, seconds=6.0)
        seen = identify_gated(outsider, SR, idx, config=_cfg(gate_threshold=0.0))
        assert not (seen.is_match and seen.stage == 1)   # it must reach the gate
        assert seen.gate_value < 1.0
        assert seen.gate_passed and len(seen.probe_hypotheses) == len(GATED_RATE_GRID)

        res = identify_gated(outsider, SR, idx,
                             config=_cfg(gate_threshold=seen.gate_value + 0.01))
        assert not res.is_match
        assert res.gate_passed is False
        assert res.probe_hypotheses == ()          # probe never ran
        assert res.timing.probe_ms == 0.0
        assert res.timing.confirm_ms == 0.0
        assert res.escalated is False              # no sweep work was done

    def test_open_gate_lets_the_probe_run(self, world):
        idx, _ = world
        rng = np.random.default_rng(3)
        noise = rng.standard_normal(SR * 5).astype(np.float32)
        res = identify_gated(noise, SR, idx, config=_cfg(gate_threshold=0.0))
        assert res.gate_passed is True
        assert len(res.probe_hypotheses) == len(GATED_RATE_GRID)

    def test_gate_reads_stage1_concentration_for_free(self, world):
        idx, tracks = world
        sped = dg.change_speed(tracks["t1"][SR * 4 : SR * 9], SR, 2.0)[0]
        res = identify_gated(sped, SR, idx, config=_cfg(stage1_threshold=0.9,
                                                        gate_threshold=0.0))
        assert res.gate_value == gate_value_of(res.stage1_decision)
        assert 0.0 <= res.gate_value <= 1.0


# 3 ------------------------------------------------------------------------
class TestProbeGate:
    def test_probe_with_no_usable_candidate_skips_confirmation(self, world):
        idx, _ = world
        # Digital silence yields no landmarks at any rate, so no candidate.
        res = identify_gated(np.zeros(SR * 5, np.float32), SR, idx,
                             config=_cfg(gate_threshold=0.0))
        assert not res.is_match
        assert res.probe_passed is False
        assert res.timing.confirm_ms == 0.0

    def test_probe_uses_only_the_first_two_seconds(self, world):
        idx, tracks = world
        sped = dg.change_speed(tracks["t1"][SR * 4 : SR * 12], SR, 2.0)[0]
        res = identify_gated(sped, SR, idx, config=_cfg(stage1_threshold=0.9))
        probe_lm = max(h.query_landmarks for h in res.probe_hypotheses)
        # The probe is ~2 s of an ~8 s query, so far fewer landmarks than the
        # full-query confirmation reports.
        assert probe_lm < res.query_landmarks
        assert PROBE_SECONDS == 2.0


# 4, 5 ---------------------------------------------------------------------
class TestSpeedRecovery:
    def test_speed_change_reaches_the_sweep_and_is_recovered(self, world):
        idx, tracks = world
        sped = dg.change_speed(tracks["t1"][SR * 4 : SR * 12], SR, 2.0)[0]
        res = identify_gated(sped, SR, idx, config=_cfg(stage1_threshold=0.9))
        assert res.gate_passed and res.probe_passed
        assert res.is_match and res.stage == 2 and res.track_id == "t1"

    def test_positive_speed_is_corrected_by_a_negative_rate(self, world):
        """Degradation comes from the harness's own independent implementation,
        so this cannot pass on a sign that merely round-trips."""
        idx, tracks = world
        sped = dg.change_speed(tracks["t1"][SR * 4 : SR * 12], SR, 2.0)[0]
        res = identify_gated(sped, SR, idx, config=_cfg(stage1_threshold=0.9))
        assert res.rate_percent < 0
        assert res.rate_percent == pytest.approx(-2.0, abs=2.0)

    def test_negative_speed_is_corrected_by_a_positive_rate(self, world):
        idx, tracks = world
        slow = dg.change_speed(tracks["t2"][SR * 4 : SR * 12], SR, -2.0)[0]
        res = identify_gated(slow, SR, idx, config=_cfg(stage1_threshold=0.9))
        assert res.is_match and res.rate_percent > 0


# 6 ------------------------------------------------------------------------
class TestGrid:
    def test_grid_is_exactly_minus4_minus2_plus2_plus4(self):
        assert GATED_RATE_GRID == (-4.0, -2.0, 2.0, 4.0)
        assert 0.0 not in GATED_RATE_GRID and len(GATED_RATE_GRID) == 4

    def test_probe_evaluates_every_grid_rate_once(self, world):
        idx, _ = world
        rng = np.random.default_rng(9)
        res = identify_gated(rng.standard_normal(SR * 4).astype(np.float32), SR, idx,
                             config=_cfg(gate_threshold=0.0))
        rates = [h.rate_percent for h in res.probe_hypotheses]
        assert sorted(rates) == sorted(GATED_RATE_GRID)
        assert len(rates) == len(set(rates))

    def test_zero_or_duplicate_rates_are_rejected(self):
        with pytest.raises(ValueError, match="must not contain 0"):
            _cfg(rate_grid=(-2.0, 0.0, 2.0)).validate()
        with pytest.raises(ValueError, match="duplicates"):
            _cfg(rate_grid=(2.0, 2.0)).validate()


# 7 ------------------------------------------------------------------------
class TestThresholdIndependence:
    def test_stage2_threshold_is_independent_of_stage1(self, world):
        idx, tracks = world
        sped = dg.change_speed(tracks["t1"][SR * 4 : SR * 12], SR, 2.0)[0]
        loose = identify_gated(sped, SR, idx,
                               config=_cfg(stage1_threshold=0.9, stage2_threshold=0.02))
        strict = identify_gated(sped, SR, idx,
                                config=_cfg(stage1_threshold=0.9, stage2_threshold=0.99))
        assert loose.is_match and loose.stage == 2
        assert not strict.is_match
        assert strict.gate_passed and strict.probe_passed  # it ran, then refused

    def test_a_strict_stage2_cannot_disturb_a_stage1_match(self, world):
        idx, tracks = world
        y = tracks["t0"][: SR * 6]
        a = identify_gated(y, SR, idx, config=_cfg(stage2_threshold=0.01))
        b = identify_gated(y, SR, idx, config=_cfg(stage2_threshold=1.0))
        assert a.stage == b.stage == 1 and a.track_id == b.track_id

    def test_config_validation(self):
        for bad in (dict(stage1_threshold=-0.1), dict(stage2_threshold=1.5),
                    dict(gate_threshold=2.0), dict(min_aligned_landmarks=0),
                    dict(probe_seconds=0.0)):
            with pytest.raises(ValueError):
                _cfg(**bad).validate()


# 8 ------------------------------------------------------------------------
class TestRejectionWithholdsTheWinner:
    def test_gate_rejection_returns_no_track(self, world):
        idx, _ = world
        rng = np.random.default_rng(11)
        res = identify_gated(rng.standard_normal(SR * 5).astype(np.float32), SR, idx,
                             config=_cfg(gate_threshold=1.0))
        assert res.decision is Decision.NO_MATCH
        assert res.track_id is None and res.stage is None

    def test_stage2_rejection_returns_no_track(self, world):
        idx, tracks = world
        sped = dg.change_speed(tracks["t1"][SR * 4 : SR * 12], SR, 2.0)[0]
        res = identify_gated(sped, SR, idx,
                             config=_cfg(stage1_threshold=0.9, stage2_threshold=0.99))
        assert res.decision is Decision.NO_MATCH
        assert res.track_id is None and res.stage is None

    def test_no_probability_surface(self, world):
        idx, tracks = world
        res = identify_gated(tracks["t0"][: SR * 6], SR, idx, config=_cfg())
        for banned in ("confidence", "probability", "certainty"):
            assert not hasattr(res, banned)


# 9 ------------------------------------------------------------------------
class TestDeterminism:
    def test_repeated_runs_are_identical(self, world):
        idx, tracks = world
        sped = dg.change_speed(tracks["t1"][SR * 4 : SR * 12], SR, 2.0)[0]
        a = identify_gated(sped, SR, idx, config=_cfg(stage1_threshold=0.9))
        b = identify_gated(sped, SR, idx, config=_cfg(stage1_threshold=0.9))
        assert a.to_dict() == b.to_dict()
        assert [h.evidence for h in a.probe_hypotheses] == \
               [h.evidence for h in b.probe_hypotheses]

    def test_winner_is_the_highest_probe_evidence(self, world):
        idx, tracks = world
        sped = dg.change_speed(tracks["t0"][SR * 3 : SR * 11], SR, 4.0)[0]
        res = identify_gated(sped, SR, idx, config=_cfg(stage1_threshold=0.9))
        usable = [h for h in res.probe_hypotheses if h.ok and h.track_id and h.aligned >= 5]
        if usable:
            assert res.rate_percent == min(
                usable, key=lambda h: (-h.evidence, abs(h.rate_percent), h.rate_percent)
            ).rate_percent

    def test_probe_order_follows_the_grid(self, world):
        idx, _ = world
        rng = np.random.default_rng(12)
        res = identify_gated(rng.standard_normal(SR * 4).astype(np.float32), SR, idx,
                             config=_cfg(gate_threshold=0.0))
        assert [h.rate_percent for h in res.probe_hypotheses] == list(GATED_RATE_GRID)

    def test_one_failing_rate_does_not_sink_the_query(self, world):
        idx, tracks = world
        sped = dg.change_speed(tracks["t1"][SR * 4 : SR * 12], SR, 2.0)[0]
        res = identify_gated(sped, SR, idx,
                             config=_cfg(stage1_threshold=0.9,
                                         rate_grid=(-100.0, -2.0, 2.0)))
        assert any(not h.ok for h in res.probe_hypotheses)
        assert res.is_match and res.track_id == "t1"
