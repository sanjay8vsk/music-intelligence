"""Tests for the speed-tolerant recognition cascade.

The cascade owns no DSP of its own, so these test ORCHESTRATION: when the sweep
runs, which correction wins, that thresholds stay separate, and above all that
the sign convention is the one the docstring claims. A sign error here would
look like a mysterious accuracy loss rather than a bug.
"""

from __future__ import annotations

import numpy as np
import pytest

from musicintel.eval import degradation as dg
from musicintel.recognition.cascade import (
    DEFAULT_RATE_GRID,
    CascadeConfig,
    Decision,
    apply_rate,
    identify_cascade,
)
from musicintel.recognition.decision import DecisionConfig, decide
from musicintel.recognition.fingerprint import FingerprintConfig, fingerprint
from musicintel.recognition.index import build_index
from musicintel.recognition.matcher import match

SR = 11025
CFG = FingerprintConfig()


def _music(seed=0, seconds=8.0, sr=SR):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    wobble = 600.0 + 200.0 * np.sin(2 * np.pi * 0.5 * t + seed)
    return (
        0.50 * np.sin(2 * np.pi * (440.0 + 7 * seed) * t)
        + 0.30 * np.sin(2 * np.pi * wobble * t)
        + 0.20 * np.sin(2 * np.pi * 1500.0 * t)
        + 0.02 * rng.standard_normal(t.size)
    ).astype(np.float32)


@pytest.fixture(scope="module")
def world():
    """A small catalog plus the audio it was built from."""
    tracks = {f"t{i}": _music(seed=i, seconds=20.0) for i in range(3)}
    idx = build_index([(k, fingerprint(v, SR, CFG)) for k, v in tracks.items()], config=CFG)
    return idx, tracks


def _cfg(**kw):
    base = dict(stage1_threshold=0.05, stage2_threshold=0.05, min_aligned_landmarks=5)
    base.update(kw)
    return CascadeConfig(**base)


# ------------------------------------------------------ rate / sign ---------
class TestRateConvention:
    def test_positive_rate_is_faster_and_higher(self):
        """Signal-level, independent of any recognizer behaviour."""
        t = np.linspace(0, 4.0, SR * 4, endpoint=False)
        tone = (0.5 * np.sin(2 * np.pi * 1000 * t)).astype(np.float32)

        def peak_hz(y):
            spec = np.abs(np.fft.rfft(y * np.hanning(len(y))))
            return float(np.fft.rfftfreq(len(y), 1 / SR)[int(np.argmax(spec))])

        up, down = apply_rate(tone, SR, 2.0), apply_rate(tone, SR, -2.0)
        assert len(up) < len(tone) < len(down)          # +2% is SHORTER
        assert peak_hz(down) < 1000.0 < peak_hz(up)      # +2% is HIGHER
        assert peak_hz(up) == pytest.approx(1020.0, abs=6)
        assert peak_hz(down) == pytest.approx(980.0, abs=6)

    def test_rate_zero_is_a_noop(self):
        y = _music(seconds=2.0)
        assert np.array_equal(apply_rate(y, SR, 0), y)

    def test_empty_audio_is_safe(self):
        assert apply_rate(np.zeros(0, np.float32), SR, 3.0).size == 0

    def test_unplayable_rate_raises(self):
        with pytest.raises(ValueError):
            apply_rate(_music(seconds=1.0), SR, -100.0)

    def test_a_faster_recording_is_corrected_by_a_negative_rate(self, world):
        """End-to-end, with the degradation produced by the eval harness's own
        independent implementation -- so this cannot pass on a sign that merely
        round-trips through apply_rate."""
        idx, tracks = world
        clean = tracks["t1"][SR * 4 : SR * 9]
        sped = dg.change_speed(clean, SR, 2.0)[0]  # played 2% FAST
        res = identify_cascade(sped, SR, idx, config=_cfg(stage1_threshold=0.9))
        assert res.is_match and res.track_id == "t1"
        assert res.rate_percent < 0, "a +2% recording must be corrected by a NEGATIVE rate"
        assert res.rate_percent == pytest.approx(-2.0, abs=1.0)

    def test_a_slower_recording_is_corrected_by_a_positive_rate(self, world):
        idx, tracks = world
        clean = tracks["t1"][SR * 4 : SR * 9]
        slow = dg.change_speed(clean, SR, -2.0)[0]
        res = identify_cascade(slow, SR, idx, config=_cfg(stage1_threshold=0.9))
        assert res.is_match and res.track_id == "t1"
        assert res.rate_percent > 0
        assert res.rate_percent == pytest.approx(2.0, abs=1.0)


# ------------------------------------------------------ short-circuit -------
class TestStage1ShortCircuit:
    def test_stage1_match_returns_without_sweeping(self, world):
        idx, tracks = world
        res = identify_cascade(tracks["t0"][: SR * 6], SR, idx, config=_cfg())
        assert res.is_match and res.stage == 1
        assert res.escalated is False
        assert res.rate_percent == 0.0
        assert res.timing.hypotheses_evaluated == 1   # stage 1 only
        assert res.timing.stage2_ms == 0.0

    def test_stage1_match_is_identical_to_the_plain_decision_layer(self, world):
        """Stage-1 behaviour must be unchanged by the cascade wrapper."""
        idx, tracks = world
        y = tracks["t2"][: SR * 6]
        plain = decide(match(fingerprint(y, SR, CFG), idx),
                       config=DecisionConfig(threshold=0.05, min_aligned_landmarks=5))
        res = identify_cascade(y, SR, idx, config=_cfg())
        assert plain.is_match and res.is_match
        assert res.track_id == plain.track_id
        assert res.evidence_score == plain.evidence_score
        assert res.aligned_landmarks == plain.aligned_landmarks

    def test_no_match_escalates(self, world):
        idx, _ = world
        rng = np.random.default_rng(4)
        res = identify_cascade(rng.standard_normal(SR * 5).astype(np.float32), SR, idx,
                               config=_cfg(stage2_threshold=0.9))
        assert res.escalated is True
        assert res.timing.hypotheses_evaluated == len(DEFAULT_RATE_GRID) + 1
        assert res.decision is Decision.NO_MATCH


# ------------------------------------------------------ hypotheses ----------
class TestHypotheses:
    def test_every_grid_rate_is_evaluated_exactly_once(self, world):
        idx, _ = world
        rng = np.random.default_rng(5)
        res = identify_cascade(rng.standard_normal(SR * 4).astype(np.float32), SR, idx,
                               config=_cfg(stage2_threshold=0.9))
        rates = [h.rate_percent for h in res.hypotheses]
        assert rates[0] == 0.0                       # stage 1, reused
        assert sorted(rates[1:]) == sorted(DEFAULT_RATE_GRID)
        assert len(rates) == len(set(rates))

    def test_rate_zero_is_reused_not_recomputed(self, world):
        """Stage 1 already ran rate 0; recomputing it would be wasted work."""
        idx, _ = world
        rng = np.random.default_rng(6)
        res = identify_cascade(rng.standard_normal(SR * 4).astype(np.float32), SR, idx,
                               config=_cfg(stage2_threshold=0.9))
        zero = [h for h in res.hypotheses if h.rate_percent == 0.0]
        assert len(zero) == 1
        assert zero[0].evidence == res.stage1_decision.evidence_score

    def test_a_grid_containing_zero_is_rejected(self):
        with pytest.raises(ValueError, match="must not contain 0"):
            _cfg(rate_grid=(-1.0, 0.0, 1.0)).validate()

    def test_duplicate_grid_rates_are_rejected(self):
        with pytest.raises(ValueError, match="duplicates"):
            _cfg(rate_grid=(1.0, 1.0)).validate()

    def test_best_hypothesis_is_the_highest_evidence(self, world):
        idx, tracks = world
        sped = dg.change_speed(tracks["t0"][SR * 3 : SR * 8], SR, 3.0)[0]
        res = identify_cascade(sped, SR, idx, config=_cfg(stage1_threshold=0.9))
        usable = [h for h in res.hypotheses if h.ok and h.track_id]
        assert res.evidence_score == max(h.evidence for h in usable)

    def test_one_failing_hypothesis_does_not_sink_the_query(self, world):
        """An unplayable rate must be recorded as an error, not raised."""
        idx, tracks = world
        sped = dg.change_speed(tracks["t1"][SR * 4 : SR * 9], SR, 2.0)[0]
        res = identify_cascade(sped, SR, idx,
                               config=_cfg(stage1_threshold=0.9,
                                           rate_grid=(-100.0, -2.0, 2.0)))
        failed = [h for h in res.hypotheses if not h.ok]
        assert len(failed) == 1 and failed[0].rate_percent == -100.0
        assert res.is_match and res.track_id == "t1"   # the rest still worked


# ------------------------------------------------------ thresholds ----------
class TestThresholdSeparation:
    def test_stage1_and_stage2_thresholds_are_independent(self, world):
        idx, tracks = world
        sped = dg.change_speed(tracks["t2"][SR * 4 : SR * 9], SR, 2.0)[0]
        loose = identify_cascade(sped, SR, idx,
                                 config=_cfg(stage1_threshold=0.9, stage2_threshold=0.02))
        strict = identify_cascade(sped, SR, idx,
                                  config=_cfg(stage1_threshold=0.9, stage2_threshold=0.99))
        assert loose.is_match and loose.stage == 2
        assert not strict.is_match
        assert strict.escalated is True   # it still swept; it just refused

    def test_a_strict_stage2_never_weakens_stage1(self, world):
        """Raising the stage-2 bar must not affect a query stage 1 already took."""
        idx, tracks = world
        y = tracks["t0"][: SR * 6]
        a = identify_cascade(y, SR, idx, config=_cfg(stage2_threshold=0.01))
        b = identify_cascade(y, SR, idx, config=_cfg(stage2_threshold=1.0))
        assert a.stage == b.stage == 1 and a.track_id == b.track_id

    def test_rejected_winner_is_withheld(self, world):
        idx, _ = world
        rng = np.random.default_rng(7)
        res = identify_cascade(rng.standard_normal(SR * 5).astype(np.float32), SR, idx,
                               config=_cfg(stage2_threshold=0.99))
        assert res.decision is Decision.NO_MATCH
        assert res.track_id is None
        assert res.stage is None

    def test_config_validation(self):
        for bad in (dict(stage1_threshold=-0.1), dict(stage2_threshold=1.5),
                    dict(min_aligned_landmarks=0)):
            with pytest.raises(ValueError):
                _cfg(**bad).validate()


# ------------------------------------------------------ determinism ---------
class TestDeterminism:
    def test_same_input_gives_the_same_result(self, world):
        idx, tracks = world
        sped = dg.change_speed(tracks["t1"][SR * 4 : SR * 9], SR, 2.0)[0]
        a = identify_cascade(sped, SR, idx, config=_cfg(stage1_threshold=0.9))
        b = identify_cascade(sped, SR, idx, config=_cfg(stage1_threshold=0.9))
        assert a.to_dict() == b.to_dict()
        assert [h.evidence for h in a.hypotheses] == [h.evidence for h in b.hypotheses]

    def test_hypothesis_order_follows_the_grid(self, world):
        idx, _ = world
        rng = np.random.default_rng(8)
        y = rng.standard_normal(SR * 4).astype(np.float32)
        grid = (-3.0, 2.0, -1.0)
        res = identify_cascade(y, SR, idx, config=_cfg(stage2_threshold=0.9, rate_grid=grid))
        assert [h.rate_percent for h in res.hypotheses] == [0.0, *grid]

    def test_ties_break_toward_the_smallest_correction(self, world):
        """Two equal-evidence hypotheses must resolve the same way every time."""
        idx, tracks = world
        y = tracks["t0"][: SR * 6]
        res = identify_cascade(y, SR, idx, config=_cfg(stage1_threshold=0.9,
                                                       stage2_threshold=0.0,
                                                       rate_grid=(-1.0, 1.0)))
        assert res.rate_percent in (-1.0, 0.0, 1.0)
        again = identify_cascade(y, SR, idx, config=_cfg(stage1_threshold=0.9,
                                                         stage2_threshold=0.0,
                                                         rate_grid=(-1.0, 1.0)))
        assert res.rate_percent == again.rate_percent


# ------------------------------------------------------ contract -----------
class TestContract:
    def test_default_grid_is_pm5_at_1pct_and_excludes_zero(self):
        assert DEFAULT_RATE_GRID == (-5.0, -4.0, -3.0, -2.0, -1.0, 1.0, 2.0, 3.0, 4.0, 5.0)
        assert 0.0 not in DEFAULT_RATE_GRID
        assert len(DEFAULT_RATE_GRID) == 10

    def test_result_exposes_stage_rate_and_evidence(self, world):
        idx, tracks = world
        res = identify_cascade(tracks["t0"][: SR * 6], SR, idx, config=_cfg())
        d = res.to_dict()
        for k in ("decision", "track_id", "stage", "rate_percent", "evidence_score",
                  "threshold", "aligned_landmarks", "escalated", "hypotheses_evaluated"):
            assert k in d
        assert d["decision"] == "MATCH"

    def test_no_probability_surface(self, world):
        idx, tracks = world
        res = identify_cascade(tracks["t0"][: SR * 6], SR, idx, config=_cfg())
        for banned in ("confidence", "probability", "certainty"):
            assert not hasattr(res, banned)
