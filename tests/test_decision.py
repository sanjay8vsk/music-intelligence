"""Focused tests for the MATCH / NO_MATCH decision layer.

These test the DECISION RULE, not the accuracy of the system. No test here
asserts a recall or FAR figure: those are measurements, they belong in
eval/reports/phase1d_baseline.md, and a unit test that hard-codes them would
just be a fragile copy of a number that is supposed to move.
"""

from __future__ import annotations

import numpy as np
import pytest

from musicintel.recognition.decision import (
    DEFAULT_DECISION_CONFIG,
    Decision,
    DecisionConfig,
    MatchDecision,
    decide,
    evidence_score,
    identify,
)
from musicintel.recognition.fingerprint import (
    FingerprintConfig,
    FingerprintResult,
    fingerprint,
)
from musicintel.recognition.index import build_index
from musicintel.recognition.matcher import match

SR = 11025
CFG = FingerprintConfig()
# An explicit threshold for the unit tests. NOT the calibrated operating point
# -- that lives in the Phase 1D report and is a measurement, not a constant.
TEST_CFG = DecisionConfig(threshold=0.05, min_aligned_landmarks=5)


# --------------------------------------------------------------- fixtures ----
def _fp(pairs, *, duration=30.0, config=CFG):
    pairs = sorted((int(h), int(a)) for h, a in pairs)
    return FingerprintResult(
        hashes=np.array([h for h, _ in pairs], dtype=np.uint32),
        anchor_frames=np.array([a for _, a in pairs], dtype=np.int32),
        config=config,
        duration_sec=duration,
        peak_count=len(pairs),
    )


def _audio(seed=0, seconds=8.0):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, seconds, int(SR * seconds), endpoint=False)
    wobble = 600.0 + 200.0 * np.sin(2 * np.pi * 0.5 * t + seed)
    return (
        0.50 * np.sin(2 * np.pi * (440.0 + 7 * seed) * t)
        + 0.30 * np.sin(2 * np.pi * wobble * t)
        + 0.20 * np.sin(2 * np.pi * 1500.0 * t)
        + 0.02 * rng.standard_normal(t.size)
    ).astype(np.float32)


def _decide(query, index, cfg=TEST_CFG):
    return decide(match(query, index), config=cfg)


# ------------------------------------------------------------------- accept --
class TestAccept:
    def test_strong_aligned_match_is_accepted(self):
        db = _fp([(1000 + i, i * 4) for i in range(100)])
        idx = build_index([("A", db)])
        d = _decide(db, idx)
        assert d.decision is Decision.MATCH and d.is_match
        assert d.track_id == "A"
        assert d.evidence_score == 1.0
        assert d.best_offset == 0

    def test_strong_spike_survives_background_noise(self):
        """40 aligned landmarks buried in 300 unaligned ones must still accept."""
        rng = np.random.default_rng(0)
        aligned = [(2000 + i, i * 4) for i in range(40)]
        noise_q = [(5000 + i, int(t)) for i, t in enumerate(rng.permutation(3000)[:300])]
        db_noise = [(5000 + i, int(t)) for i, t in enumerate(rng.permutation(3000)[:300])]
        idx = build_index([("A", _fp(aligned + db_noise))])
        d = _decide(_fp(aligned + noise_q), idx)
        assert d.is_match and d.track_id == "A"
        assert d.aligned_landmarks >= 40
        assert d.evidence_score >= 40 / 340

    def test_partial_but_coherent_evidence_is_accepted(self):
        """A degraded query keeps only a fraction of its landmarks; the rule
        must accept on a fraction, not demand near-perfection."""
        db = _fp([(3000 + i, i * 4) for i in range(200)])
        idx = build_index([("A", db)])
        surviving = _fp(
            [(3000 + i, i * 4) for i in range(0, 200, 8)]  # 25 of 200 landmarks
            + [(90000 + i, i * 7) for i in range(150)]  # plus unrelated ones
        )
        d = _decide(surviving, idx)
        assert d.is_match and d.evidence_score == pytest.approx(25 / 175, rel=1e-6)


# ------------------------------------------------------------------- reject --
class TestReject:
    def test_weak_evidence_is_rejected(self):
        db = _fp([(4000 + i, i * 4) for i in range(200)])
        idx = build_index([("A", db)])
        # 6 aligned landmarks out of 400 -> rate 0.015, below 0.05
        q = _fp(
            [(4000 + i, i * 4) for i in range(6)]
            + [(70000 + i, i * 5) for i in range(394)]
        )
        d = _decide(q, idx)
        assert d.decision is Decision.NO_MATCH and not d.is_match
        assert d.evidence_score < TEST_CFG.threshold

    def test_random_collisions_are_rejected(self):
        rng = np.random.default_rng(2)
        db = _fp([(int(h), int(t)) for h, t in
                  zip(rng.integers(0, 2**20, 600), rng.integers(0, 4000, 600))])
        idx = build_index([("A", db)])
        q = _fp([(int(h), int(t)) for h, t in
                 zip(rng.integers(0, 2**20, 600), rng.integers(0, 4000, 600))])
        assert not _decide(q, idx).is_match

    def test_many_hits_but_scattered_offsets_are_rejected(self):
        """The case a hit-counting recognizer accepts and this one must not:
        every query landmark matches, none of them agree on when."""
        rng = np.random.default_rng(3)
        hashes = [6000 + i for i in range(300)]
        db = _fp([(h, int(t)) for h, t in zip(hashes, rng.permutation(20000)[:300])])
        idx = build_index([("A", db)])
        q = _fp([(h, i * 4) for i, h in enumerate(hashes)])
        r = match(q, idx)
        assert r.candidates[0].total_hits == 300  # every landmark hit
        assert not decide(r, config=TEST_CFG).is_match  # none of them aligned

    def test_empty_candidates_are_rejected(self):
        idx = build_index([("A", _fp([(1, 0), (2, 4)]))])
        d = _decide(_fp([(1234, 0)]), idx)
        assert d.decision is Decision.NO_MATCH
        assert d.track_id is None and d.candidates == ()

    def test_unknown_hashes_are_rejected(self):
        idx = build_index([("A", _fp([(i, i * 3) for i in range(50)]))])
        d = _decide(_fp([(500000 + i, i * 3) for i in range(50)]), idx)
        assert d.decision is Decision.NO_MATCH
        assert d.aligned_landmarks == 0 and d.evidence_score == 0.0

    def test_empty_query_is_rejected(self):
        assert not _decide(_fp([]), build_index([("A", _fp([(1, 0)]))])).is_match

    def test_empty_index_is_rejected(self):
        assert not _decide(_fp([(1, 0)]), build_index([])).is_match

    def test_rejected_winner_is_withheld(self):
        """A caller must not be able to read a rejected hypothesis as an answer."""
        db = _fp([(7000 + i, i * 4) for i in range(200)])
        idx = build_index([("A", db)])
        q = _fp([(7000 + i, i * 4) for i in range(6)]
                + [(80000 + i, i * 5) for i in range(394)])
        d = _decide(q, idx)
        assert d.track_id is None
        assert d.candidates  # evidence still inspectable
        assert d.candidates[0].track_id == "A"

    def test_min_aligned_floor_blocks_degenerate_queries(self):
        """Two landmarks, both aligned, is a rate of 1.0 on no evidence."""
        idx = build_index([("A", _fp([(9, 100), (10, 104)]))])
        d = _decide(_fp([(9, 100), (10, 104)]), idx)
        assert d.evidence_score == 1.0  # rate says perfect
        assert d.decision is Decision.NO_MATCH  # floor says no
        assert d.aligned_landmarks < TEST_CFG.min_aligned_landmarks


# ------------------------------------------------------------ real negatives -
class TestNegativeAudio:
    def test_noise_audio_is_rejected(self):
        rng = np.random.default_rng(11)
        idx = build_index([(f"t{i}", fingerprint(_audio(i), SR)) for i in range(3)])
        for seed in range(4):
            noise = fingerprint(
                rng.standard_normal(SR * 6).astype(np.float32), SR
            )
            assert not decide(match(noise, idx), config=TEST_CFG).is_match

    def test_silence_is_rejected(self):
        idx = build_index([(f"t{i}", fingerprint(_audio(i), SR)) for i in range(2)])
        silent = fingerprint(np.zeros(SR * 5, dtype=np.float32), SR)
        assert not decide(match(silent, idx), config=TEST_CFG).is_match

    def test_out_of_catalog_audio_is_rejected(self):
        """Real structured audio that simply is not in the index."""
        idx = build_index([(f"t{i}", fingerprint(_audio(i), SR)) for i in range(3)])
        outsider = fingerprint(_audio(99), SR)
        assert not decide(match(outsider, idx), config=TEST_CFG).is_match

    def test_in_catalog_audio_is_accepted(self):
        """The complement: rejection must not be achieved by rejecting everything."""
        items = [(f"t{i}", fingerprint(_audio(i), SR)) for i in range(3)]
        idx = build_index(items)
        for tid, q in items:
            d = decide(match(q, idx), config=TEST_CFG)
            assert d.is_match and d.track_id == tid


# ----------------------------------------------------------- normalization ---
class TestQueryLengthNormalization:
    def test_score_is_a_rate_not_a_count(self):
        db = _fp([(8000 + i, i * 4) for i in range(400)])
        idx = build_index([("A", db)])
        short = _fp([(8000 + i, i * 4) for i in range(100)])
        long_ = _fp([(8000 + i, i * 4) for i in range(400)])
        ds, dl = _decide(short, idx), _decide(long_, idx)
        assert dl.aligned_landmarks == 4 * ds.aligned_landmarks  # counts differ 4x
        assert ds.evidence_score == dl.evidence_score == 1.0  # rate does not

    def test_same_fraction_gives_the_same_decision_at_any_length(self):
        """10% of landmarks aligned decides identically at 300 and 1200 landmarks."""
        db = _fp([(i, i * 4) for i in range(2000)])
        idx = build_index([("A", db)])
        outcomes = []
        for total in (300, 1200):
            aligned = total // 10
            q = _fp(
                [(i, i * 4) for i in range(aligned)]
                + [(300000 + i, i * 6) for i in range(total - aligned)]
            )
            d = _decide(q, idx)
            outcomes.append((d.decision, round(d.evidence_score, 6)))
        assert outcomes[0] == outcomes[1]
        assert outcomes[0][1] == pytest.approx(0.1)

    def test_an_unnormalized_count_would_have_disagreed(self):
        """Why normalization exists: the same 10% is 30 landmarks at one length
        and 120 at another, so any fixed count threshold splits them."""
        db = _fp([(i, i * 4) for i in range(2000)])
        idx = build_index([("A", db)])
        counts = []
        for total in (300, 1200):
            aligned = total // 10
            q = _fp([(i, i * 4) for i in range(aligned)]
                    + [(300000 + i, i * 6) for i in range(total - aligned)])
            counts.append(_decide(q, idx).aligned_landmarks)
        assert counts == [30, 120]
        assert any(c < 60 for c in counts) and any(c >= 60 for c in counts)

    def test_evidence_score_helper_matches_the_decision(self):
        db = _fp([(9000 + i, i * 4) for i in range(50)])
        idx = build_index([("A", db)])
        r = match(db, idx)
        assert evidence_score(r.candidates[0], r.query_landmark_count) == (
            decide(r, config=TEST_CFG).evidence_score
        )

    def test_empty_query_does_not_divide_by_zero(self):
        db = _fp([(1, 0), (2, 4)])
        idx = build_index([("A", db)])
        r = match(db, idx)
        assert evidence_score(r.candidates[0], 0) == 0.0


# -------------------------------------------------------------- thresholds ---
class TestThresholdBehaviour:
    def _setup(self):
        db = _fp([(11000 + i, i * 4) for i in range(400)])
        idx = build_index([("A", db)])
        q = _fp([(11000 + i, i * 4) for i in range(80)]
                + [(400000 + i, i * 6) for i in range(320)])  # rate = 0.2
        return idx, q

    def test_below_threshold_rejects_above_accepts(self):
        idx, q = self._setup()
        assert decide(match(q, idx), config=DecisionConfig(threshold=0.10)).is_match
        assert not decide(match(q, idx), config=DecisionConfig(threshold=0.50)).is_match

    def test_threshold_is_inclusive(self):
        idx, q = self._setup()
        score = decide(match(q, idx), config=DecisionConfig(threshold=0.0)).evidence_score
        assert decide(match(q, idx), config=DecisionConfig(threshold=score)).is_match

    def test_raising_the_threshold_never_adds_matches(self):
        idx, q = self._setup()
        accepted = [
            decide(match(q, idx), config=DecisionConfig(threshold=t)).is_match
            for t in (0.0, 0.05, 0.1, 0.2, 0.3, 0.9)
        ]
        assert accepted == sorted(accepted, reverse=True)  # monotone

    def test_threshold_is_reported_on_the_result(self):
        idx, q = self._setup()
        d = decide(match(q, idx), config=DecisionConfig(threshold=0.123))
        assert d.threshold == 0.123

    def test_invalid_config_is_rejected(self):
        for bad in (
            DecisionConfig(threshold=-0.1),
            DecisionConfig(threshold=1.5),
            DecisionConfig(min_aligned_landmarks=0),
        ):
            with pytest.raises(ValueError):
                bad.validate()


# ------------------------------------------------------------ determinism ----
class TestDeterminism:
    def test_repeated_decisions_are_identical(self):
        items = [(f"t{i}", fingerprint(_audio(i), SR)) for i in range(3)]
        idx = build_index(items)
        a = _decide(items[0][1], idx)
        b = _decide(items[0][1], idx)
        assert a.to_dict() == b.to_dict()

    def test_equal_evidence_decides_deterministically(self):
        """Two tracks with identical evidence: the verdict and the winner must
        be stable, and must not depend on catalog insertion order."""
        pairs = [(12000 + i, i * 4) for i in range(60)]
        forward = build_index([("zebra", _fp(pairs)), ("alpha", _fp(pairs))])
        reverse = build_index([("alpha", _fp(pairs)), ("zebra", _fp(pairs))])
        d1, d2 = _decide(_fp(pairs), forward), _decide(_fp(pairs), reverse)
        assert d1.decision is d2.decision is Decision.MATCH
        assert d1.track_id == d2.track_id == "alpha"
        assert d1.margin == d2.margin == 0  # a genuine tie, reported as one

    def test_identify_matches_the_two_step_path(self):
        items = [(f"t{i}", fingerprint(_audio(i), SR)) for i in range(2)]
        idx = build_index(items)
        one = identify(items[0][1], idx, decision_config=TEST_CFG)
        two = decide(match(items[0][1], idx), config=TEST_CFG)
        assert one.to_dict() == two.to_dict()


# ---------------------------------------------------------------- contract ---
class TestContract:
    def test_no_probability_surface_exists(self):
        db = _fp([(13000 + i, i * 4) for i in range(50)])
        d = _decide(db, build_index([("A", db)]))
        for banned in ("confidence", "probability", "certainty", "likelihood"):
            assert not hasattr(d, banned)

    def test_decision_serializes_as_plain_text(self):
        db = _fp([(14000 + i, i * 4) for i in range(50)])
        d = _decide(db, build_index([("A", db)]))
        assert d.to_dict()["decision"] == "MATCH"
        assert Decision.NO_MATCH.value == "NO_MATCH"

    def test_result_exposes_the_required_evidence(self):
        db = _fp([(15000 + i, i * 4) for i in range(60)])
        idx = build_index([("A", db), ("B", _fp([(15000 + i, i * 9) for i in range(60)]))])
        d = _decide(db, idx)
        assert isinstance(d, MatchDecision)
        for field in ("decision", "track_id", "evidence_score", "threshold",
                      "aligned_landmarks", "best_offset", "concentration",
                      "runner_up_track_id", "runner_up_score", "margin"):
            assert hasattr(d, field), field
        assert d.runner_up_track_id == "B"
        assert d.margin == d.aligned_landmarks - d.runner_up_score

    def test_default_config_is_documented_as_provisional(self):
        """The shipped default is a placeholder; the real operating point is a
        measurement recorded in the Phase 1D report."""
        assert 0.0 < DEFAULT_DECISION_CONFIG.threshold < 1.0
        assert DEFAULT_DECISION_CONFIG.min_aligned_landmarks >= 1
