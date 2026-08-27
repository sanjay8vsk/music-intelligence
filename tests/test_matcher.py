"""Focused tests for the offset-histogram matcher.

Most tests build fingerprints by hand rather than from audio. Temporal
consistency is a property of (hash, anchor_frame) arithmetic, and hand-built
landmarks let a test state exactly which offsets exist -- which is the only way
to prove the matcher selects a cluster for the right reason rather than by luck.
"""

from __future__ import annotations

import numpy as np
import pytest

from musicintel.recognition.fingerprint import (
    FingerprintConfig,
    FingerprintResult,
    fingerprint,
)
from musicintel.recognition.index import build_index
from musicintel.recognition.matcher import (
    MatchConfig,
    MatchResult,
    _best_cluster,
    _best_cluster_compiled,
    _count_distinct,
    match,
    warm_up,
)

SR = 11025
CFG = FingerprintConfig()


# --------------------------------------------------------------- fixtures ----
def _fp(pairs, *, duration: float = 30.0, config: FingerprintConfig = CFG):
    """Build a FingerprintResult from explicit (hash, anchor_frame) pairs."""
    pairs = sorted((int(h), int(a)) for h, a in pairs)
    return FingerprintResult(
        hashes=np.array([h for h, _ in pairs], dtype=np.uint32),
        anchor_frames=np.array([a for _, a in pairs], dtype=np.int32),
        config=config,
        duration_sec=duration,
        peak_count=len(pairs),
    )


def _shift(result: FingerprintResult, frames: int) -> FingerprintResult:
    """Same hashes, every anchor moved by `frames`."""
    return FingerprintResult(
        hashes=result.hashes,
        anchor_frames=(result.anchor_frames + frames).astype(np.int32),
        config=result.config,
        duration_sec=result.duration_sec,
        peak_count=result.peak_count,
    )


def _audio(seed: int = 0, seconds: float = 8.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.linspace(0, seconds, int(SR * seconds), endpoint=False)
    wobble = 600.0 + 200.0 * np.sin(2 * np.pi * 0.5 * t + seed)
    return (
        0.50 * np.sin(2 * np.pi * (440.0 + 7 * seed) * t)
        + 0.30 * np.sin(2 * np.pi * wobble * t)
        + 0.20 * np.sin(2 * np.pi * 1500.0 * t)
        + 0.02 * rng.standard_normal(t.size)
    ).astype(np.float32)


# ------------------------------------------------------------ exact match ----
class TestPerfectMatch:
    def test_identical_query_recovers_track_and_zero_offset(self):
        a = _fp([(1000 + i, i * 4) for i in range(40)])
        b = _fp([(9000 + i, i * 4) for i in range(40)])
        idx = build_index([("A", a), ("B", b)])
        r = match(a, idx)
        assert r.top_id == "A"
        assert r.best.best_offset == 0
        assert r.best.score == 40  # every query landmark aligned
        assert r.best.concentration == 1.0

    def test_shifted_database_recovers_the_known_offset(self):
        """The reference sits 137 frames later than the query it came from."""
        base = _fp([(2000 + i, i * 5) for i in range(50)])
        idx = build_index([("A", _shift(base, 137))])
        r = match(base, idx)
        assert r.top_id == "A"
        assert r.best.best_offset == 137
        assert r.best.score == 50

    def test_negative_offset_is_representable(self):
        base = _fp([(3000 + i, 500 + i * 5) for i in range(30)])
        idx = build_index([("A", _shift(base, -200))])
        r = match(base, idx)
        assert r.best.best_offset == -200
        assert r.best.score == 30

    def test_offset_seconds_follows_the_frame_geometry(self):
        base = _fp([(4000 + i, i * 3) for i in range(20)])
        idx = build_index([("A", _shift(base, 86))])
        r = match(base, idx)
        expected = 86 * CFG.hop_length / CFG.sample_rate
        assert r.best.best_offset == 86
        assert r.best.best_offset_seconds == pytest.approx(expected)
        assert r.best.best_offset_seconds == pytest.approx(0.998, abs=0.01)

    def test_excerpt_of_a_track_finds_its_position(self):
        full = _fp([(5000 + i, i * 4) for i in range(200)])
        idx = build_index([("A", full)])
        # landmarks 50..99 -> anchors 200..396; as a query they start at 0
        piece = _fp([(5000 + i, (i - 50) * 4) for i in range(50, 100)])
        r = match(piece, idx)
        assert r.top_id == "A"
        assert r.best.best_offset == 200
        assert r.best.score == 50


# ------------------------------------------------------ discrimination ------
class TestDiscrimination:
    def test_query_from_a_outranks_unrelated_b(self):
        a = _fp([(1000 + i, i * 4) for i in range(60)])
        b = _fp([(7000 + i, i * 4) for i in range(60)])
        idx = build_index([("A", a), ("B", b)])
        r = match(a, idx)
        assert r.top_id == "A"
        assert [c.track_id for c in r.candidates] == ["A"]  # B shares nothing

    def test_shared_hashes_are_resolved_by_temporal_consistency(self):
        """Both tracks hold every query hash; only one holds them in order.

        This is the case a hash-count matcher gets wrong: B has exactly as many
        matching hashes as A, so counting hits cannot separate them. Only the
        offsets can.
        """
        hashes = [6000 + i for i in range(40)]
        a = _fp([(h, i * 6) for i, h in enumerate(hashes)])  # in order
        rng = np.random.default_rng(0)
        scattered = rng.permutation(4000)[:40]
        b = _fp([(h, int(t)) for h, t in zip(hashes, scattered)])  # same hashes, shuffled
        idx = build_index([("A", a), ("B", b)])
        r = match(a, idx)
        by_id = {c.track_id: c for c in r.candidates}
        assert by_id["A"].total_hits == by_id["B"].total_hits == 40  # tied on hits
        assert r.top_id == "A"
        assert by_id["A"].score == 40
        assert by_id["B"].score < 5

    def test_many_scattered_hits_lose_to_fewer_aligned_hits(self):
        """Requirement: coherence beats volume."""
        aligned = _fp([(100 + i, i * 3) for i in range(12)])
        rng = np.random.default_rng(1)
        noisy_pairs = [
            (100 + i, int(t)) for i, t in enumerate(rng.permutation(5000)[:12])
        ]
        # give the noisy track four times the hits, all incoherent
        for rep in range(1, 4):
            noisy_pairs += [
                (100 + i, int(t))
                for i, t in enumerate(rng.permutation(5000)[:12])
            ]
        idx = build_index([("ALIGNED", aligned), ("NOISY", _fp(noisy_pairs))])
        r = match(aligned, idx)
        by_id = {c.track_id: c for c in r.candidates}
        assert by_id["NOISY"].total_hits > by_id["ALIGNED"].total_hits
        assert r.top_id == "ALIGNED"
        assert by_id["ALIGNED"].concentration > by_id["NOISY"].concentration

    def test_real_audio_self_match_outranks_other_tracks(self):
        items = [(f"t{i}", fingerprint(_audio(i), SR)) for i in range(3)]
        idx = build_index(items)
        for tid, q in items:
            r = match(q, idx)
            assert r.top_id == tid
            assert r.best.best_offset == 0
            assert r.best.score == len(q)


# ----------------------------------------------------------- multi-posting ---
class TestMultiplePostings:
    def test_every_posting_is_considered_not_just_the_first(self):
        """One hash, three positions in one track: all three must be seen."""
        db = _fp([(42, 10), (42, 900), (42, 2000)])
        idx = build_index([("A", db)])
        r = match(_fp([(42, 0)]), idx)
        assert r.total_hits == 3
        assert r.candidates[0].total_hits == 3

    def test_all_postings_across_tracks_are_considered(self):
        idx = build_index([("A", _fp([(42, 5)])), ("B", _fp([(42, 77)]))])
        r = match(_fp([(42, 0)]), idx)
        assert {c.track_id for c in r.candidates} == {"A", "B"}
        assert r.total_hits == 2

    def test_repeated_hash_cannot_inflate_the_score(self):
        """A single query landmark is one unit of evidence however many
        postings its hash has. `best_offset_count` may exceed `score`; the
        ranking uses `score`."""
        db = _fp([(42, 100), (42, 101), (42, 102)])  # offsets 0,1,2 -> one window
        idx = build_index([("A", db)])
        r = match(_fp([(42, 100)]), idx)
        c = r.candidates[0]
        assert c.best_offset_count == 3  # three raw hits
        assert c.score == 1  # but only one distinct query landmark
        assert c.matched_query_landmarks == 1

    def test_two_query_landmarks_score_two(self):
        db = _fp([(42, 100), (43, 104)])
        idx = build_index([("A", db)])
        r = match(_fp([(42, 100), (43, 104)]), idx)
        assert r.candidates[0].score == 2


# --------------------------------------------------------------- clusters ----
class TestOffsetClusters:
    def test_strongest_of_several_peaks_wins(self):
        """Three coherent clusters at different offsets; the tallest is chosen."""
        pairs = []
        pairs += [(200 + i, i * 4 + 50) for i in range(5)]  # offset 50, 5 votes
        pairs += [(300 + i, i * 4 + 800) for i in range(20)]  # offset 800, 20 votes
        pairs += [(400 + i, i * 4 + 1500) for i in range(9)]  # offset 1500, 9 votes
        idx = build_index([("A", _fp(pairs))])
        query = _fp(
            [(200 + i, i * 4) for i in range(5)]
            + [(300 + i, i * 4) for i in range(20)]
            + [(400 + i, i * 4) for i in range(9)]
        )
        r = match(query, idx, compute_second_best=True)
        c = r.candidates[0]
        assert c.best_offset == 800
        assert c.score == 20
        assert c.second_best_score == 9  # the runner-up cluster, not 19
        assert c.second_best_offset == 1500

    def test_second_best_excludes_the_winning_window(self):
        pairs = [(500 + i, i * 4 + 300) for i in range(10)]
        idx = build_index([("A", _fp(pairs))])
        r = match(
            _fp([(500 + i, i * 4) for i in range(10)]), idx, compute_second_best=True
        )
        c = r.candidates[0]
        assert c.best_offset == 300
        assert c.second_best_score == 0  # only one cluster exists
        assert c.second_best_offset is None
        assert c.margin == c.score

    def test_tolerance_absorbs_small_frame_drift(self):
        """Offsets 100..102 are one alignment at the default tolerance of 2."""
        db = _fp([(600 + i, i * 10 + 100 + (i % 3)) for i in range(12)])
        idx = build_index([("A", db)])
        r = match(_fp([(600 + i, i * 10) for i in range(12)]), idx)
        assert r.candidates[0].score == 12

    def test_zero_tolerance_splits_that_drift(self):
        db = _fp([(600 + i, i * 10 + 100 + (i % 3)) for i in range(12)])
        idx = build_index([("A", db)])
        r = match(
            _fp([(600 + i, i * 10) for i in range(12)]),
            idx,
            config=MatchConfig(offset_tolerance_frames=0),
        )
        assert r.candidates[0].score == 4  # each exact offset gets a third

    def test_wider_tolerance_merges_more(self):
        db = _fp([(700 + i, i * 20 + 50 + 4 * (i % 3)) for i in range(9)])
        idx = build_index([("A", db)])
        q = _fp([(700 + i, i * 20) for i in range(9)])
        narrow = match(q, idx, config=MatchConfig(offset_tolerance_frames=1))
        wide = match(q, idx, config=MatchConfig(offset_tolerance_frames=8))
        assert wide.candidates[0].score > narrow.candidates[0].score
        assert wide.candidates[0].score == 9


# ------------------------------------------------------------- empty cases ---
class TestNoMatches:
    def test_unknown_hashes_give_no_candidates(self):
        idx = build_index([("A", _fp([(1, 0), (2, 4)]))])
        r = match(_fp([(9999, 0), (8888, 4)]), idx)
        assert r.candidates == () and r.top_id is None and r.best is None
        assert r.total_hits == 0 and r.matched_query_landmarks == 0

    def test_empty_query_gives_no_candidates(self):
        idx = build_index([("A", _fp([(1, 0)]))])
        assert match(_fp([]), idx).candidates == ()

    def test_empty_index_gives_no_candidates(self):
        assert match(_fp([(1, 0)]), build_index([])).candidates == ()

    def test_empty_result_is_not_a_rejection(self):
        """1C makes no accept/reject decision; NO_MATCH is Phase 1D."""
        idx = build_index([("A", _fp([(1, 0)]))])
        r = match(_fp([(2, 0)]), idx)
        assert isinstance(r, MatchResult)
        assert not hasattr(r, "is_match")  # no decision surface exists yet

    def test_noise_query_still_returns_a_ranked_list(self):
        """Deliberate: weak candidates are returned, not suppressed."""
        rng = np.random.default_rng(5)
        items = [(f"t{i}", fingerprint(_audio(i), SR)) for i in range(2)]
        idx = build_index(items)
        noise = fingerprint(rng.standard_normal(SR * 6).astype(np.float32), SR)
        r = match(noise, idx)
        if r.candidates:  # collisions are likely but not guaranteed
            assert r.best.concentration < 0.5  # dispersed, as expected


# ------------------------------------------------------------ determinism ----
class TestDeterminism:
    def test_repeated_match_is_identical(self):
        items = [(f"t{i}", fingerprint(_audio(i), SR)) for i in range(3)]
        idx = build_index(items)
        a, b = match(items[0][1], idx), match(items[0][1], idx)
        assert [(c.track_id, c.score, c.best_offset) for c in a.candidates] == [
            (c.track_id, c.score, c.best_offset) for c in b.candidates
        ]

    def test_ties_break_deterministically_by_track_id(self):
        """Two tracks with byte-identical evidence must still rank stably."""
        pairs = [(800 + i, i * 4) for i in range(15)]
        idx = build_index([("zebra", _fp(pairs)), ("alpha", _fp(pairs))])
        r = match(_fp(pairs), idx)
        assert [c.track_id for c in r.candidates] == ["alpha", "zebra"]
        assert r.candidates[0].score == r.candidates[1].score
        # Insertion order must not leak into the ranking.
        flipped = build_index([("alpha", _fp(pairs)), ("zebra", _fp(pairs))])
        assert [c.track_id for c in match(_fp(pairs), flipped).candidates] == [
            "alpha",
            "zebra",
        ]

    def test_ranking_is_by_score_descending(self):
        scores = [c.score for c in match(
            _fp([(900 + i, i * 4) for i in range(30)]),
            build_index([
                ("A", _fp([(900 + i, i * 4) for i in range(30)])),
                ("B", _fp([(900 + i, i * 4 + 7 * i) for i in range(30)])),
            ]),
        ).candidates]
        assert scores == sorted(scores, reverse=True)

    def test_top_k_accessors(self):
        items = [(f"t{i}", fingerprint(_audio(i), SR)) for i in range(3)]
        idx = build_index(items)
        r = match(items[0][1], idx)
        assert r.top_ids(1) == ["t0"]
        assert len(r.top(3)) <= 3
        assert r.top_ids(3)[0] == "t0"

    def test_max_candidates_truncates_without_reordering(self):
        items = [(f"t{i}", fingerprint(_audio(i), SR)) for i in range(3)]
        idx = build_index(items)
        full = match(items[0][1], idx)
        one = match(items[0][1], idx, config=MatchConfig(max_candidates=1))
        assert len(one.candidates) == 1
        assert one.candidates[0].track_id == full.candidates[0].track_id


# ---------------------------------------------------------------- contract ---
class TestContract:
    def test_config_mismatch_is_rejected(self):
        idx = build_index([("A", _fp([(1, 0)]))])
        other = _fp([(1, 0)], config=FingerprintConfig(fan_out=2))
        with pytest.raises(ValueError, match="different config"):
            match(other, idx)

    def test_invalid_match_config_is_rejected(self):
        with pytest.raises(ValueError):
            MatchConfig(offset_tolerance_frames=-1).validate()
        with pytest.raises(ValueError):
            MatchConfig(max_candidates=0).validate()

    def test_result_reports_query_shape_and_timing(self):
        q = fingerprint(_audio(0), SR)
        r = match(q, build_index([("A", q)]))
        assert r.query_landmark_count == len(q)
        assert r.query_duration_sec == pytest.approx(8.0, abs=0.05)
        assert r.timing.lookup >= 0 and r.timing.histogram >= 0
        assert r.timing.total == pytest.approx(
            r.timing.lookup + r.timing.histogram + r.timing.ranking
        )

    def test_score_is_never_presented_as_a_probability(self):
        q = fingerprint(_audio(0), SR)
        c = match(q, build_index([("A", q)])).candidates[0]
        assert isinstance(c.score, int)
        assert c.score > 1.0  # a count, not a [0,1] quantity
        assert not hasattr(c, "confidence") and not hasattr(c, "probability")


# ------------------------------------------------------- compiled kernel ----
class TestCompiledBestCluster:
    """`_best_cluster` is the reference; `_best_cluster_compiled` is what runs.

    These tests exist so the pair cannot drift. If someone edits one and not the
    other, the randomised comparison below fails rather than the difference
    reaching a benchmark months later.
    """

    @pytest.mark.parametrize("tolerance", [0, 1, 2, 5])
    def test_compiled_matches_reference_on_randomised_inputs(self, tolerance):
        rng = np.random.default_rng(20260827)
        for _ in range(300):
            size = int(rng.integers(0, 60))
            n_query = int(rng.integers(1, 40))
            offsets = np.sort(rng.integers(-50, 50, size)).astype(np.int64)
            query_idx = rng.integers(0, n_query, size).astype(np.int64)
            assert tuple(_best_cluster(offsets, query_idx, tolerance)) == tuple(
                _best_cluster_compiled(offsets, query_idx, tolerance, n_query)
            )

    def test_compiled_breaks_offset_ties_toward_the_smaller_value(self):
        """Two offsets equally common in the window; the smaller is reported."""
        offsets = np.array([10, 10, 20, 20], dtype=np.int64)
        query_idx = np.array([0, 1, 2, 3], dtype=np.int64)
        assert _best_cluster_compiled(offsets, query_idx, 10, 4)[2] == 10
        assert _best_cluster(offsets, query_idx, 10)[2] == 10

    def test_compiled_keeps_the_earliest_window_on_a_tie(self):
        """Two windows with identical (distinct, hits); the earlier one wins."""
        offsets = np.array([0, 1, 100, 101], dtype=np.int64)
        query_idx = np.array([0, 1, 2, 3], dtype=np.int64)
        ref = _best_cluster(offsets, query_idx, 2)
        assert tuple(_best_cluster_compiled(offsets, query_idx, 2, 4)) == tuple(ref)
        assert ref[3] == 0 and ref[4] == 1  # the earlier window, not 100..101

    def test_count_distinct_matches_numpy(self):
        rng = np.random.default_rng(7)
        for _ in range(200):
            v = rng.integers(0, 30, int(rng.integers(0, 80))).astype(np.int64)
            assert _count_distinct(v) == int(np.unique(v).size)

    def test_empty_input_is_handled(self):
        empty = np.empty(0, dtype=np.int64)
        assert tuple(_best_cluster_compiled(empty, empty, 2, 1)) == (0, 0, 0, 0, 0)
        assert _count_distinct(empty) == 0


class TestSecondBestIsLazy:
    """The runner-up pass is opt-in; opting in reproduces the eager values."""

    @staticmethod
    def _two_clusters():
        pairs = [(300 + i, i * 4 + 800) for i in range(20)]
        pairs += [(400 + i, i * 4 + 1500) for i in range(9)]
        idx = build_index([("A", _fp(pairs))])
        query = _fp(
            [(300 + i, i * 4) for i in range(20)]
            + [(400 + i, i * 4) for i in range(9)]
        )
        return query, idx

    def test_default_does_not_compute_the_runner_up(self):
        query, idx = self._two_clusters()
        c = match(query, idx).candidates[0]
        assert c.second_best_score == 0
        assert c.second_best_offset is None
        assert c.margin == c.score  # margin degenerates to score, by construction

    def test_opting_in_recovers_the_runner_up(self):
        query, idx = self._two_clusters()
        c = match(query, idx, compute_second_best=True).candidates[0]
        assert c.second_best_score == 9
        assert c.second_best_offset == 1500
        assert c.margin == c.score - 9

    def test_laziness_changes_nothing_else(self):
        """Every field the recognition path reads is identical either way."""
        query, idx = self._two_clusters()
        lazy = match(query, idx)
        eager = match(query, idx, compute_second_best=True)
        assert len(lazy.candidates) == len(eager.candidates)
        for a, b in zip(lazy.candidates, eager.candidates):
            for field in (
                "track_id",
                "score",
                "best_offset",
                "best_offset_seconds",
                "best_offset_count",
                "total_hits",
                "matched_query_landmarks",
                "concentration",
            ):
                assert getattr(a, field) == getattr(b, field), field
        assert lazy.total_hits == eager.total_hits
        assert lazy.matched_query_landmarks == eager.matched_query_landmarks


class TestWarmUp:
    def test_warm_up_is_idempotent_and_reports_seconds(self):
        first = warm_up()
        second = warm_up()
        assert first >= 0.0 and second >= 0.0
        # Already compiled by the time any test runs, so the second call is cheap.
        assert second < 0.5
