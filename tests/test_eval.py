"""Focused tests for the recognition evaluation harness.

These deliberately do NOT require the fixture audio corpus. Where audio is
needed a few seconds of it are synthesized in a tmp dir, so the suite runs
anywhere, including CI without network access.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import soundfile as sf

from musicintel.eval import degradation as dg
from musicintel.eval.manifest import Manifest, Track, sha256_bytes
from musicintel.eval.metrics import (
    QueryOutcome,
    latency_stats,
    percentile,
    summarize,
    threshold_sweep,
)


# --------------------------------------------------------------- fixtures ----
def _track(tid: str, dur: float = 60.0, held_out: bool = False) -> Track:
    return Track(
        track_id=tid,
        path=f"data/eval/corpus/{tid}.mp3",
        sha256=sha256_bytes(tid.encode()),
        source="test",
        license="CC-BY",
        license_url="https://creativecommons.org/licenses/by/4.0/",
        duration_sec=dur,
        held_out=held_out,
    )


@pytest.fixture
def tone_wav(tmp_path):
    """20 s of deterministic audio, written to disk."""
    sr = 22050
    t = np.linspace(0, 20.0, sr * 20, endpoint=False)
    y = (0.4 * np.sin(2 * np.pi * 440 * t) + 0.2 * np.sin(2 * np.pi * 1109 * t)).astype(
        np.float32
    )
    p = tmp_path / "tone.wav"
    sf.write(p, y, sr, subtype="PCM_16")
    return p


def _outcome(qid, truth, returned, *, neg=False, ms=10.0, dist=None, err=None):
    return QueryOutcome(
        query_id=qid,
        condition="clean",
        family="clean",
        duration=5.0,
        position="middle",
        is_negative=neg,
        latency_ms=ms,
        returned_ids=list(returned),
        truth_track_id=truth,
        top_distance=dist,
        error=err,
    )


# --------------------------------------------------------------- manifest ----
class TestManifest:
    def test_save_load_roundtrip(self, tmp_path):
        m = Manifest(tracks=[_track("a"), _track("b"), _track("c")])
        p = tmp_path / "manifest.json"
        m.save(p)
        loaded = Manifest.load(p)
        assert len(loaded) == 3
        assert [t.track_id for t in loaded] == ["a", "b", "c"]
        assert loaded.by_id("b").license == "CC-BY"
        assert loaded.by_id("zzz") is None

    def test_content_hash_is_order_independent(self):
        a = Manifest(tracks=[_track("x"), _track("y")])
        b = Manifest(tracks=[_track("y"), _track("x")])
        assert a.content_hash() == b.content_hash()

    def test_content_hash_changes_with_audio(self):
        a = Manifest(tracks=[_track("x")])
        b = Manifest(tracks=[_track("x")])
        b.tracks[0].sha256 = "different"
        assert a.content_hash() != b.content_hash()

    def test_assign_holdout_is_deterministic_and_exclusive(self):
        m = Manifest(tracks=[_track(f"t{i:02d}") for i in range(10)])
        m.assign_holdout(3)
        first = {t.track_id for t in m.held_out}
        assert len(first) == 3
        assert len(m.catalog) == 7
        assert first.isdisjoint({t.track_id for t in m.catalog})
        m.assign_holdout(3)
        assert {t.track_id for t in m.held_out} == first

    def test_verify_reports_missing_audio(self, tmp_path):
        m = Manifest(tracks=[_track("missing")])
        problems = m.verify(tmp_path)
        assert len(problems) == 1 and "missing audio" in problems[0]

    def test_verify_detects_duplicate_ids(self, tmp_path):
        m = Manifest(tracks=[_track("dup"), _track("dup")])
        assert any("duplicate track_id" in p for p in m.verify(tmp_path))

    def test_license_counts(self):
        t = _track("z")
        t.license = "CC0-1.0"
        m = Manifest(tracks=[_track("a"), _track("b"), t])
        assert m.license_counts() == {"CC-BY": 2, "CC0-1.0": 1}


# ------------------------------------------------------ query generation ----
class TestQueryGeneration:
    def test_seed_is_deterministic_and_id_dependent(self):
        assert dg.derive_seed("abc") == dg.derive_seed("abc")
        assert dg.derive_seed("abc") != dg.derive_seed("abd")
        assert 0 <= dg.derive_seed("abc") < 2**32

    def test_excerpt_offsets(self):
        assert dg.excerpt_offset(100.0, 10.0, "beginning") == 0.0
        assert dg.excerpt_offset(100.0, 10.0, "middle") == 45.0
        assert dg.excerpt_offset(100.0, 10.0, "end") == 90.0

    def test_excerpt_never_runs_past_source(self):
        for pos in dg.POSITIONS:
            off = dg.excerpt_offset(12.0, 10.0, pos)
            assert off >= 0.0
            assert off + 10.0 <= 12.0 + 1e-9

    def test_excerpt_offset_clamps_when_too_short(self):
        assert dg.excerpt_offset(5.0, 10.0, "end") == 0.0

    def test_unknown_position_raises(self):
        with pytest.raises(ValueError):
            dg.excerpt_offset(100.0, 10.0, "sideways")

    def test_plan_is_reproducible(self):
        from musicintel.eval.recognition import plan_positive_queries

        tracks = [_track("t1"), _track("t2")]
        a = plan_positive_queries(tracks)
        b = plan_positive_queries(tracks)
        assert [s.query_id for s in a] == [s.query_id for s in b]
        assert [s.seed for s in a] == [s.seed for s in b]
        assert len(a) > 0

    def test_plan_skips_durations_longer_than_track(self):
        from musicintel.eval.recognition import plan_positive_queries

        short = _track("short", dur=4.0)
        specs = plan_positive_queries([short])
        assert specs, "a 4s track should still yield 3s queries"
        assert all(s.duration <= 4.0 for s in specs)
        assert all(s.duration != 10.0 for s in specs)

    def test_query_ids_are_unique(self):
        from musicintel.eval.recognition import plan_positive_queries

        specs = plan_positive_queries([_track("t1"), _track("t2")])
        ids = [s.query_id for s in specs]
        assert len(ids) == len(set(ids))

    def test_load_excerpt_rejects_oversized_request(self, tone_wav):
        with pytest.raises(ValueError):
            dg.load_excerpt(tone_wav, duration=60.0, position="middle")

    def test_load_excerpt_returns_requested_length(self, tone_wav):
        y, sr = dg.load_excerpt(tone_wav, duration=5.0, position="middle")
        assert sr == dg.QUERY_SAMPLE_RATE
        assert abs(len(y) / sr - 5.0) < 0.05


# ---------------------------------------------------------- degradations ----
class TestDegradations:
    def test_condition_matrix_covers_every_axis(self):
        fams = {f for _, f, _ in dg.condition_matrix()}
        assert fams == {"clean", "noise", "codec", "filter", "speed", "pitch"}
        labels = [c for c, _, _ in dg.condition_matrix()]
        assert len(labels) == len(set(labels)), "condition labels must be unique"

    def test_noise_is_deterministic_for_a_seed(self):
        y = np.sin(np.linspace(0, 50, 22050)).astype(np.float32)
        a, _ = dg.add_noise(y, 10, "pink", np.random.default_rng(7))
        b, _ = dg.add_noise(y, 10, "pink", np.random.default_rng(7))
        assert np.array_equal(a, b)

    def test_noise_differs_across_seeds(self):
        y = np.sin(np.linspace(0, 50, 22050)).astype(np.float32)
        a, _ = dg.add_noise(y, 10, "pink", np.random.default_rng(1))
        b, _ = dg.add_noise(y, 10, "pink", np.random.default_rng(2))
        assert not np.array_equal(a, b)

    def test_lower_snr_means_more_noise(self):
        y = np.sin(np.linspace(0, 500, 22050)).astype(np.float32)
        quiet, _ = dg.add_noise(y, 20, "white", np.random.default_rng(0))
        loud, _ = dg.add_noise(y, 0, "white", np.random.default_rng(0))
        assert np.mean((loud - y) ** 2) > np.mean((quiet - y) ** 2)

    def test_achieved_snr_matches_target(self):
        y = np.sin(np.linspace(0, 500, 22050)).astype(np.float32)
        for target in (20, 10, 5, 0):
            _, info = dg.add_noise(y, target, "white", np.random.default_rng(3))
            assert abs(info["achieved_snr_db"] - target) < 0.5

    def test_reference_audio_is_not_mutated(self):
        y = np.sin(np.linspace(0, 50, 22050)).astype(np.float32)
        original = y.copy()
        dg.add_noise(y, 5, "white", np.random.default_rng(0))
        dg.apply_filter(y, 22050, "telephone")
        dg.change_speed(y, 22050, 5.0)
        assert np.array_equal(y, original)

    def test_speed_changes_length_in_right_direction(self):
        y = np.sin(np.linspace(0, 500, 22050 * 2)).astype(np.float32)
        faster, info = dg.change_speed(y, 22050, 5.0)
        slower, _ = dg.change_speed(y, 22050, -5.0)
        assert len(faster) < len(y) < len(slower)
        assert info["factor"] == pytest.approx(1.05)

    def test_telephone_filter_attenuates_low_frequencies(self):
        sr = 22050
        t = np.linspace(0, 1.0, sr, endpoint=False)
        low = np.sin(2 * np.pi * 100 * t).astype(np.float32)  # below the 300 Hz edge
        out, info = dg.apply_filter(low, sr, "telephone")
        assert info["band_hz"][0] == 300.0
        assert np.max(np.abs(out[sr // 2 :])) < 0.5 * np.max(np.abs(low))

    @pytest.mark.parametrize("codec,kbps", [("mp3", 128), ("mp3", 32), ("opus", 64)])
    def test_codec_roundtrip_preserves_duration(self, codec, kbps):
        sr = 22050
        t = np.linspace(0, 3.0, sr * 3, endpoint=False)
        y = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        out, out_sr, info = dg.apply_codec(y, sr, codec, kbps)
        assert abs(len(out) / out_sr - 3.0) < 0.25
        assert info["measured_kbps"] > 0
        assert info["codec"] == codec


# ---------------------------------------------------------------- metrics ----
class TestMetrics:
    def test_percentile_and_empty(self):
        assert percentile([], 50) is None
        assert percentile([10.0], 95) == 10.0
        assert percentile([1.0, 2.0, 3.0, 4.0], 50) == pytest.approx(2.5)

    def test_latency_aggregation(self):
        outs = [_outcome(f"q{i}", "t", ["t"], ms=float(i)) for i in range(1, 101)]
        stats = latency_stats(outs)
        assert stats["n"] == 100
        assert stats["mean_ms"] == pytest.approx(50.5)
        assert stats["p50_ms"] == pytest.approx(50.5)
        assert stats["p95_ms"] == pytest.approx(95.05, abs=0.5)
        assert stats["min_ms"] == 1.0 and stats["max_ms"] == 100.0

    def test_latency_excludes_errored_queries(self):
        outs = [
            _outcome("a", "t", ["t"], ms=10.0),
            _outcome("b", "t", [], ms=999.0, err="Boom"),
        ]
        assert latency_stats(outs)["n"] == 1
        assert latency_stats(outs)["mean_ms"] == 10.0

    def test_recall_at_1_and_3(self):
        outs = [
            _outcome("a", "t1", ["t1", "t2", "t3"]),  # correct@1
            _outcome("b", "t2", ["t9", "t2", "t3"]),  # correct@3 only
            _outcome("c", "t3", ["t7", "t8", "t9"]),  # wrong
            _outcome("d", "t4", []),                   # no match
        ]
        s = summarize(outs)
        assert s["recall_at_1"] == 0.25
        assert s["recall_at_3"] == 0.5
        assert s["no_match_rate"] == 0.25
        assert s["far"] is None  # no negatives present

    def test_recall_at_3_ignores_rank_four_and_beyond(self):
        s = summarize([_outcome("a", "t4", ["t1", "t2", "t3", "t4"])])
        assert s["recall_at_1"] == 0.0
        assert s["recall_at_3"] == 0.0

    def test_far_and_correct_rejection(self):
        outs = [
            _outcome("n1", None, ["t1"], neg=True),  # false accept
            _outcome("n2", None, ["t2"], neg=True),  # false accept
            _outcome("n3", None, [], neg=True),      # correct rejection
            _outcome("n4", None, [], neg=True),      # correct rejection
        ]
        s = summarize(outs)
        assert s["far"] == 0.5
        assert s["correct_rejection_rate"] == 0.5
        assert s["false_accepts"] == 2
        assert s["recall_at_1"] is None  # no positives present

    def test_far_is_one_when_recognizer_always_answers(self):
        outs = [_outcome(f"n{i}", None, ["t1"], neg=True) for i in range(20)]
        assert summarize(outs)["far"] == 1.0
        assert summarize(outs)["correct_rejection_rate"] == 0.0

    def test_summarize_empty(self):
        assert summarize([])["queries"] == 0

    def test_mixed_group_keeps_positive_and_negative_separate(self):
        outs = [
            _outcome("p1", "t1", ["t1"]),
            _outcome("n1", None, ["t1"], neg=True),
        ]
        s = summarize(outs)
        assert s["recall_at_1"] == 1.0  # over the 1 positive only
        assert s["far"] == 1.0          # over the 1 negative only
        assert s["queries"] == 2

    def test_threshold_sweep_separates_when_separable(self):
        pos = [_outcome(f"p{i}", "t1", ["t1"], dist=1.0) for i in range(10)]
        neg = [_outcome(f"n{i}", None, ["t1"], neg=True, dist=50.0) for i in range(10)]
        sweep = threshold_sweep(pos + neg)
        assert sweep["available"]
        best = sweep["operating_points"]["far_le_0.01"]
        assert best is not None and best["recall_at_1"] == 1.0

    def test_threshold_sweep_fails_when_inseparable(self):
        pos = [_outcome(f"p{i}", "t1", ["t1"], dist=10.0) for i in range(10)]
        neg = [_outcome(f"n{i}", None, ["t1"], neg=True, dist=10.0) for i in range(10)]
        best = threshold_sweep(pos + neg)["operating_points"]["far_le_0.01"]
        assert best is None or best["recall_at_1"] == 0.0

    def test_threshold_sweep_unavailable_without_negatives(self):
        pos = [_outcome("p1", "t1", ["t1"], dist=1.0)]
        assert threshold_sweep(pos)["available"] is False


# ------------------------------------------------------------- interface ----
class TestRecognizerInterface:
    def test_result_semantics(self):
        from musicintel.eval.recognizer import Candidate, RecognitionResult

        empty = RecognitionResult([])
        assert not empty.is_match and empty.top_id is None

        r = RecognitionResult([Candidate("a", 1.0, 0.5), Candidate("b", 0.5, 2.0)])
        assert r.is_match and r.top_id == "a"
        assert r.top_k_ids(1) == ["a"]
        assert r.top_k_ids(5) == ["a", "b"]

    def test_a_fake_recognizer_satisfies_the_protocol(self):
        """The harness must accept any engine, not just the current prototype."""
        from musicintel.eval.recognizer import Candidate, RecognitionResult

        class FakeRecognizer:
            name = "fake"
            version = "1"

            def prepare(self, tracks):
                self.ids = [t.track_id for t in tracks]

            def recognize(self, audio_path):
                return RecognitionResult([Candidate(self.ids[0], 1.0, 0.0)])

        from musicintel.eval.recognition import run_queries

        rec = FakeRecognizer()
        rec.prepare([_track("t1")])
        spec = dg.QuerySpec(
            query_id="q1", track_id="t1", duration=5.0, position="middle",
            condition="clean", family="clean", seed=1, source_hash="x",
        )
        outcomes = run_queries(rec, [(spec, "unused.wav")], verbose=False)
        assert len(outcomes) == 1
        assert outcomes[0].correct_at_1
        assert outcomes[0].latency_ms >= 0.0


def test_query_spec_is_json_serializable():
    spec = dg.QuerySpec(
        query_id="q", track_id="t", duration=5.0, position="middle",
        condition="clean", family="clean", seed=1, source_hash="h",
    )
    assert json.loads(json.dumps(spec.to_dict()))["query_id"] == "q"
