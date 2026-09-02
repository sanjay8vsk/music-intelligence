"""Stage 4 evaluation foundation: keys, synthetic fixtures, metrics, harness.

No detector exists yet, and none is implemented here. These tests exercise the
machinery that will judge one -- because a metric that is wrong in the same
direction as a detector's bug is worse than no metric.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from musicintel.analysis.evaluation import (
    NO_KEY, NO_TEMPO, Prediction, evaluate_bpm, evaluate_key, format_confusion,
)
from musicintel.analysis.fixtures import (
    DEFAULT_SR, click_track, midi_to_hz, synthetic_fixtures, tonal_progression,
)
from musicintel.analysis.keys import (
    ALL_KEYS, KEY_LABELS, Key, KeyParseError, mirex_score, normalize_key,
    parse_key, relation,
)

REPO = Path(__file__).resolve().parent.parent
MANIFEST = REPO / "eval/fixtures/bpm_key_annotation_manifest.json"


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ------------------------------------------------------------------ keys --
class TestKeyRepresentation:
    def test_there_are_exactly_24_classes_with_unique_indices(self):
        assert len(ALL_KEYS) == 24
        assert len({k.index for k in ALL_KEYS}) == 24
        assert len(KEY_LABELS) == 24

    @pytest.mark.parametrize("text,expected", [
        ("C", "C major"), ("C major", "C major"), ("Cmaj", "C major"),
        ("c#m", "C# minor"), ("Db minor", "C# minor"), ("F# Minor", "F# minor"),
        ("Bbm", "A# minor"), ("A maj", "A major"), ("E♭ major", "D# major"),
    ])
    def test_parsing_and_enharmonic_normalisation(self, text, expected):
        assert normalize_key(text) == expected

    def test_enharmonic_spellings_are_the_same_class(self):
        assert parse_key("Db minor") == parse_key("C# minor")
        assert parse_key("Gb major") == parse_key("F# major")

    @pytest.mark.parametrize("bad", ["", "H major", "C lydian", "major", "X#m", None])
    def test_unparseable_input_is_refused_not_guessed(self, bad):
        with pytest.raises(KeyParseError):
            parse_key(bad)

    def test_the_musical_relations(self):
        c = parse_key("C major")
        assert str(c.relative()) == "A minor"
        assert str(c.parallel()) == "C minor"
        assert str(c.dominant()) == "G major"
        assert str(c.subdominant()) == "F major"
        assert parse_key("A minor").relative() == c   # symmetric

    @pytest.mark.parametrize("pred,rel,score", [
        ("C major", "exact", 1.0), ("G major", "dominant", 0.5),
        ("F major", "subdominant", 0.5), ("A minor", "relative", 0.3),
        ("C minor", "parallel", 0.2), ("E major", "other", 0.0),
    ])
    def test_relations_and_mirex_weights(self, pred, rel, score):
        truth = parse_key("C major")
        assert relation(truth, parse_key(pred)) == rel
        assert mirex_score(truth, parse_key(pred)) == score

    def test_relation_is_defined_for_every_pair(self):
        allowed = {"exact", "relative", "parallel", "dominant", "subdominant", "other"}
        for a in ALL_KEYS:
            for b in ALL_KEYS:
                assert relation(a, b) in allowed


# -------------------------------------------------------------- fixtures --
class TestSyntheticFixtures:
    def test_the_set_has_the_expected_shape(self):
        fx = synthetic_fixtures()
        bpm = [f for f in fx if f.kind == "bpm"]
        key = [f for f in fx if f.kind == "key"]
        assert len(bpm) == 14 and len(key) == 24
        assert len({f.fixture_id for f in fx}) == len(fx)
        assert {f.key for f in key} == {str(k) for k in ALL_KEYS}

    def test_every_tempo_has_its_octave_partner_present(self):
        """So a 2x/1/2x error is visibly an octave error, not an unrelated one."""
        values = {f.bpm for f in synthetic_fixtures() if f.kind == "bpm"}
        pairs = [(v, v * 2) for v in values if v * 2 in values]
        assert len(pairs) >= 4, f"only {len(pairs)} octave pairs: {pairs}"

    def test_generation_is_deterministic(self):
        for f in synthetic_fixtures()[:6]:
            assert np.array_equal(f.render(), f.render())

    def test_two_processes_generate_identical_audio(self):
        """Determinism must survive a fresh interpreter, not just a fresh call."""
        code = (f"import sys; sys.path.insert(0, {str(REPO)!r});"
                "from musicintel.analysis.fixtures import click_track;"
                "import hashlib;"
                "print(hashlib.sha256(click_track(120.0, 5.0, fixture_id='x')"
                ".tobytes()).hexdigest())")
        a = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        b = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        assert a.returncode == 0 and a.stdout.strip() == b.stdout.strip()

    def test_click_track_length_and_level(self):
        y = click_track(120.0, 4.0, sr=DEFAULT_SR, fixture_id="t")
        assert y.shape == (4 * DEFAULT_SR,)
        assert y.dtype == np.float32
        assert 0.0 < float(np.abs(y).max()) <= 0.71

    def test_click_track_actually_has_the_requested_period(self):
        """Ground truth is only ground truth if the signal really has it."""
        sr, bpm = DEFAULT_SR, 120.0
        y = click_track(bpm, 8.0, sr=sr, fixture_id="p")
        energy = np.abs(y)
        peaks = np.flatnonzero(energy > 0.25 * energy.max())
        # collapse each burst to its first sample
        starts = [peaks[0]] + [b for a, b in zip(peaks, peaks[1:]) if b - a > sr * 0.05]
        gaps = np.diff(starts) / sr
        assert np.allclose(gaps, 60.0 / bpm, atol=0.01), f"gaps {gaps[:5]}"

    def test_a_nonpositive_tempo_is_refused(self):
        with pytest.raises(ValueError):
            click_track(0.0, 1.0)

    def test_tonal_progression_is_finite_and_bounded(self):
        for k in (parse_key("C major"), parse_key("F# minor")):
            y = tonal_progression(k, 4.0, fixture_id="k")
            assert np.all(np.isfinite(y)) and float(np.abs(y).max()) <= 0.71

    def test_midi_reference_pitch(self):
        assert midi_to_hz(69) == pytest.approx(440.0)
        assert midi_to_hz(60) == pytest.approx(261.626, abs=0.01)


# --------------------------------------------------------------- metrics --
class TestBpmMetrics:
    def test_tolerance_boundary(self):
        r = evaluate_bpm([Prediction("a", 100.0, 102.0),      # exactly 2%
                          Prediction("b", 100.0, 102.5)])     # 2.5%
        assert r["raw_accuracy"] == 0.5

    def test_raw_and_octave_tolerant_are_separate(self):
        r = evaluate_bpm([Prediction("a", 120.0, 120.0),
                          Prediction("b", 120.0, 240.0),
                          Prediction("c", 120.0, 60.0)])
        assert r["raw_accuracy"] == pytest.approx(1 / 3, abs=0.001)
        assert r["octave_tolerant_accuracy"] == 1.0
        assert "accuracy" not in {k for k in r if k.endswith("accuracy")} - {
            "raw_accuracy", "octave_tolerant_accuracy", "triple_only_accuracy"}

    def test_double_and_half_errors_are_counted_explicitly(self):
        r = evaluate_bpm([Prediction("a", 100.0, 200.0), Prediction("b", 100.0, 50.0),
                          Prediction("c", 100.0, 137.0)])
        assert r["double_errors"] == 1 and r["half_errors"] == 1

    def test_no_stable_tempo_is_excluded_not_failed(self):
        r = evaluate_bpm([Prediction("a", 120.0, 120.0), Prediction("b", NO_TEMPO, None)])
        assert r["excluded_no_stable_tempo"] == 1
        assert r["denominator"] == 1 and r["raw_accuracy"] == 1.0
        assert r["excluded_tracks"] == ["b"]

    def test_a_detector_crash_counts_as_wrong(self):
        r = evaluate_bpm([Prediction("a", 120.0, 120.0),
                          Prediction("b", 120.0, None, "RuntimeError: boom")])
        assert r["failed"] == 1 and r["denominator"] == 2
        assert r["raw_accuracy"] == 0.5
        assert r["failures"][0]["error"].startswith("RuntimeError")

    def test_error_distribution_is_reported(self):
        r = evaluate_bpm([Prediction(str(i), 100.0, 100.0 + i) for i in range(10)])
        d = r["relative_error_distribution"]
        assert {"p50", "p95", "min", "max", "mean"} <= set(d)

    def test_empty_input_does_not_divide_by_zero(self):
        r = evaluate_bpm([])
        assert r["denominator"] == 0 and r["raw_accuracy"] is None


class TestKeyMetrics:
    def test_exact_accuracy_and_mirex_differ_appropriately(self):
        r = evaluate_key([Prediction("a", "C major", "C major"),
                          Prediction("b", "C major", "G major")])
        assert r["exact_accuracy"] == 0.5
        assert r["mirex_weighted_score"] == 0.75      # (1.0 + 0.5) / 2

    def test_relation_breakdown_is_always_present(self):
        r = evaluate_key([Prediction("a", "C major", "A minor")])
        assert r["relation_breakdown"]["relative"] == 1
        assert set(r["relation_breakdown"]) == {
            "exact", "relative", "parallel", "dominant", "subdominant", "other"}

    def test_confusion_matrix_is_24x24_and_indexed_correctly(self):
        r = evaluate_key([Prediction("a", "C major", "A minor")])
        m = r["confusion_matrix"]
        assert len(m) == 24 and all(len(row) == 24 for row in m)
        assert m[parse_key("C major").index][parse_key("A minor").index] == 1
        assert sum(sum(row) for row in m) == 1

    def test_enharmonic_labels_land_in_the_same_cell(self):
        r = evaluate_key([Prediction("a", "Db minor", "C# minor")])
        assert r["exact_accuracy"] == 1.0

    def test_no_tonal_centre_is_excluded_not_failed(self):
        r = evaluate_key([Prediction("a", "C major", "C major"),
                          Prediction("b", NO_KEY, None)])
        assert r["excluded_no_tonal_centre"] == 1
        assert r["denominator"] == 1 and r["exact_accuracy"] == 1.0

    def test_a_detector_crash_counts_as_wrong(self):
        r = evaluate_key([Prediction("a", "C major", "C major"),
                          Prediction("b", "D major", None, "ValueError: x")])
        assert r["failed"] == 1 and r["denominator"] == 2
        assert r["exact_accuracy"] == 0.5
        assert r["mirex_weighted_score"] == 0.5      # the failure contributes 0.0

    def test_confusion_formatting_is_readable(self):
        r = evaluate_key([Prediction("a", "C major", "A minor")])
        text = format_confusion(r["confusion_matrix"])
        assert "C major" in text and "A minor" in text

    def test_empty_input_does_not_divide_by_zero(self):
        r = evaluate_key([])
        assert r["denominator"] == 0 and r["exact_accuracy"] is None


# -------------------------------------------------------------- manifest --
class TestAnnotationManifest:
    def test_it_exists_and_is_unlabelled(self):
        d = json.loads(MANIFEST.read_text())
        assert d["annotated_count"] == 0
        assert all(t["bpm"] is None for t in d["tracks"])
        assert all(t["key"] is None for t in d["tracks"])
        assert all(t["annotation_status"] == "pending" for t in d["tracks"])

    def test_it_carries_the_required_schema(self):
        d = json.loads(MANIFEST.read_text())
        required = {"track_id", "path", "sha256", "title", "artist", "duration_sec",
                    "license", "license_url", "bpm", "key", "annotation_status",
                    "annotator", "annotated_utc", "notes"}
        assert required <= set(d["tracks"][0])
        assert len(d["content_hash"]) == 64
        assert 50 <= d["track_count"] <= 100

    def test_every_track_is_permissively_licensed(self):
        d = json.loads(MANIFEST.read_text())
        forbidden = ("-nd", "-nc", "by-nc", "by-nd", "nc-sa", "nc-nd")
        for t in d["tracks"]:
            blob = f"{t['license']} {t['license_url']}".lower()
            assert not any(tok in blob for tok in forbidden), t["track_id"]

    def test_ordering_is_deterministic(self):
        d = json.loads(MANIFEST.read_text())
        ids = [t["track_id"] for t in d["tracks"]]
        assert ids == sorted(ids)

    def test_the_builder_reproduces_it_byte_for_byte(self, tmp_path):
        mod = _load("bkm", "scripts/build_bpm_key_manifest.py")
        out = tmp_path / "m.json"
        assert mod.main(["--count", "60", "--out", str(out)]) == 0
        assert out.read_text() == MANIFEST.read_text()

    def test_the_builder_never_writes_a_label(self, tmp_path):
        mod = _load("bkm2", "scripts/build_bpm_key_manifest.py")
        out = tmp_path / "m.json"
        mod.main(["--count", "20", "--out", str(out)])
        d = json.loads(out.read_text())
        assert all(t["bpm"] is None and t["key"] is None for t in d["tracks"])


# --------------------------------------------------------------- harness --
class TestHarness:
    def test_it_refuses_to_report_without_a_detector(self, capsys):
        mod = _load("ebk", "scripts/eval_bpm_key.py")
        assert mod.main(["--synthetic", "--dry-run"]) == 0
        assert "NONE CONFIGURED" in capsys.readouterr().out

    def test_it_scores_an_oracle_end_to_end(self):
        """Plumbing only -- an oracle stub, not a detector implementation.

        An oracle must score 100%. If it does not, the harness is miswiring
        truth to predictions and every future detector number is meaningless.
        """
        mod = _load("ebk2", "scripts/eval_bpm_key.py")
        items = mod.synthetic_items()
        truth = {id(i): i for i in items}
        seen = []

        def oracle(samples, sr, _items=iter(items)):
            item = next(_items)
            seen.append(item["id"])
            return {"bpm": item["bpm_truth"], "key": item["key_truth"]}

        results = mod.run(oracle, items, warmup=0)
        assert results["bpm"]["denominator"] == 14
        assert results["key"]["denominator"] == 24
        assert results["bpm"]["failed"] == 0
        assert results["bpm"]["raw_accuracy"] == 1.0
        assert results["key"]["exact_accuracy"] == 1.0
        assert results["key"]["mirex_weighted_score"] == 1.0
        assert seen == [i["id"] for i in items]      # order preserved
        assert results["timing_ms"]["total_ms"]["n"] == len(items)

    def test_returning_no_answer_counts_as_a_failure_not_a_free_pass(self):
        """A detector that declines to answer must not improve its own score."""
        mod = _load("ebk5", "scripts/eval_bpm_key.py")
        items = mod.synthetic_items()
        results = mod.run(lambda s, sr: {"bpm": None, "key": None}, items, warmup=0)
        assert results["bpm"]["failed"] == 14
        assert results["key"]["failed"] == 24
        assert results["bpm"]["raw_accuracy"] == 0.0
        assert results["key"]["mirex_weighted_score"] == 0.0

    def test_a_raising_detector_is_recorded_as_failures(self):
        mod = _load("ebk3", "scripts/eval_bpm_key.py")
        items = mod.synthetic_items()[:4]
        def broken(samples, sr):
            raise RuntimeError("detector exploded")
        results = mod.run(broken, items, warmup=0)
        failed = (results["bpm"]["failed"] if results["bpm"] else 0)
        assert failed == sum(1 for i in items if i["bpm_truth"] is not None)
        assert "detector exploded" in results["bpm"]["failures"][0]["error"]

    def test_timing_excludes_nothing_it_claims_to_include(self):
        mod = _load("ebk4", "scripts/eval_bpm_key.py")
        items = mod.synthetic_items()[:3]
        results = mod.run(lambda s, sr: {"bpm": 120.0, "key": "C major"},
                          items, warmup=0)
        t = results["timing_ms"]["total_ms"]
        assert t["n"] == 3 and t["p50"] >= 0


class TestArchitectureBoundary:
    def test_recognition_does_not_import_analysis(self):
        for f in (REPO / "musicintel/recognition").glob("*.py"):
            assert "musicintel.analysis" not in f.read_text(), f

    def test_analysis_does_not_import_recognition(self):
        for f in (REPO / "musicintel/analysis").glob("*.py"):
            text = f.read_text()
            assert "musicintel.recognition" not in text, f

    def test_no_detector_has_been_implemented(self):
        """This task builds evaluation only. A detector would need its own review."""
        names = {f.name for f in (REPO / "musicintel/analysis").glob("*.py")}
        assert names == {"__init__.py", "features.py", "keys.py",
                         "fixtures.py", "evaluation.py"}, names
