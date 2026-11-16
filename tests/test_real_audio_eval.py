"""Stage 4 real-audio evaluation path: sheet -> validated labels -> decoded audio.

Every test builds its own 20-track pilot world in a temporary directory -- real
audio files, a manifest pinning their sha256, and a pilot sheet in the exact
shipped format -- and drives the real CLI. Labels here are TEST INPUTS written
by the test, never ground truth for any real recording, and the detectors are
stubs that return fixed answers: no BPM or key detection exists anywhere.

The real repository sheet and manifest are only ever read, and one test proves
they are byte-identical after the harness has run against them.
"""

from __future__ import annotations

import csv
import dataclasses
import hashlib
import importlib.util
import io
import itertools
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

REPO = Path(__file__).resolve().parent.parent
REAL_CSV = REPO / "eval/fixtures/bpm_key_pilot_annotation.csv"
REAL_MANIFEST = REPO / "eval/fixtures/bpm_key_annotation_manifest.json"
NATIVE_SR = 44100               # deliberately != the analysis rate: decode must resample
WRITE_SECONDS = {i: round(1.0 + 0.1 * i, 1) for i in range(20)}   # unique per track


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ebk = _load("ebk_real", "scripts/eval_bpm_key.py")
vpa = _load("vpa_real", "scripts/validate_pilot_annotations.py")

from musicintel.analysis.evaluation import (  # noqa: E402
    ANALYSIS_SAMPLE_RATE, ANNOTATED, EXCLUDED, MANIFEST_STATUSES, NO_KEY, NO_TEMPO,
    PENDING, PILOT_STATUSES,
)
from musicintel.recognition.fingerprint import FingerprintConfig, load_audio  # noqa: E402

_counter = itertools.count()

DETECTORS = '''
import hashlib
import time
import numpy as np

CALLS = []
TIMINGS = None          # what `with_timings` reports; set by the test
WORK_SECONDS = 0.02     # how long `timed` works for

def _log(samples, sr, **extra):
    CALLS.append({"n": int(samples.size), "sr": int(sr), "dtype": str(samples.dtype),
                  "ndim": int(samples.ndim),
                  "sha": hashlib.sha256(np.ascontiguousarray(samples).tobytes()).hexdigest(),
                  **extra})

def constant(samples, sr):
    _log(samples, sr)
    return {"bpm": 120.0, "key": "C major"}

def with_timings(samples, sr):
    _log(samples, sr)
    out = {"bpm": 120.0, "key": "C major"}
    if TIMINGS != "absent":
        out["timings"] = TIMINGS
    return out

ANSWER = None           # what `answering` returns; set by the test

def answering(samples, sr):
    _log(samples, sr)
    return ANSWER

def timed(samples, sr):
    """Measures its OWN execution, independently of the harness."""
    started = time.perf_counter()
    time.sleep(WORK_SECONDS)
    result = {"bpm": 120.0, "key": "C major"}
    _log(samples, sr, self_ms=(time.perf_counter() - started) * 1000.0)
    return result

def raising(samples, sr):
    _log(samples, sr)
    raise RuntimeError("detector exploded")

def declining(samples, sr):
    _log(samples, sr)
    return {"bpm": "no_stable_tempo", "key": "no_tonal_centre"}

def garbage(samples, sr):
    _log(samples, sr)
    return {"bpm": "fast", "key": "H major"}

def not_a_dict(samples, sr):
    _log(samples, sr)
    return 120.0
'''


# ---------------------------------------------------------------- world --
@dataclasses.dataclass
class World:
    root: Path
    csv: Path
    manifest: Path
    reports: Path
    tracks: list[dict]
    det: object            # the stub detector module
    det_name: str

    def audio(self, tid: str) -> Path:
        return self.root / next(t["path"] for t in self.tracks if t["track_id"] == tid)

    def tid(self, i: int) -> str:
        return self.tracks[i]["track_id"]

    def annotate(self, labels: dict[int, tuple[str, str]], *,
                 order: list[int] | None = None, **kw) -> None:
        """Write the sheet with rows {index: (bpm, key)} annotated, rest pending.
        `order` permutes the sheet's rows (the validator accepts any order)."""
        _write_sheet(self.csv, self.tracks, labels, order=order, **kw)

    def set_manifest(self, i: int, **fields) -> None:
        """Change track i's manifest fields (a TEST manifest, never the real one)."""
        self.tracks[i].update(fields)
        self.manifest.write_text(json.dumps({"content_hash": "0" * 64,
                                             "fixture_set": "test",
                                             "tracks": self.tracks}, indent=2))

    def run(self, *extra: str, detector: str | None = "constant") -> int:
        argv = ["--real", "--pilot-csv", str(self.csv), "--manifest", str(self.manifest),
                "--audio-root", str(self.root), "--report-dir", str(self.reports),
                "--warmup", "0", *extra]
        if detector:
            argv += ["--detector", f"{self.det_name}:{detector}"]
        return ebk.main(argv)

    def report(self) -> dict:
        return json.loads((self.reports / ebk.REPORT_FILENAMES[ebk.REAL_AUDIO]).read_text())

    def no_report_written(self) -> bool:
        return not self.reports.exists() or not any(self.reports.iterdir())


def _tone(i: int, seconds: float) -> np.ndarray:
    t = np.arange(int(seconds * NATIVE_SR)) / NATIVE_SR
    left = 0.3 * np.sin(2 * np.pi * (220.0 + 37.0 * i) * t)
    right = 0.2 * np.sin(2 * np.pi * (330.0 + 41.0 * i) * t)
    return np.stack([left, right], axis=1).astype(np.float32)     # stereo


def _write_sheet(path: Path, tracks: list[dict], labels: dict[int, tuple[str, str]],
                 *, annotator: str = "test-annotator",
                 when: str = "2026-11-16T10:00:00Z",
                 order: list[int] | None = None) -> None:
    rows = []
    for i in (order if order is not None else range(len(tracks))):
        t = tracks[i]
        row = {"track_id": t["track_id"], "path": t["path"], "title": t["title"],
               "artist": t["artist"], "duration_sec": t["duration_sec"],
               "license": t["license"], "license_url": t["license_url"],
               "bpm": "", "key": "", "annotation_status": PENDING,
               "annotator": "", "annotated_utc": "", "notes": ""}
        if i in labels:
            bpm, key = labels[i]
            row.update(bpm=bpm, key=key, annotation_status=ANNOTATED,
                       annotator=annotator, annotated_utc=when, notes="test input")
        rows.append(row)
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=vpa.COLUMNS, lineterminator="\n")
    w.writeheader()
    w.writerows(rows)
    path.write_text("# test pilot sheet -- labels are test inputs\n" + buf.getvalue())


@pytest.fixture
def world(tmp_path, monkeypatch) -> World:
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    tracks = []
    for i in range(20):
        tid = f"t{i:02d}"
        rel = f"audio/{tid}.wav"
        sf.write(tmp_path / rel, _tone(i, WRITE_SECONDS[i]), NATIVE_SR, subtype="FLOAT")
        tracks.append({
            "track_id": tid, "path": rel,
            "sha256": hashlib.sha256((tmp_path / rel).read_bytes()).hexdigest(),
            "title": f"Title {i}", "artist": f"Artist {i}",
            "duration_sec": WRITE_SECONDS[i], "license": "CC0-1.0",
            "license_url": "http://creativecommons.org/publicdomain/zero/1.0/",
            "bpm": None, "key": None, "annotation_status": PENDING,
            "annotator": None, "annotated_utc": None, "notes": "",
        })
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"content_hash": "0" * 64, "fixture_set": "test",
                                    "tracks": tracks}, indent=2))
    csv_path = tmp_path / "pilot.csv"
    _write_sheet(csv_path, tracks, {})

    name = f"stub_detectors_{next(_counter)}"
    (tmp_path / f"{name}.py").write_text(DETECTORS)
    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()
    det = importlib.import_module(name)
    yield World(tmp_path, csv_path, manifest, tmp_path / "reports", tracks, det, name)
    sys.modules.pop(name, None)


def _ids(per_track: list[dict]) -> set[str]:
    return {p["track_id"] for p in per_track}


# ------------------------------------------------------- row selection --
class TestOnlyAnnotatedRowsAreEvaluated:
    def test_a_mixed_sheet_scores_exactly_the_annotated_rows(self, world):
        world.annotate({3: ("120", "C major"), 7: ("128", "A minor"),
                        11: ("96.5", "G major")})
        assert world.run() == ebk.EXIT_OK
        r = world.report()
        expected = {world.tid(3), world.tid(7), world.tid(11)}
        assert set(r["dataset"]["scored_tracks"]) == expected
        assert set(r["dataset"]["annotated_tracks"]) == expected
        assert r["dataset"]["annotated"] == 3 and r["dataset"]["pending"] == 17
        assert _ids(r["results"]["bpm"]["per_track"]) == expected
        assert _ids(r["results"]["key"]["per_track"]) == expected
        # the detector saw three recordings and nothing else
        assert len(world.det.CALLS) == 3

    def test_pending_rows_are_never_scored_or_listed_as_scored(self, world):
        world.annotate({0: ("120", "C major")})
        assert world.run() == ebk.EXIT_OK
        r = world.report()
        pending = set(r["dataset"]["pending_tracks"])
        assert len(pending) == 19 and world.tid(0) not in pending
        scored = (_ids(r["results"]["bpm"]["per_track"])
                  | _ids(r["results"]["key"]["per_track"])
                  | set(r["results"]["bpm"]["excluded_tracks"])
                  | set(r["results"]["key"]["excluded_tracks"]))
        assert scored.isdisjoint(pending)

    def test_pending_rows_are_never_even_read(self, world):
        """Delete every pending row's audio: the run must not notice."""
        world.annotate({5: ("120", "C major")})
        for i in range(20):
            if i != 5:
                world.audio(world.tid(i)).unlink()
        assert world.run() == ebk.EXIT_OK
        assert world.report()["dataset"]["scored_tracks"] == [world.tid(5)]

    def test_the_detector_sees_each_annotated_recording_once_in_sheet_order(self, world):
        """Exact count, exact order, exact recording per call.

        The sheet's rows are permuted so that sheet order differs from both
        track-id order and annotation-dict order; every recording is distinct
        audio, so each call is identified by the sha256 of the samples the
        detector actually received, against an independent decode."""
        order = [17, 4, 11, 0, 19, 2, 9, 14, 6, 1, 3, 5, 7, 8, 10, 12, 13, 15, 16, 18]
        world.annotate({2: ("100", "D major"), 9: ("140", "E minor"),
                        14: ("90", "G major"), 17: ("128", "A minor")}, order=order)
        assert world.run() == ebk.EXIT_OK

        expected_ids = [world.tid(i) for i in (17, 2, 9, 14)]   # sheet order
        cfg = dataclasses.replace(FingerprintConfig(), sample_rate=ANALYSIS_SAMPLE_RATE)

        def digest(tid):
            y, _ = load_audio(world.audio(tid), cfg)
            return hashlib.sha256(
                np.ascontiguousarray(y, dtype=np.float32).tobytes()).hexdigest()

        expected = [digest(t) for t in expected_ids]
        assert len(set(expected)) == len(expected)        # calls are distinguishable
        got = [c["sha"] for c in world.det.CALLS]
        assert len(got) == len(expected_ids) == 4         # exact invocation count
        assert got == expected                            # exact order + recording
        assert world.report()["dataset"]["scored_tracks"] == expected_ids


# --------------------------------------------------------- exclusions --
class TestExclusionCategories:
    def test_no_stable_tempo_is_counted_and_removed_from_the_bpm_denominator(self, world):
        world.annotate({1: ("120", "C major"), 4: (NO_TEMPO, "C major")})
        assert world.run() == ebk.EXIT_OK
        bpm = world.report()["results"]["bpm"]
        assert bpm["excluded_no_stable_tempo"] == 1
        assert bpm["excluded_tracks"] == [world.tid(4)]
        assert bpm["denominator"] == 1 and bpm["raw_accuracy"] == 1.0
        # the key label on that row is still scored
        key = world.report()["results"]["key"]
        assert key["denominator"] == 2 and key["exact_accuracy"] == 1.0
        assert world.report()["dataset"]["no_stable_tempo"] == 1

    def test_no_tonal_centre_is_counted_and_removed_from_the_key_denominator(self, world):
        world.annotate({1: ("120", "C major"), 6: ("120", NO_KEY)})
        assert world.run() == ebk.EXIT_OK
        r = world.report()
        key = r["results"]["key"]
        assert key["excluded_no_tonal_centre"] == 1
        assert key["excluded_tracks"] == [world.tid(6)]
        assert key["denominator"] == 1
        assert r["results"]["bpm"]["denominator"] == 2
        assert r["dataset"]["no_tonal_centre"] == 1

    def test_a_row_with_both_exclusions_is_counted_but_never_decoded(self, world):
        world.annotate({1: ("120", "C major"), 8: (NO_TEMPO, NO_KEY)})
        world.audio(world.tid(8)).unlink()           # proves it is never read
        assert world.run() == ebk.EXIT_OK
        r = world.report()
        assert r["dataset"]["exclusion_only_tracks"] == [world.tid(8)]
        assert world.tid(8) not in r["dataset"]["scored_tracks"]
        assert r["results"]["bpm"]["excluded_tracks"] == [world.tid(8)]
        assert r["results"]["key"]["excluded_tracks"] == [world.tid(8)]
        assert len(world.det.CALLS) == 1

    def test_only_exclusions_means_nothing_to_evaluate(self, world):
        world.annotate({0: (NO_TEMPO, NO_KEY), 1: (NO_TEMPO, NO_KEY)})
        assert world.run() == ebk.EXIT_NOTHING_TO_EVALUATE
        assert world.no_report_written() and world.det.CALLS == []


# ----------------------------------------------------------- failures --
class TestDetectorFailuresCountAsWrong:
    def test_a_raising_detector_fails_every_scorable_label(self, world):
        world.annotate({0: ("120", "C major"), 1: ("90", "A minor"),
                        2: (NO_TEMPO, "F major")})
        assert world.run(detector="raising") == ebk.EXIT_OK
        r = world.report()["results"]
        assert r["bpm"]["failed"] == 2 and r["bpm"]["denominator"] == 2
        assert r["bpm"]["raw_accuracy"] == 0.0
        assert r["key"]["failed"] == 3 and r["key"]["denominator"] == 3
        assert r["key"]["exact_accuracy"] == 0.0
        assert all("detector exploded" in f["error"] for f in r["bpm"]["failures"])

    @pytest.mark.parametrize("detector,needle", [
        ("declining", "declined"),
        ("garbage", "invalid"),
        ("not_a_dict", "expected a dict"),
    ])
    def test_declined_invalid_or_malformed_answers_are_failures(self, world, detector,
                                                                needle):
        world.annotate({0: ("120", "C major"), 1: ("90", "A minor")})
        assert world.run(detector=detector) == ebk.EXIT_OK
        r = world.report()["results"]
        assert r["bpm"]["failed"] == 2 and r["key"]["failed"] == 2
        assert r["bpm"]["raw_accuracy"] == 0.0 and r["key"]["mirex_weighted_score"] == 0.0
        assert all(needle in f["error"] for f in r["bpm"]["failures"] + r["key"]["failures"])

    def test_scores_are_computed_against_the_human_labels(self, world):
        """Constant 120 / C major against labels it gets partly right."""
        world.annotate({0: ("120", "C major"),     # both right
                        1: ("60", "G major"),      # half-time; dominant
                        2: ("143.2", "A minor")})  # wrong; relative
        assert world.run() == ebk.EXIT_OK
        r = world.report()["results"]
        assert r["bpm"]["raw_accuracy"] == pytest.approx(1 / 3, abs=1e-4)
        assert r["bpm"]["octave_tolerant_accuracy"] == pytest.approx(2 / 3, abs=1e-4)
        assert r["bpm"]["double_errors"] == 1
        assert r["key"]["relation_breakdown"]["exact"] == 1
        assert r["key"]["relation_breakdown"]["subdominant"] == 1   # C is IV of G
        assert r["key"]["relation_breakdown"]["relative"] == 1


# ----------------------------------------------------------- refusals --
class TestRefusals:
    def test_zero_annotated_rows_exits_nonzero_and_writes_no_report(self, world):
        assert world.run() == ebk.EXIT_NOTHING_TO_EVALUATE
        assert world.no_report_written() and world.det.CALLS == []

    def test_the_shipped_sheet_is_refused_and_left_byte_identical(self, tmp_path):
        """The real repository sheet is entirely pending: the harness must refuse
        it, write nothing, and leave both annotation sources untouched."""
        before = {p: p.read_bytes() for p in (REAL_CSV, REAL_MANIFEST)}
        code = ebk.main(["--real", "--detector", "builtins:print",
                         "--report-dir", str(tmp_path / "reports"), "--warmup", "0"])
        assert code == ebk.EXIT_NOTHING_TO_EVALUATE
        assert not (tmp_path / "reports").exists()
        assert {p: p.read_bytes() for p in before} == before

    def test_an_invalid_sheet_is_refused_before_any_audio_or_detector(self, world):
        world.annotate({0: ("120", "Db minor")})          # flat spelling: invalid
        assert world.run() == ebk.EXIT_INVALID_ANNOTATIONS
        assert world.no_report_written() and world.det.CALLS == []

    def test_an_invalid_sheet_is_reported_even_without_a_detector(self, world):
        world.annotate({0: ("fast", "C major")})
        assert world.run(detector=None) == ebk.EXIT_INVALID_ANNOTATIONS

    def test_missing_audio_for_an_annotated_row_is_refused(self, world):
        world.annotate({0: ("120", "C major"), 1: ("90", "A minor")})
        world.audio(world.tid(1)).unlink()
        assert world.run() == ebk.EXIT_AUDIO_ERROR
        assert world.no_report_written() and world.det.CALLS == []

    def test_audio_that_differs_from_the_manifest_is_refused(self, world):
        """A label is only valid for the recording it was made from."""
        world.annotate({0: ("120", "C major")})
        sf.write(world.audio(world.tid(0)), _tone(19, 1.0), NATIVE_SR, subtype="FLOAT")
        assert world.run() == ebk.EXIT_AUDIO_ERROR
        assert world.no_report_written() and world.det.CALLS == []

    def test_undecodable_audio_is_refused_not_counted_as_a_detector_failure(self, world):
        world.annotate({0: ("120", "C major")})
        bad = world.audio(world.tid(0))
        bad.write_bytes(b"not audio at all")
        world.tracks[0]["sha256"] = hashlib.sha256(bad.read_bytes()).hexdigest()
        world.manifest.write_text(json.dumps({"content_hash": "0" * 64,
                                              "tracks": world.tracks}))
        assert world.run() == ebk.EXIT_AUDIO_ERROR
        assert world.no_report_written() and world.det.CALLS == []

    def test_a_detector_requires_an_explicit_mode(self, world):
        with pytest.raises(SystemExit):
            ebk.main(["--detector", f"{world.det_name}:constant"])

    def test_synthetic_and_real_cannot_be_requested_together(self, world):
        with pytest.raises(SystemExit):
            ebk.main(["--synthetic", "--real", "--detector", f"{world.det_name}:constant"])

    def test_dry_run_evaluates_but_writes_nothing(self, world):
        world.annotate({0: ("120", "C major")})
        assert world.run("--dry-run") == ebk.EXIT_OK
        assert world.no_report_written() and len(world.det.CALLS) == 1


# ------------------------------------------------------------- decode --
class TestDecoding:
    def test_audio_is_decoded_by_the_project_loader_at_the_declared_rate(self, world):
        world.annotate({4: ("120", "C major")})
        assert world.run() == ebk.EXIT_OK
        (call,) = world.det.CALLS
        assert call["sr"] == ANALYSIS_SAMPLE_RATE == 22050
        assert call["dtype"] == "float32" and call["ndim"] == 1     # mono
        # byte-identical to what the project's own loader produces at that rate
        expected, _ = load_audio(world.audio(world.tid(4)),
                                 dataclasses.replace(FingerprintConfig(),
                                                     sample_rate=ANALYSIS_SAMPLE_RATE))
        assert call["sha"] == hashlib.sha256(
            np.ascontiguousarray(expected, dtype=np.float32).tobytes()).hexdigest()
        r = world.report()
        assert r["analysis_sample_rate"] == 22050
        assert "load_audio" in r["decoder"]
        assert r["results"]["timing_ms"]["decode_ms"]["n"] == 1

    def test_an_overridden_rate_is_used_and_recorded(self, world):
        world.annotate({4: ("120", "C major")})
        assert world.run("--sample-rate", "16000") == ebk.EXIT_OK
        (call,) = world.det.CALLS
        assert call["sr"] == 16000
        assert abs(call["n"] - WRITE_SECONDS[4] * 16000) <= 2
        assert world.report()["analysis_sample_rate"] == 16000

    def test_decode_time_is_kept_out_of_detector_timing(self, world, monkeypatch):
        """A controlled 400 ms delay in decoding against a ~20 ms detector that
        times itself. If decode leaked into the detector measurement, total_ms
        would be >= 400 ms; it must instead track the detector's own clock."""
        decode_delay_s = 0.4
        decode_calls = []
        real_load = ebk.load_audio

        def slow_load(path, cfg):
            started = time.perf_counter()
            time.sleep(decode_delay_s)
            out = real_load(path, cfg)
            decode_calls.append((time.perf_counter() - started) * 1000.0)
            return out

        monkeypatch.setattr(ebk, "load_audio", slow_load)
        world.annotate({0: ("120", "C major"), 1: ("90", "A minor")})
        assert world.run(detector="timed") == ebk.EXIT_OK

        t = world.report()["results"]["timing_ms"]
        self_ms = [c["self_ms"] for c in world.det.CALLS]
        assert len(decode_calls) == 2 and len(self_ms) == 2
        assert min(decode_calls) >= decode_delay_s * 1000
        # decode is measured, and every decode carried the delay
        assert t["decode_ms"]["n"] == 2
        assert t["decode_ms"]["p50"] >= decode_delay_s * 1000
        # detector execution: at least what the detector measured itself, and
        # nowhere near the decode delay it would contain if decode leaked in
        assert t["total_ms"]["n"] == 2
        assert t["total_ms"]["p50"] >= min(self_ms)
        assert t["total_ms"]["max"] < decode_delay_s * 1000 / 2, t["total_ms"]


# --------------------------------------------- synthetic vs real reports --
class TestReportsAreDistinguishable:
    def test_each_kind_writes_its_own_labelled_file(self, world):
        world.annotate({0: ("120", "C major")})
        assert world.run() == ebk.EXIT_OK
        assert ebk.main(["--synthetic", "--detector", f"{world.det_name}:constant",
                         "--report-dir", str(world.reports), "--warmup", "0"]) == ebk.EXIT_OK
        files = sorted(p.name for p in world.reports.iterdir())
        assert files == sorted(ebk.REPORT_FILENAMES.values())
        real = world.report()
        synth = json.loads((world.reports / ebk.REPORT_FILENAMES[ebk.SYNTHETIC]).read_text())

        assert real["evaluation_kind"] == "real_audio"
        assert synth["evaluation_kind"] == "synthetic"
        assert real["dataset"]["kind"] == "real_audio"
        assert synth["dataset"]["kind"] == "synthetic"
        assert "human annotation" in real["ground_truth"]
        assert "by construction" in synth["ground_truth"]
        assert "SYNTHETIC" in synth["title"] and "SYNTHETIC" not in real["title"]
        assert "pilot_csv" not in synth["dataset"]
        assert synth["decoder"] is None and real["decoder"]
        assert any("CANNOT substantiate" in s for s in synth["limitations"])
        assert any("human annotation" in s for s in real["limitations"])

    def test_the_real_report_records_its_resolution_and_annotator_limits(self, world):
        world.annotate({0: ("120", "C major"), 1: ("90", "A minor")})
        assert world.run() == ebk.EXIT_OK
        lim = " ".join(world.report()["limitations"])
        assert "50.0 percentage points" in lim       # 2 scored tracks
        assert "single annotator" in lim


# ----------------------------------------------- sources stay untouched --
class TestAnnotationSourcesAreReadOnly:
    def test_sheet_and_manifest_bytes_are_unchanged_after_evaluation(self, world):
        world.annotate({0: ("120", "C major"), 1: (NO_TEMPO, NO_KEY),
                        2: ("90", "A minor")})
        before = {p: p.read_bytes() for p in (world.csv, world.manifest)}
        for det in ("constant", "raising", "garbage"):
            assert world.run(detector=det) == ebk.EXIT_OK
        assert {p: p.read_bytes() for p in before} == before

    def test_the_report_pins_the_exact_sources_it_read(self, world):
        world.annotate({0: ("120", "C major")})
        assert world.run() == ebk.EXIT_OK
        d = world.report()["dataset"]
        assert d["pilot_csv_sha256"] == hashlib.sha256(world.csv.read_bytes()).hexdigest()
        assert d["manifest_sha256"] == hashlib.sha256(world.manifest.read_bytes()).hexdigest()

    def test_no_prediction_reaches_the_sheet(self, world):
        world.annotate({0: ("120", "C major")})
        assert world.run(detector="garbage") == ebk.EXIT_OK
        assert "fast" not in world.csv.read_text()
        assert "H major" not in world.csv.read_text()

    def test_a_source_changing_mid_run_is_refused(self, world, monkeypatch):
        world.annotate({0: ("120", "C major")})
        original = ebk.run

        def tampering_run(*a, **kw):
            out = original(*a, **kw)
            world.csv.write_text(world.csv.read_text() + "\n")
            return out

        monkeypatch.setattr(ebk, "run", tampering_run)
        assert world.run() == ebk.EXIT_SOURCES_CHANGED
        assert world.no_report_written()


# ------------------------------------------------------ vocabulary --
class TestSharedVocabulary:
    def test_there_is_one_no_key_sentinel(self):
        from musicintel.analysis import evaluation, keys
        assert keys.NO_KEY is evaluation.NO_KEY
        assert NO_KEY == "no_tonal_centre"
        assert NO_TEMPO == "no_stable_tempo"

    def test_the_validator_and_harness_share_the_status_vocabulary(self):
        assert vpa.STATUSES is PILOT_STATUSES
        assert PILOT_STATUSES == {PENDING, ANNOTATED} == {"pending", "annotated"}
        assert MANIFEST_STATUSES == PILOT_STATUSES | {EXCLUDED}
        assert EXCLUDED == "excluded"

    def test_exclusion_labels_and_the_exclusion_status_are_distinct_words(self):
        assert not {NO_TEMPO, NO_KEY} & MANIFEST_STATUSES

    def test_the_sheet_and_manifest_use_only_their_own_vocabulary(self):
        rows = vpa.read_rows(REAL_CSV)
        assert {r["annotation_status"] for r in rows} <= PILOT_STATUSES
        tracks = json.loads(REAL_MANIFEST.read_text())["tracks"]
        assert {t["annotation_status"] for t in tracks} <= MANIFEST_STATUSES

    def test_a_sheet_row_cannot_be_excluded(self, world):
        """Whole-recording exclusion is a manifest decision, not a sheet status."""
        world.annotate({})
        text = world.csv.read_text().replace(",pending,", ",excluded,", 1)
        world.csv.write_text(text)
        errors, _ = vpa.validate(world.csv, world.manifest)
        assert any("annotation_status" in e for e in errors)
        assert world.run() == ebk.EXIT_INVALID_ANNOTATIONS

    def test_the_builder_still_documents_manifest_exclusion(self):
        src = (REPO / "scripts/build_bpm_key_manifest.py").read_text()
        assert "pending | annotated | excluded" in src

    def test_the_declared_analysis_rate_is_the_fixture_rate(self):
        from musicintel.analysis.fixtures import DEFAULT_SR
        assert DEFAULT_SR == ANALYSIS_SAMPLE_RATE == 22050


# ------------------------------------------- manifest-level exclusion --
class TestManifestExclusion:
    """`excluded` on a manifest track withdraws the whole recording."""

    def test_an_excluded_recording_is_never_decoded_or_scored(self, world):
        world.set_manifest(3, annotation_status=EXCLUDED,
                           notes="test: damaged audio")
        world.annotate({0: ("120", "C major"), 3: ("90", "A minor")})
        world.audio(world.tid(3)).unlink()         # proves it is never read
        assert world.run() == ebk.EXIT_OK
        d = world.report()["dataset"]
        assert d["scored_tracks"] == [world.tid(0)]
        assert d["evaluated_tracks"] == [world.tid(0)]
        assert d["annotated_tracks"] == [world.tid(0), world.tid(3)]
        assert d["manifest_excluded"] == 1
        assert d["manifest_excluded_tracks"] == [
            {"track_id": world.tid(3), "pilot_status": ANNOTATED,
             "reason": "test: damaged audio"}]
        r = world.report()["results"]
        assert world.tid(3) not in r["bpm"]["excluded_tracks"]   # not a label exclusion
        assert r["bpm"]["denominator"] == 1 and len(world.det.CALLS) == 1
        assert any("excluded whole in the manifest" in s
                   for s in world.report()["limitations"])

    def test_it_is_not_counted_as_a_label_exclusion(self, world):
        world.set_manifest(3, annotation_status=EXCLUDED, notes="test: duplicate")
        world.annotate({0: ("120", "C major"), 3: (NO_TEMPO, NO_KEY)})
        assert world.run() == ebk.EXIT_OK
        r = world.report()
        assert r["results"]["bpm"]["excluded_no_stable_tempo"] == 0
        assert r["results"]["key"]["excluded_no_tonal_centre"] == 0
        assert r["dataset"]["exclusion_only_tracks"] == []

    def test_a_pending_row_of_an_excluded_recording_is_still_listed(self, world):
        world.set_manifest(5, annotation_status=EXCLUDED, notes="test: not music")
        world.annotate({0: ("120", "C major")})
        assert world.run() == ebk.EXIT_OK
        assert world.report()["dataset"]["manifest_excluded_tracks"] == [
            {"track_id": world.tid(5), "pilot_status": PENDING,
             "reason": "test: not music"}]

    def test_when_every_annotated_recording_is_excluded_nothing_is_evaluable(self, world):
        world.set_manifest(0, annotation_status=EXCLUDED, notes="test: licence")
        world.annotate({0: ("120", "C major")})
        assert world.run() == ebk.EXIT_NOTHING_TO_EVALUATE
        assert world.no_report_written() and world.det.CALLS == []

    def test_an_exclusion_without_a_reason_is_invalid(self, world):
        world.set_manifest(3, annotation_status=EXCLUDED, notes="  ")
        world.annotate({0: ("120", "C major")})
        errors, _ = vpa.validate(world.csv, world.manifest)
        assert any("requires the reason" in e for e in errors)
        assert world.run() == ebk.EXIT_INVALID_ANNOTATIONS
        assert world.no_report_written() and world.det.CALLS == []

    @pytest.mark.parametrize("status", ["withdrawn", "no_stable_tempo", "", None])
    def test_an_unknown_manifest_status_is_invalid(self, world, status):
        world.set_manifest(7, annotation_status=status)
        world.annotate({0: ("120", "C major")})
        errors, _ = vpa.validate(world.csv, world.manifest)
        assert any("manifest track" in e and "annotation_status" in e for e in errors)
        assert world.run() == ebk.EXIT_INVALID_ANNOTATIONS

    def test_the_validator_summary_names_excluded_pilot_recordings(self, world):
        world.set_manifest(4, annotation_status=EXCLUDED, notes="test: clipped")
        errors, summary = vpa.validate(world.csv, world.manifest)
        assert errors == [] and summary["manifest_excluded"] == [world.tid(4)]


# ------------------------------------------------------ missing inputs --
class TestUnreadableInputs:
    """Expected file-access problems: a named refusal, never a traceback."""

    def _assert_refused(self, world, capsys, code, *needles):
        out = capsys.readouterr().out
        assert code == ebk.EXIT_UNREADABLE_INPUT
        assert "REFUSED (exit 6)" in out and "No report written." in out
        for n in needles:
            assert n in out, out
        assert "Traceback" not in out
        assert world.no_report_written() and world.det.CALLS == []

    def test_a_missing_pilot_sheet(self, world, capsys):
        world.annotate({0: ("120", "C major")})
        world.csv.unlink()
        self._assert_refused(world, capsys, world.run(),
                             "pilot annotation sheet not found", str(world.csv))

    def test_a_missing_manifest(self, world, capsys):
        world.annotate({0: ("120", "C major")})
        world.manifest.unlink()
        self._assert_refused(world, capsys, world.run(),
                             "annotation manifest not found", str(world.manifest))

    @pytest.mark.skipif(os.name != "posix" or os.geteuid() == 0,
                        reason="needs POSIX permissions and a non-root user")
    @pytest.mark.parametrize("which", ["csv", "manifest"])
    def test_an_unreadable_file(self, world, capsys, which):
        world.annotate({0: ("120", "C major")})
        target = getattr(world, which)
        target.chmod(0)
        try:
            self._assert_refused(world, capsys, world.run(),
                                 "permission denied", str(target))
        finally:
            target.chmod(0o644)

    def test_a_directory_in_place_of_the_sheet(self, world, capsys):
        code = ebk.main(["--real", "--pilot-csv", str(world.root),
                         "--manifest", str(world.manifest),
                         "--report-dir", str(world.reports), "--warmup", "0",
                         "--detector", f"{world.det_name}:constant"])
        self._assert_refused(world, capsys, code, "is a directory")

    def test_a_manifest_that_is_not_json(self, world, capsys):
        world.annotate({0: ("120", "C major")})
        world.manifest.write_text("{ not json")
        self._assert_refused(world, capsys, world.run(),
                             "annotation manifest is not valid JSON")

    def test_a_manifest_without_tracks(self, world, capsys):
        world.annotate({0: ("120", "C major")})
        world.manifest.write_text(json.dumps({"tracks": "nope"}))
        self._assert_refused(world, capsys, world.run(), "has no 'tracks' list")

    def test_a_sheet_that_is_not_text(self, world, capsys):
        world.csv.write_bytes(b"\xff\xfe\x00garbage")
        self._assert_refused(world, capsys, world.run(),
                             "pilot annotation sheet is not UTF-8")

    def test_missing_input_without_a_detector_is_also_refused(self, world, capsys):
        world.csv.unlink()
        self._assert_refused(world, capsys, world.run(detector=None),
                             "pilot annotation sheet not found")

    def test_the_validator_cli_reports_rather_than_crashes(self, world, capsys):
        world.manifest.unlink()
        assert vpa.main(["--csv", str(world.csv), "--manifest",
                         str(world.manifest)]) == 1
        out = capsys.readouterr().out
        assert "cannot read the sheet or manifest" in out and "Traceback" not in out

    def test_a_sheet_deleted_during_the_run_is_a_change(self, world, monkeypatch, capsys):
        world.annotate({0: ("120", "C major")})
        original = ebk.run

        def deleting_run(*a, **kw):
            out = original(*a, **kw)
            world.csv.unlink()
            return out

        monkeypatch.setattr(ebk, "run", deleting_run)
        assert world.run() == ebk.EXIT_SOURCES_CHANGED
        assert "now unreadable" in capsys.readouterr().out
        assert world.no_report_written()


# ----------------------------------------------------- timing telemetry --
def _timing_run(world, timings):
    world.det.TIMINGS = timings
    world.annotate({0: ("120", "C major"), 1: ("60", "A minor")})
    assert world.run(detector="with_timings") == ebk.EXIT_OK
    return world.report()


class TestTimingTelemetry:
    """Optional `timings` never affect scoring and are never invented."""

    @staticmethod
    def _scored_normally(report):
        r = report["results"]
        assert r["bpm"]["failed"] == 0 and r["key"]["failed"] == 0
        assert r["bpm"]["raw_accuracy"] == 0.5             # 120 right, 60 wrong
        assert r["bpm"]["octave_tolerant_accuracy"] == 1.0
        assert r["key"]["relation_breakdown"]["exact"] == 1
        assert r["key"]["relation_breakdown"]["relative"] == 1

    def test_absent_timings_are_simply_not_reported(self, world):
        rep = _timing_run(world, "absent")
        self._scored_normally(rep)
        t = rep["results"]["timing_ms"]
        assert not {"bpm_ms", "key_ms", "features_ms"} & set(t)
        assert rep["results"]["timing_issues"] == []
        assert not any("malformed" in s for s in rep["limitations"])

    def test_null_timings_are_treated_as_absent(self, world):
        rep = _timing_run(world, None)
        self._scored_normally(rep)
        assert rep["results"]["timing_issues"] == []

    def test_valid_timings_are_recorded_exactly(self, world):
        rep = _timing_run(world, {"bpm_ms": 12.5, "key_ms": 30, "features_ms": 7.25})
        self._scored_normally(rep)
        t = rep["results"]["timing_ms"]
        assert t["bpm_ms"] == {"n": 2, "p50": 12.5, "p95": 12.5, "p99": 12.5, "max": 12.5}
        assert t["key_ms"]["n"] == 2 and t["key_ms"]["max"] == 30.0
        assert t["features_ms"]["max"] == 7.25
        assert all("unavailable_calls" not in v for v in t.values())
        assert rep["results"]["timing_issues"] == []

    @pytest.mark.parametrize("timings", [
        [1, 2, 3], "fast", 12.5, True,
    ])
    def test_a_non_dict_marks_every_component_unavailable(self, world, timings):
        rep = _timing_run(world, timings)
        self._scored_normally(rep)
        t = rep["results"]["timing_ms"]
        for name in ("bpm_ms", "key_ms", "features_ms"):
            assert t[name] == {"n": 0, "unavailable_calls": 2}    # nothing invented
        issues = rep["results"]["timing_issues"]
        assert [i["track_id"] for i in issues] == [world.tid(0), world.tid(1)]
        assert all("expected a dict" in i["issue"] for i in issues)
        assert any("malformed in 2 case(s)" in s for s in rep["limitations"])

    @pytest.mark.parametrize("bad", ["x", float("nan"), float("inf"), -1.0, True, [5]])
    def test_a_malformed_component_is_dropped_and_the_rest_kept(self, world, bad):
        rep = _timing_run(world, {"bpm_ms": bad, "key_ms": 30.0})
        self._scored_normally(rep)
        t = rep["results"]["timing_ms"]
        assert t["bpm_ms"] == {"n": 0, "unavailable_calls": 2}
        assert t["key_ms"]["n"] == 2 and t["key_ms"]["max"] == 30.0
        assert "features_ms" not in t
        issues = rep["results"]["timing_issues"]
        assert len(issues) == 2 and all("timings.bpm_ms" in i["issue"] for i in issues)

    def test_numpy_timings_are_valid(self):
        items = ebk.synthetic_items()[:2]
        r = ebk.run(lambda s, sr: {"bpm": 120.0, "key": "C major",
                                   "timings": {"bpm_ms": np.float32(4.0)}},
                    items, warmup=0)
        assert r["timing_ms"]["bpm_ms"]["n"] == 2 and r["timing_issues"] == []

    def test_a_failing_answer_is_still_a_failure_whatever_its_timings(self):
        items = [i for i in ebk.synthetic_items() if i["id"] == "bpm_120"]
        r = ebk.run(lambda s, sr: {"bpm": None, "timings": {"bpm_ms": 3.0}},
                    items, warmup=0)
        assert r["bpm"]["failed"] == 1
        assert r["timing_ms"]["bpm_ms"]["n"] == 1           # valid telemetry kept


# ----------------------------------- post-review regressions (5 findings) --
class TestStructurallyInvalidInputs:
    """Finding 1: malformed content and unreadable audio are named refusals."""

    def _refused(self, world, capsys, code, expected, *needles):
        out = capsys.readouterr().out
        assert code == expected, out
        assert f"REFUSED (exit {expected})" in out and "No report written." in out
        for n in needles:
            assert n in out, out
        assert world.no_report_written() and world.det.CALLS == []

    def _write_tracks(self, world, tracks):
        world.manifest.write_text(json.dumps({"content_hash": "0" * 64,
                                              "tracks": tracks}))

    def test_a_track_without_track_id_is_refused_not_a_keyerror(self, world, capsys):
        world.annotate({0: ("120", "C major")})
        self._write_tracks(world, world.tracks + [{"path": "x.mp3"}])
        self._refused(world, capsys, world.run(), ebk.EXIT_UNREADABLE_INPUT,
                      "tracks[20] has no valid track_id", str(world.manifest))

    @pytest.mark.parametrize("entry,needle", [
        ("a string", "tracks[20] is str"), (7, "tracks[20] is int"),
        (None, "tracks[20] is NoneType"), ([1], "tracks[20] is list"),
    ])
    def test_a_track_entry_that_is_not_an_object(self, world, capsys, entry, needle):
        world.annotate({0: ("120", "C major")})
        self._write_tracks(world, world.tracks + [entry])
        self._refused(world, capsys, world.run(), ebk.EXIT_UNREADABLE_INPUT, needle)

    @pytest.mark.parametrize("tid", [None, 7, "", "   ", ["t00"]])
    def test_an_invalid_track_id(self, world, capsys, tid):
        world.annotate({0: ("120", "C major")})
        self._write_tracks(world, world.tracks + [{"track_id": tid}])
        self._refused(world, capsys, world.run(), ebk.EXIT_UNREADABLE_INPUT,
                      "has no valid track_id")

    @pytest.mark.parametrize("field", ["title", "duration_sec", "path", "sha256"])
    def test_a_pilot_track_missing_a_field_is_refused(self, world, capsys, field):
        world.annotate({0: ("120", "C major")})
        tracks = [dict(t) for t in world.tracks]
        del tracks[0][field]
        self._write_tracks(world, tracks)
        self._refused(world, capsys, world.run(), ebk.EXIT_UNREADABLE_INPUT,
                      "malformed")

    def test_a_sheet_row_with_missing_cells_is_refused(self, world, capsys):
        world.annotate({0: ("120", "C major")})
        lines = world.csv.read_text().splitlines()
        lines[2] = lines[2].split(",")[0]            # first data row: id only
        world.csv.write_text("\n".join(lines) + "\n")
        self._refused(world, capsys, world.run(), ebk.EXIT_UNREADABLE_INPUT,
                      "structurally malformed")

    def test_a_non_numeric_sheet_duration_is_an_ordinary_validation_error(
            self, world, capsys):
        world.annotate({0: ("120", "C major")})
        world.csv.write_text(world.csv.read_text().replace(",1.0,", ",abc,", 1))
        self._refused(world, capsys, world.run(), ebk.EXIT_INVALID_ANNOTATIONS,
                      "duration_sec was edited")

    @pytest.mark.parametrize("break_it", ["no_track_id", "non_object", "missing_field"])
    def test_the_no_detector_path_is_protected_too(self, world, capsys, break_it):
        tracks = [dict(t) for t in world.tracks]
        if break_it == "no_track_id":
            tracks.append({"path": "x"})
        elif break_it == "non_object":
            tracks.append("nope")
        else:
            del tracks[0]["duration_sec"]
        self._write_tracks(world, tracks)
        self._refused(world, capsys, world.run(detector=None),
                      ebk.EXIT_UNREADABLE_INPUT, "malformed")

    @pytest.mark.skipif(os.name != "posix" or os.geteuid() == 0,
                        reason="needs POSIX permissions and a non-root user")
    def test_unreadable_audio_is_an_audio_refusal(self, world, capsys):
        world.annotate({0: ("120", "C major")})
        audio = world.audio(world.tid(0))
        audio.chmod(0)
        try:
            self._refused(world, capsys, world.run(), ebk.EXIT_AUDIO_ERROR,
                          "cannot be read", str(audio))
        finally:
            audio.chmod(0o644)

    def test_a_directory_in_place_of_audio_is_an_audio_refusal(self, world, capsys):
        world.annotate({0: ("120", "C major")})
        audio = world.audio(world.tid(0))
        audio.unlink()
        audio.mkdir()
        self._refused(world, capsys, world.run(), ebk.EXIT_AUDIO_ERROR, "missing")


class TestKeyAndBpmAreJudgedIndependently:
    """Finding 2: a malformed key answer never costs a correct BPM answer."""

    MALFORMED_KEYS = [
        np.array(["C major", "A minor"]),          # multi-element: ambiguous
        np.zeros(24),                               # a key-probability vector
        np.array(["C major"]),                      # even one element: not coerced
        ["C major"], ("C major",), {"key": "C major"}, 7, 3.5, True, b"C major",
    ]

    @pytest.mark.parametrize("bad_key", MALFORMED_KEYS,
                             ids=lambda v: type(v).__name__ + str(np.size(v)))
    def test_a_malformed_key_fails_the_key_only(self, world, bad_key):
        world.det.ANSWER = {"bpm": 120.0, "key": bad_key}
        world.annotate({0: ("120", "C major"), 1: ("120", "A minor")})
        assert world.run(detector="answering") == ebk.EXIT_OK
        r = world.report()["results"]
        assert r["bpm"]["failed"] == 0 and r["bpm"]["evaluable"] == 2
        assert r["bpm"]["raw_accuracy"] == 1.0              # BPM untouched
        assert r["key"]["failed"] == 2 and r["key"]["evaluable"] == 0
        assert r["key"]["exact_accuracy"] == 0.0
        assert all("invalid key prediction" in f["error"] for f in r["key"]["failures"])

    @pytest.mark.parametrize("bad_bpm", [np.array([120.0, 60.0]), [120.0], "120",
                                         {"bpm": 120}, float("nan"), -5, True,
                                         10 ** 5000],
                             ids=lambda v: type(v).__name__)
    def test_a_malformed_bpm_fails_the_bpm_only(self, world, bad_bpm):
        world.det.ANSWER = {"bpm": bad_bpm, "key": "C major"}
        world.annotate({0: ("120", "C major")})
        assert world.run(detector="answering") == ebk.EXIT_OK
        r = world.report()["results"]
        assert r["bpm"]["failed"] == 1 and r["bpm"]["raw_accuracy"] == 0.0
        assert r["key"]["failed"] == 0 and r["key"]["exact_accuracy"] == 1.0

    def test_string_and_key_object_answers_are_still_accepted(self, world):
        from musicintel.analysis.keys import parse_key
        for answer in ("C major", np.str_("C major"), "c MAJOR", parse_key("C major")):
            assert ebk._key_prediction(answer) == ("C major", None), answer
        assert ebk._key_prediction("Db minor") == ("C# minor", None)   # normalised

    def test_an_answer_whose_comparison_raises_is_isolated(self):
        class Hostile(str):
            def __eq__(self, other):
                raise RuntimeError("no comparing")
            __hash__ = str.__hash__

        items = [i for i in ebk.synthetic_items() if i["id"] in ("bpm_120", "key_C_major")]
        r = ebk.run(lambda s, sr: {"bpm": 120.0, "key": Hostile("C major")},
                    items, warmup=0)
        assert r["bpm"]["failed"] == 0 and r["bpm"]["raw_accuracy"] == 1.0
        assert r["key"]["failed"] == 1
        assert "RuntimeError" in r["key"]["failures"][0]["error"]


class TestLabelExclusionCountsCoverEligibleRowsOnly:
    """Finding 3: manifest-excluded recordings never inflate label counts."""

    def test_counts_against_a_mixed_sheet(self, world):
        # 0 scorable; 1 no-tempo; 2 no-key; 3 both labels; 4, 5 carry exclusion
        # labels but their recordings are excluded whole; the rest pending.
        world.set_manifest(4, annotation_status=EXCLUDED, notes="test: duplicate")
        world.set_manifest(5, annotation_status=EXCLUDED, notes="test: damaged")
        world.set_manifest(9, annotation_status=EXCLUDED, notes="test: pending one")
        world.annotate({0: ("120", "C major"), 1: (NO_TEMPO, "C major"),
                        2: ("120", NO_KEY), 3: (NO_TEMPO, NO_KEY),
                        4: (NO_TEMPO, NO_KEY), 5: ("120", NO_KEY)})
        assert world.run() == ebk.EXIT_OK
        rep = world.report()
        d, r = rep["dataset"], rep["results"]

        assert d["annotated"] == 6 and d["pending"] == 14
        assert d["evaluated_tracks"] == [world.tid(i) for i in (0, 1, 2, 3)]
        assert d["no_stable_tempo"] == 2 == r["bpm"]["excluded_no_stable_tempo"]
        assert d["no_tonal_centre"] == 2 == r["key"]["excluded_no_tonal_centre"]
        assert r["bpm"]["excluded_tracks"] == [world.tid(1), world.tid(3)]
        assert r["key"]["excluded_tracks"] == [world.tid(2), world.tid(3)]
        # the whole-recording population is separate and complete
        assert d["manifest_excluded"] == 3
        assert [(e["track_id"], e["pilot_status"]) for e in d["manifest_excluded_tracks"]] \
            == [(world.tid(4), ANNOTATED), (world.tid(5), ANNOTATED),
                (world.tid(9), PENDING)]
        withdrawn = {world.tid(4), world.tid(5), world.tid(9)}
        assert withdrawn.isdisjoint(r["bpm"]["excluded_tracks"] + r["key"]["excluded_tracks"]
                                    + d["scored_tracks"] + d["exclusion_only_tracks"])
        assert r["bpm"]["denominator"] == 2 and r["key"]["denominator"] == 2

    def test_without_manifest_exclusions_the_counts_are_unchanged(self, world):
        world.annotate({0: ("120", "C major"), 1: (NO_TEMPO, NO_KEY)})
        assert world.run() == ebk.EXIT_OK
        d = world.report()["dataset"]
        assert d["no_stable_tempo"] == 1 and d["no_tonal_centre"] == 1


class TestTimingNeverAbortsARun:
    """Finding 4: no telemetry value, however extreme, can end the evaluation."""

    @pytest.mark.parametrize("bad", [10 ** 400, -(10 ** 400), 10 ** 5000],
                             ids=["1e400", "-1e400", "1e5000"])
    def test_an_integer_too_large_for_a_float(self, world, bad):
        rep = _timing_run(world, {"bpm_ms": bad, "key_ms": 30.0})
        TestTimingTelemetry._scored_normally(rep)
        t = rep["results"]["timing_ms"]
        assert t["bpm_ms"] == {"n": 0, "unavailable_calls": 2}     # nothing invented
        assert t["key_ms"]["n"] == 2 and t["key_ms"]["max"] == 30.0
        issues = rep["results"]["timing_issues"]
        assert [i["track_id"] for i in issues] == [world.tid(0), world.tid(1)]
        assert all("timings.bpm_ms" in i["issue"] and len(i["issue"]) < 200
                   for i in issues)
        assert any("malformed in 2 case(s)" in s for s in rep["limitations"])

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf"),
                                     np.float64("nan"), np.inf, "12", None.__class__,
                                     complex(1, 2), np.array([1.0, 2.0]), {"ms": 1}],
                             ids=lambda v: repr(v)[:20])
    def test_non_finite_and_wrong_typed_values(self, world, bad):
        items = [i for i in ebk.synthetic_items() if i["id"] in ("bpm_120", "key_C_major")]
        r = ebk.run(lambda s, sr: {"bpm": 120.0, "key": "C major",
                                   "timings": {"features_ms": bad, "bpm_ms": 2.0}},
                    items, warmup=0)
        assert r["bpm"]["raw_accuracy"] == 1.0 and r["key"]["exact_accuracy"] == 1.0
        assert r["timing_ms"]["features_ms"] == {"n": 0, "unavailable_calls": 2}
        assert r["timing_ms"]["bpm_ms"]["n"] == 2
        assert len(r["timing_issues"]) == 2

    def test_a_timings_object_that_raises_on_access(self):
        class Exploding(dict):
            def get(self, *a):
                raise RuntimeError("boom")

        items = [i for i in ebk.synthetic_items() if i["id"] in ("bpm_120", "key_C_major")]
        r = ebk.run(lambda s, sr: {"bpm": 120.0, "key": "C major",
                                   "timings": Exploding()}, items, warmup=0)
        assert r["bpm"]["raw_accuracy"] == 1.0 and r["key"]["exact_accuracy"] == 1.0
        assert len(r["timing_issues"]) == 2
        assert all("could not be read" in i["issue"] for i in r["timing_issues"])
        for name in ("bpm_ms", "key_ms", "features_ms"):
            assert r["timing_ms"][name] == {"n": 0, "unavailable_calls": 2}


class TestSourcesAreReadOnce:
    """Finding 5: one read of each source; integrity checks intact."""

    def _count_reads(self, world, monkeypatch):
        reads = {world.csv: 0, world.manifest: 0}
        real_bytes, real_text = Path.read_bytes, Path.read_text

        def counting_bytes(self):
            if self in reads:
                reads[self] += 1
            return real_bytes(self)

        def counting_text(self, *a, **kw):
            if self in reads:
                reads[self] += 1
            return real_text(self, *a, **kw)

        monkeypatch.setattr(Path, "read_bytes", counting_bytes)
        monkeypatch.setattr(Path, "read_text", counting_text)
        return reads

    def test_each_source_is_read_once_to_load_and_once_to_recheck(self, world,
                                                                  monkeypatch):
        world.annotate({0: ("120", "C major")})
        reads = self._count_reads(world, monkeypatch)
        assert world.run("--dry-run") == ebk.EXIT_OK
        # exactly: the initial snapshot + the final integrity re-hash
        assert reads == {world.csv: 2, world.manifest: 2}

    def test_a_refused_run_reads_each_source_exactly_once(self, world, monkeypatch):
        reads = self._count_reads(world, monkeypatch)
        assert world.run() == ebk.EXIT_NOTHING_TO_EVALUATE
        assert reads == {world.csv: 1, world.manifest: 1}

    def test_what_is_evaluated_is_what_was_hashed(self, world, monkeypatch):
        """Swap the sheet on disk right after the snapshot: the run evaluates
        the snapshot's labels, then the final comparison refuses it."""
        world.annotate({0: ("120", "C major")})
        original = ebk.read_sources
        seen = {}

        def swap_after_read(*a, **kw):
            sources = original(*a, **kw)
            world.annotate({5: ("90", "A minor"), 6: ("90", "A minor")})
            return sources

        real_run = ebk.run

        def recording_run(detector, items, **kw):
            seen["ids"] = [i["id"] for i in items]
            return real_run(detector, items, **kw)

        monkeypatch.setattr(ebk, "read_sources", swap_after_read)
        monkeypatch.setattr(ebk, "run", recording_run)
        assert world.run() == ebk.EXIT_SOURCES_CHANGED
        assert seen["ids"] == [world.tid(0)]          # the snapshot, not the swap
        assert world.no_report_written()

    def test_the_report_pins_the_snapshot_hashes(self, world):
        world.annotate({0: ("120", "C major")})
        expected = {p: hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in (world.csv, world.manifest)}
        assert world.run() == ebk.EXIT_OK
        d = world.report()["dataset"]
        assert d["pilot_csv_sha256"] == expected[world.csv]
        assert d["manifest_sha256"] == expected[world.manifest]

    def test_the_validator_parsed_and_path_entry_points_agree(self, world):
        world.annotate({0: ("120", "C major"), 1: ("fast", "Db minor")})
        by_path = vpa.validate(world.csv, world.manifest)
        by_data = vpa.validate_parsed(vpa.read_rows(world.csv),
                                      json.loads(world.manifest.read_text())["tracks"])
        assert by_path == by_data and by_path[0]


# ------------------------------------------------------------ boundary --
class TestBoundary:
    def test_the_harness_not_the_analysis_package_imports_the_decoder(self):
        assert "musicintel.recognition" in (REPO / "scripts/eval_bpm_key.py").read_text()
        for f in (REPO / "musicintel/analysis").glob("*.py"):
            assert "musicintel.recognition" not in f.read_text(), f
        for f in (REPO / "musicintel/recognition").glob("*.py"):
            assert "musicintel.analysis" not in f.read_text(), f
