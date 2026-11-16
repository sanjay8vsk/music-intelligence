#!/usr/bin/env python
"""Stage 4 BPM/key evaluation harness. No detector is implemented yet.

Follows the manifest-driven pattern of scripts/eval_phase1*.py: fixed inputs,
content hashes, a provenance block, and a json report under eval/reports/.

A detector is supplied by dotted path and must expose:

    analyze(samples: np.ndarray, sample_rate: int) -> dict
        {"bpm": float | None, "key": str | None}

with per-component timings optionally in {"timings": {...}}. Nothing implements
that yet, which is the point: the harness exists so that when a detector lands
it is measured honestly from its first commit rather than demonstrated on a
convenient example.

TWO EVALUATIONS, NEVER MIXED
----------------------------
Exactly one per run, each with its own report file and its own
`evaluation_kind`, so a synthetic score cannot be mistaken for a real one:

  --synthetic  the 38 deterministic fixtures. Ground truth is exact BY
               CONSTRUCTION: functional validation only, it cannot substantiate
               any accuracy target.        -> stage4_bpm_key_synthetic.json
  --real       real recordings labelled by a human listener in the validated
               pilot sheet.                -> stage4_bpm_key_real_audio.json

THE REAL-AUDIO PATH -- every step is a refusal point, none is skipped
--------------------------------------------------------------------
  1. Validate the pilot sheet against the manifest with
     scripts/validate_pilot_annotations.py. Any error: exit 1.
  2. Keep only rows whose status is `annotated`. A `pending` row is never
     decoded, never shown to the detector and never scored.
  3. If no annotated row carries a scorable label (a number, or one of the 24
     keys), refuse: exit 3. `no_stable_tempo` / `no_tonal_centre` are counted
     and reported as exclusions, but they are not something to score.
  4. Before the detector runs at all, check that every audio file to be scored
     exists and matches the manifest's sha256, so a label can only ever be
     judged against the recording it was made from. Any mismatch: exit 4.
  5. Decode with the project's file loader
     (musicintel.recognition.fingerprint.load_audio: librosa, mono) at the
     DECLARED analysis rate -- `ANALYSIS_SAMPLE_RATE` unless --sample-rate says
     otherwise -- and record that rate in the report. A decode failure is an
     environment problem, not a detector failure: exit 4.
  6. Run the detector. A crash, a missing answer, a decline or an answer that is
     not a valid BPM / 24-class key is a FAILURE, counted in the denominator as
     wrong.
  7. Re-hash the pilot sheet and the manifest. If either differs from what was
     read, exit 5. This harness only ever reads them; it never writes a
     prediction, a label or anything else back into an annotation source.

Before step 1, both sources must exist, be readable, and parse (UTF-8 sheet,
JSON manifest with a `tracks` list); otherwise exit 6.

A manifest track whose status is `excluded` is a whole-recording exclusion: it
is never decoded or scored, even if its pilot row is annotated, and the report
lists it with its reason. That is distinct from the per-quantity labels
`no_stable_tempo` / `no_tonal_centre`, which describe a recording that WAS
listened to.

A detector's optional `timings` are telemetry. A malformed value is dropped and
reported as unavailable -- never estimated -- and never affects the scoring of
that call's BPM/key answers.

In every refusal case no report is written.

    python scripts/eval_bpm_key.py --synthetic --dry-run
    python scripts/eval_bpm_key.py --synthetic --detector musicintel.analysis.tempo:analyze
    python scripts/eval_bpm_key.py --real      --detector musicintel.analysis.tempo:analyze
"""

from __future__ import annotations

import argparse
import dataclasses
import functools
import hashlib
import importlib
import importlib.util
import json
import math
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np                                                    # noqa: E402

from musicintel.analysis.evaluation import (                          # noqa: E402
    ANALYSIS_SAMPLE_RATE, ANNOTATED, EXCLUDED, NO_KEY, NO_TEMPO, PENDING,
    Prediction, evaluate_bpm, evaluate_key, format_confusion,
)
from musicintel.analysis.fixtures import click_track, synthetic_fixtures  # noqa: E402
from musicintel.analysis.keys import KeyParseError, normalize_key     # noqa: E402
from musicintel.eval.provenance import git_state, source_fingerprint  # noqa: E402
# The project's official file decoder. Imported here, in the harness, and never
# from musicintel/analysis/ -- the analysis package does not depend on
# recognition, and a test asserts that boundary.
from musicintel.recognition.fingerprint import (                      # noqa: E402
    FingerprintConfig, load_audio,
)

SCHEMA_VERSION = 2

DEFAULT_PILOT_CSV = REPO_ROOT / "eval/fixtures/bpm_key_pilot_annotation.csv"
DEFAULT_MANIFEST = REPO_ROOT / "eval/fixtures/bpm_key_annotation_manifest.json"
VALIDATOR_PATH = Path(__file__).resolve().with_name("validate_pilot_annotations.py")

SYNTHETIC = "synthetic"
REAL_AUDIO = "real_audio"
REPORT_FILENAMES = {
    SYNTHETIC: "stage4_bpm_key_synthetic.json",
    REAL_AUDIO: "stage4_bpm_key_real_audio.json",
}

DECODER = "musicintel.recognition.fingerprint.load_audio (librosa.load, mono=True)"

EXIT_OK = 0
EXIT_INVALID_ANNOTATIONS = 1
# 2 is argparse's usage-error code and is left to it.
EXIT_NOTHING_TO_EVALUATE = 3
EXIT_AUDIO_ERROR = 4
EXIT_SOURCES_CHANGED = 5
EXIT_UNREADABLE_INPUT = 6

WARMUP_SECONDS = 4.0
# Optional per-component telemetry a detector may report under "timings".
TIMING_COMPONENTS = ("bpm_ms", "key_ms", "features_ms")

ANALYSIS_SOURCES = (
    "musicintel/analysis/keys.py",
    "musicintel/analysis/fixtures.py",
    "musicintel/analysis/evaluation.py",
    "scripts/eval_bpm_key.py",
    "scripts/validate_pilot_annotations.py",
)


class HarnessRefusal(Exception):
    """The harness declines to evaluate. Carries the exit code and the reasons."""

    def __init__(self, code: int, message: str, details: list[str] | None = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details or []


class AudioError(HarnessRefusal):
    """Audio for an annotated row is missing, altered or undecodable."""

    def __init__(self, message: str):
        super().__init__(EXIT_AUDIO_ERROR, message)


# ---------------------------------------------------------------- helpers --
def load_detector(spec: str | None):
    """Import `module:function`. Returns None when no detector is configured."""
    if not spec:
        return None
    module_name, _, attr = spec.partition(":")
    if not attr:
        raise SystemExit(f"detector spec must be 'module:function', got {spec!r}")
    return getattr(importlib.import_module(module_name), attr)


def sha256_bytes(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _unreadable(what: str, path: Path, why: str) -> HarnessRefusal:
    return HarnessRefusal(EXIT_UNREADABLE_INPUT, f"{what} {why}: {path}")


def _read_source(path: Path, what: str) -> bytes:
    """Read an annotation source, turning an access error into a refusal."""
    path = Path(path)
    try:
        return path.read_bytes()
    except FileNotFoundError:
        raise _unreadable(what, path, "not found") from None
    except IsADirectoryError:
        raise _unreadable(what, path, "is a directory, not a file") from None
    except PermissionError:
        raise _unreadable(what, path, "is not readable (permission denied)") from None
    except OSError as exc:
        raise _unreadable(what, path, f"cannot be read ({exc.strerror or exc})") from None


def source_hashes(csv_path: Path, manifest_path: Path) -> dict[str, str]:
    """Check both annotation sources are readable and parseable; return their
    sha256. Raises HarnessRefusal(EXIT_UNREADABLE_INPUT) naming the file."""
    csv_bytes = _read_source(csv_path, "pilot annotation sheet")
    man_bytes = _read_source(manifest_path, "annotation manifest")
    try:
        csv_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise _unreadable("pilot annotation sheet", Path(csv_path),
                          f"is not UTF-8 text ({exc.reason} at byte {exc.start})") from None
    try:
        data = json.loads(man_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise _unreadable("annotation manifest", Path(manifest_path),
                          f"is not valid JSON ({exc})") from None
    if not isinstance(data, dict) or not isinstance(data.get("tracks"), list):
        raise _unreadable("annotation manifest", Path(manifest_path),
                          "has no 'tracks' list")
    return {"pilot_csv": hashlib.sha256(csv_bytes).hexdigest(),
            "manifest": hashlib.sha256(man_bytes).hexdigest()}


def _rehash(path: Path) -> str | None:
    """sha256 now, or None if the file has become unreadable (= changed)."""
    try:
        return sha256_bytes(path)
    except OSError:
        return None


def _rel(path: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _percentiles(values: list[float]) -> dict:
    if not values:
        return {}
    a = np.asarray(values, dtype=float)
    return {"n": len(values),
            **{f"p{p}": round(float(np.percentile(a, p)), 3)
               for p in (50, 95, 99)},
            "max": round(float(a.max()), 3)}


def _is_scorable_bpm(truth) -> bool:
    return isinstance(truth, (int, float)) and not isinstance(truth, bool)


def _is_scorable_key(truth) -> bool:
    return truth is not None and truth != NO_KEY


def _scorable(item: dict) -> bool:
    """Does this item carry at least one label a detector can be judged on?"""
    return (_is_scorable_bpm(item.get("bpm_truth"))
            or _is_scorable_key(item.get("key_truth")))


def _bpm_prediction(value) -> tuple[float | None, str | None]:
    """A detector's BPM answer, or (None, why it cannot be scored)."""
    if value is None:
        return None, "detector returned no BPM"
    if isinstance(value, str) and value == NO_TEMPO:
        return None, f"detector declined ({NO_TEMPO}); counted as wrong"
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer,
                                                          np.floating)):
        return None, f"invalid BPM prediction {value!r}"
    bpm = float(value)
    if not math.isfinite(bpm) or bpm <= 0:
        return None, f"invalid BPM prediction {value!r}"
    return bpm, None


def _key_prediction(value) -> tuple[str | None, str | None]:
    """A detector's key answer, normalised, or (None, why it cannot be scored)."""
    if value is None:
        return None, "detector returned no key"
    if value == NO_KEY:
        return None, f"detector declined ({NO_KEY}); counted as wrong"
    try:
        return normalize_key(str(value)), None
    except KeyParseError:
        return None, f"invalid key prediction {value!r}"


def _component_timings(raw) -> tuple[dict[str, float], list[str], list[str]]:
    """A detector's optional `timings`: (valid values, unavailable components,
    issues). Telemetry only -- it never affects how the answers are scored, and
    a malformed value is dropped and reported, never estimated or replaced."""
    if raw is None:                      # absent: simply not reported
        return {}, [], []
    if not isinstance(raw, dict):
        return {}, list(TIMING_COMPONENTS), [
            f"timings is {type(raw).__name__}, expected a dict; all component "
            "timings unavailable for this call"]
    valid: dict[str, float] = {}
    unavailable: list[str] = []
    issues: list[str] = []
    for name in TIMING_COMPONENTS:
        value = raw.get(name)
        if value is None:
            continue
        ok = (not isinstance(value, bool)
              and isinstance(value, (int, float, np.integer, np.floating))
              and math.isfinite(float(value)) and float(value) >= 0)
        if ok:
            valid[name] = float(value)
        else:
            unavailable.append(name)
            issues.append(f"timings.{name}={value!r} is not a finite, non-negative "
                          "number of milliseconds; unavailable for this call")
    return valid, unavailable, issues


# ------------------------------------------------------------- evaluation --
def _warm(detector, sample_rate: int, count: int) -> None:
    """Warm the detector on synthetic clicks so first-call import/JIT cost is not
    in the percentiles. Never on evaluation audio: nothing outside the measured
    loop is shown a real recording."""
    for i in range(count):
        try:
            detector(click_track(120.0, WARMUP_SECONDS, sr=sample_rate,
                                 fixture_id=f"warmup-{i}"), sample_rate)
        except Exception:
            break


def _samples_for(item: dict, timings: dict) -> tuple[np.ndarray, int]:
    if "samples" in item:
        return item["samples"], item["sample_rate"]
    started = time.perf_counter()
    samples, sr = item["load"]()           # AudioError propagates: a refusal
    timings["decode_ms"].append((time.perf_counter() - started) * 1000.0)
    return samples, sr


def run(detector, items: list[dict], *, warmup: int = 3) -> dict:
    """Evaluate and time. `items` carry samples (or a loader) plus ground truth.

    An item whose every label is an exclusion is counted but neither decoded nor
    shown to the detector: there is nothing to score, so nothing to compute.
    """
    bpm_preds: list[Prediction] = []
    key_preds: list[Prediction] = []
    timings = {"total_ms": [], "bpm_ms": [], "key_ms": [], "features_ms": [],
               "decode_ms": []}
    unavailable_calls = {name: 0 for name in TIMING_COMPONENTS}
    timing_issues: list[dict] = []
    scored: list[str] = []
    exclusion_only: list[str] = []

    if items and warmup > 0:
        _warm(detector, int(items[0].get("sample_rate", ANALYSIS_SAMPLE_RATE)),
              warmup)

    for item in items:
        tid = item["id"]
        bpm_truth, key_truth = item.get("bpm_truth"), item.get("key_truth")
        bpm_pred = key_pred = None
        bpm_err = key_err = None

        if not _scorable(item):
            exclusion_only.append(tid)
        else:
            samples, sr = _samples_for(item, timings)
            scored.append(tid)
            started = time.perf_counter()
            try:
                result = detector(samples, sr)
                timings["total_ms"].append((time.perf_counter() - started) * 1000.0)
                if not isinstance(result, dict):
                    raise TypeError(f"detector returned {type(result).__name__}, "
                                    "expected a dict")
                bpm_pred, bpm_err = _bpm_prediction(result.get("bpm"))
                key_pred, key_err = _key_prediction(result.get("key"))
            except Exception as exc:                  # a crash is a failure
                bpm_pred = key_pred = None
                bpm_err = key_err = f"{type(exc).__name__}: {exc}"
                result = None
            # Telemetry, deliberately outside the scoring path above: however
            # malformed, it cannot change whether an answer counts.
            if isinstance(result, dict):
                valid, unavailable, issues = _component_timings(result.get("timings"))
                for name, value in valid.items():
                    timings[name].append(value)
                for name in unavailable:
                    unavailable_calls[name] += 1
                timing_issues.extend({"track_id": tid, "issue": i} for i in issues)

        if bpm_truth is not None:
            bpm_preds.append(Prediction(tid, bpm_truth, bpm_pred, bpm_err))
        if key_truth is not None:
            key_preds.append(Prediction(tid, key_truth, key_pred, key_err))

    timing_ms = {k: _percentiles(v) for k, v in timings.items() if v}
    for name, n in unavailable_calls.items():
        if n:      # malformed telemetry is marked, not filled in
            timing_ms.setdefault(name, {"n": 0})["unavailable_calls"] = n

    return {
        "bpm": evaluate_bpm(bpm_preds) if bpm_preds else None,
        "key": evaluate_key(key_preds) if key_preds else None,
        "timing_ms": timing_ms,
        "timing_issues": timing_issues,
        "scored_tracks": scored,
        "exclusion_only_tracks": exclusion_only,
    }


# -------------------------------------------------------------- synthetic --
def synthetic_items(sample_rate: int | None = None) -> list[dict]:
    out = []
    for f in synthetic_fixtures(*([sample_rate] if sample_rate else [])):
        out.append({"id": f.fixture_id, "samples": f.render(),
                    "sample_rate": f.sample_rate,
                    "bpm_truth": f.bpm, "key_truth": f.key,
                    "tags": list(f.tags)})
    return out


# -------------------------------------------------------------- real audio --
def _load_validator():
    spec = importlib.util.spec_from_file_location("_pilot_validator", VALIDATOR_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def decode_audio(path: Path, sample_rate: int) -> tuple[np.ndarray, int]:
    """The project's file decode at the declared analysis rate.

    `load_audio` takes its rate from a FingerprintConfig and reads nothing else
    from it, so the config here carries only the analysis rate. No fingerprint
    is computed.
    """
    cfg = dataclasses.replace(FingerprintConfig(), sample_rate=int(sample_rate))
    try:
        samples, sr = load_audio(path, cfg)
    except Exception as exc:
        raise AudioError(f"cannot decode {path}: {type(exc).__name__}: {exc}") from exc
    if sr != sample_rate:
        raise AudioError(f"{path} decoded at {sr} Hz, declared {sample_rate} Hz")
    if samples.size == 0:
        raise AudioError(f"{path} decoded to no audio")
    return np.ascontiguousarray(samples, dtype=np.float32), sr


def _verify_audio(path: Path, expected_sha256: str) -> None:
    if not path.is_file():
        raise AudioError(f"audio for an annotated row is missing: {path}")
    actual = sha256_bytes(path)
    if actual != expected_sha256:
        raise AudioError(
            f"{path} does not match the manifest (sha256 {actual[:12]}..., "
            f"manifest {expected_sha256[:12]}...); the label was made from "
            "different audio")


def load_real_items(csv_path: Path, manifest_path: Path, *,
                    sample_rate: int = ANALYSIS_SAMPLE_RATE,
                    audio_root: Path = REPO_ROOT) -> tuple[list[dict], dict]:
    """Validated, human-annotated rows -> lazily decoded evaluation items.

    Raises HarnessRefusal (exit code attached) rather than returning anything
    that could be evaluated misleadingly. Reads the sheet and manifest only.
    """
    source_hashes(csv_path, manifest_path)          # readable + parseable, or exit 6
    validator = _load_validator()
    try:
        errors, summary = validator.validate(Path(csv_path), Path(manifest_path))
        rows = validator.read_rows(Path(csv_path))
        manifest = json.loads(Path(manifest_path).read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        # Became unreadable after the pre-check: still a refusal, not a traceback.
        raise HarnessRefusal(EXIT_UNREADABLE_INPUT,
                             f"annotation source became unreadable: "
                             f"{type(exc).__name__}: {exc}") from None
    if errors:
        raise HarnessRefusal(EXIT_INVALID_ANNOTATIONS,
                             "the pilot annotation sheet failed validation",
                             errors)

    by_id = {t["track_id"]: t for t in manifest["tracks"]}

    # Whole-recording exclusions recorded in the manifest. Listed by name with
    # their reason in every report; never decoded, never scored -- including
    # when the pilot row carries labels.
    manifest_excluded = [
        {"track_id": r["track_id"].strip(),
         "pilot_status": r["annotation_status"].strip(),
         "reason": str(by_id[r["track_id"].strip()].get("notes") or "").strip()}
        for r in rows
        if by_id[r["track_id"].strip()].get("annotation_status") == EXCLUDED]
    withdrawn = {e["track_id"] for e in manifest_excluded}

    annotated = [r for r in rows if r["annotation_status"].strip() == ANNOTATED]
    pending = [r["track_id"].strip() for r in rows
               if r["annotation_status"].strip() == PENDING]

    items: list[dict] = []
    for r in annotated:
        tid = r["track_id"].strip()
        if tid in withdrawn:
            continue
        bpm_raw, key_raw = r["bpm"].strip(), r["key"].strip()
        track = by_id[tid]          # the validator guarantees membership
        path = Path(audio_root) / track["path"]
        items.append({
            "id": tid,
            "path": track["path"],
            "sha256": track["sha256"],
            "sample_rate": int(sample_rate),
            # Labels exactly as the human wrote them, made typed. Validation
            # has already guaranteed both parse.
            "bpm_truth": NO_TEMPO if bpm_raw == NO_TEMPO else float(bpm_raw),
            "key_truth": NO_KEY if key_raw == NO_KEY else normalize_key(key_raw),
            "annotator": r["annotator"].strip(),
            "load": functools.partial(decode_audio, path, int(sample_rate)),
            "_audio_path": path,
        })

    stats = {
        "rows": summary.get("rows", len(rows)),
        "annotated": len(annotated),
        "pending": len(pending),
        "pending_tracks": pending,
        "annotated_tracks": [r["track_id"].strip() for r in annotated],
        "evaluated_tracks": [i["id"] for i in items],
        "manifest_excluded": len(manifest_excluded),
        "manifest_excluded_tracks": manifest_excluded,
        "no_stable_tempo": summary.get("no_stable_tempo", 0),
        "no_tonal_centre": summary.get("no_tonal_centre", 0),
        "annotators": summary.get("annotators", []),
        "manifest_content_hash": manifest.get("content_hash"),
        "fixture_set": manifest.get("fixture_set"),
    }

    if not any(_scorable(i) for i in items):
        raise HarnessRefusal(
            EXIT_NOTHING_TO_EVALUATE,
            f"no real-audio row is evaluable: {len(annotated)} annotated "
            f"(of {stats['rows']}; {len(withdrawn & set(stats['annotated_tracks']))} "
            "of those excluded in the manifest), none remaining with a scorable "
            "BPM or key label")

    # Pre-flight, before any detector work: every recording that will be scored
    # must be present and be the exact audio the manifest pins.
    for item in items:
        if _scorable(item):
            _verify_audio(item["_audio_path"], item["sha256"])
    return items, stats


# ----------------------------------------------------------------- report --
def _limitations(kind: str, results: dict, stats: dict | None) -> list[str]:
    if kind == SYNTHETIC:
        return [
            "SYNTHETIC FIXTURES: ground truth is exact by construction. This is "
            "functional validation only and CANNOT substantiate the >=90% BPM or "
            ">=75% key acceptance targets, which are claims about real music.",
            "Timing excludes audio decode, which is already counted in the "
            "Stage 3 latency budget.",
        ]
    out = [
        "Real-audio ground truth is human annotation from the validated pilot "
        "sheet. Pending rows are never evaluated and never guessed.",
        "Timing excludes audio decode; decode_ms is reported separately.",
    ]
    for name in ("bpm", "key"):
        r = results.get(name)
        if r and r["denominator"]:
            out.append(
                f"{name.upper()}: {r['denominator']} scored track(s) -- one error "
                f"moves accuracy by {100.0 / r['denominator']:.1f} percentage "
                "points. A pilot this size cannot substantiate the acceptance "
                "targets.")
    if stats and len(stats.get("annotators", [])) <= 1:
        out.append("Labels come from a single annotator; inter-annotator "
                   "agreement has not been measured.")
    if stats and stats.get("manifest_excluded"):
        out.append(f"{stats['manifest_excluded']} recording(s) are excluded "
                   "whole in the manifest and were not evaluated; see "
                   "dataset.manifest_excluded_tracks for each reason.")
    out.append("Audio is verified by sha256 and then decoded in a separate "
               "read; a file replaced between the two would not be detected. "
               "Known, low-risk, deliberately deferred.")
    return out


def _timing_limitation(results: dict) -> list[str]:
    issues = results.get("timing_issues") or []
    if not issues:
        return []
    return [f"Detector-reported component timings were malformed in "
            f"{len(issues)} case(s) (see results.timing_issues). Those values "
            "are reported as unavailable, not estimated; BPM/key scoring is "
            "unaffected."]


def build_report(kind: str, results: dict, *, detector: str, sample_rate: int,
                 dataset: dict, stats: dict | None = None) -> dict:
    synthetic = kind == SYNTHETIC
    return {
        "schema_version": SCHEMA_VERSION,
        "stage": "4-bpm-key",
        "evaluation_kind": kind,
        "title": ("BPM and key detection -- SYNTHETIC fixtures "
                  "(functional validation only)" if synthetic else
                  "BPM and key detection -- real audio, human-annotated"),
        "ground_truth": ("exact by construction (synthetic fixtures)" if synthetic
                         else "human annotation (validated pilot sheet)"),
        "provenance": {
            "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "git": git_state(REPO_ROOT),
            "analysis_sources_sha256": source_fingerprint(REPO_ROOT, ANALYSIS_SOURCES),
            "detector": detector,
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "analysis_sample_rate": int(sample_rate),
        "decoder": None if synthetic else DECODER,
        "dataset": dataset,
        "results": results,
        "limitations": _limitations(kind, results, stats) + _timing_limitation(results),
    }


def _write_report(report: dict, report_dir: Path, kind: str) -> Path:
    report_dir.mkdir(parents=True, exist_ok=True)
    out = report_dir / REPORT_FILENAMES[kind]
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return out


def _print_results(results: dict) -> None:
    if results.get("bpm"):
        b = results["bpm"]
        print(f"\n  BPM  raw {b['raw_accuracy']}  octave-tolerant "
              f"{b['octave_tolerant_accuracy']}  2x {b['double_errors']}  "
              f"1/2x {b['half_errors']}  excluded {b['excluded_no_stable_tempo']}  "
              f"failed {b['failed']}")
    if results.get("key"):
        k = results["key"]
        print(f"  KEY  exact {k['exact_accuracy']}  MIREX {k['mirex_weighted_score']}  "
              f"excluded {k['excluded_no_tonal_centre']}  failed {k['failed']}")
        print(f"       {k['relation_breakdown']}")
        print(format_confusion(k["confusion_matrix"]))
    for t in results.get("timing_issues") or []:
        print(f"  timing telemetry unavailable: {t['track_id']}: {t['issue']}")


def _refuse(r: HarnessRefusal) -> int:
    print(f"\n  REFUSED (exit {r.code}): {r.message}")
    for d in r.details:
        print(f"    - {d}")
    print("  No report written.")
    return r.code


# ------------------------------------------------------------------- main --
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--detector", help="dotted path 'module:function'")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--synthetic", action="store_true",
                      help="evaluate the deterministic synthetic fixture set")
    mode.add_argument("--real", action="store_true",
                      help="evaluate human-annotated real audio from the pilot sheet")
    ap.add_argument("--pilot-csv", type=Path, default=DEFAULT_PILOT_CSV)
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--audio-root", type=Path, default=REPO_ROOT,
                    help="directory the manifest's relative audio paths resolve against")
    ap.add_argument("--sample-rate", type=int, default=ANALYSIS_SAMPLE_RATE,
                    help=f"declared analysis decode rate (default {ANALYSIS_SAMPLE_RATE})")
    ap.add_argument("--warmup", type=int, default=3,
                    help="synthetic warm-up calls before timing")
    ap.add_argument("--report-dir", type=Path, default=REPO_ROOT / "eval/reports")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    if args.sample_rate <= 0:
        ap.error("--sample-rate must be positive")

    print("Stage 4 BPM/key evaluation harness")
    print(f"  synthetic fixtures      : {len(synthetic_fixtures())}")
    print(f"  pilot annotation sheet  : {_rel(args.pilot_csv)}")
    print(f"  analysis sample rate    : {args.sample_rate} Hz")

    detector = load_detector(args.detector)
    print(f"  detector                : {args.detector or 'NONE CONFIGURED'}")
    if detector is None:
        if args.real:
            try:
                source_hashes(args.pilot_csv, args.manifest)
            except HarnessRefusal as r:
                return _refuse(r)
            errors, summary = _load_validator().validate(args.pilot_csv, args.manifest)
            if errors:
                return _refuse(HarnessRefusal(
                    EXIT_INVALID_ANNOTATIONS,
                    "the pilot annotation sheet failed validation", errors))
            print(f"  pilot sheet             : {summary['annotated']} annotated, "
                  f"{summary['pending']} pending")
        print("\n  No detector supplied, so no accuracy can be reported.")
        print("  The harness is ready; the detector slot is deliberately empty.")
        return EXIT_OK
    if not (args.synthetic or args.real):
        ap.error("a detector needs exactly one mode: --synthetic or --real")

    if args.synthetic:
        kind = SYNTHETIC
        items = synthetic_items(args.sample_rate)
        results = run(detector, items, warmup=args.warmup)
        dataset = {"kind": SYNTHETIC, "synthetic_fixtures": len(items)}
        report = build_report(kind, results, detector=args.detector,
                              sample_rate=args.sample_rate, dataset=dataset)
    else:
        kind = REAL_AUDIO
        try:
            before = source_hashes(args.pilot_csv, args.manifest)
            items, stats = load_real_items(args.pilot_csv, args.manifest,
                                           sample_rate=args.sample_rate,
                                           audio_root=args.audio_root)
            print(f"  annotated / pending     : {stats['annotated']} / {stats['pending']}")
            results = run(detector, items, warmup=args.warmup)
        except HarnessRefusal as r:
            return _refuse(r)
        after = {"pilot_csv": _rehash(args.pilot_csv),
                 "manifest": _rehash(args.manifest)}
        if after != before:
            return _refuse(HarnessRefusal(
                EXIT_SOURCES_CHANGED,
                "an annotation source changed during evaluation",
                [f"{k}: {before[k][:12]}... -> "
                 f"{after[k][:12] + '...' if after[k] else 'now unreadable'}"
                 for k in before if before[k] != after[k]]))
        dataset = {
            "kind": REAL_AUDIO,
            "pilot_csv": _rel(args.pilot_csv),
            "pilot_csv_sha256": before["pilot_csv"],
            "manifest": _rel(args.manifest),
            "manifest_sha256": before["manifest"],
            "audio_verified_by": "sha256 against the manifest, before evaluation",
            **{k: v for k, v in stats.items()},
            "scored_tracks": results["scored_tracks"],
            "exclusion_only_tracks": results["exclusion_only_tracks"],
        }
        report = build_report(kind, results, detector=args.detector,
                              sample_rate=args.sample_rate, dataset=dataset,
                              stats=stats)

    if args.dry_run:
        print("\n  dry run: no report written")
    else:
        out = _write_report(report, args.report_dir, kind)
        print(f"\n  written: {out}")
    _print_results(results)
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
