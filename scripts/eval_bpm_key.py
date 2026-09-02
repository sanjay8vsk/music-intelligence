#!/usr/bin/env python
"""Stage 4 BPM/key evaluation harness. No detector is implemented yet.

Follows the manifest-driven pattern of scripts/eval_phase1*.py: fixed inputs,
content hashes, a provenance block, and a json+md report under eval/reports/.

A detector is supplied by dotted path and must expose:

    analyze(samples: np.ndarray, sample_rate: int) -> dict
        {"bpm": float | None, "key": str | None}

with per-component timings optionally in {"timings": {...}}. Nothing implements
that yet, which is the point: the harness exists so that when a detector lands
it is measured honestly from its first commit rather than demonstrated on a
convenient example.

    python scripts/eval_bpm_key.py --synthetic --dry-run
    python scripts/eval_bpm_key.py --synthetic --detector musicintel.analysis.tempo:analyze
"""

from __future__ import annotations

import argparse
import importlib
import json
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
    NO_KEY, NO_TEMPO, Prediction, evaluate_bpm, evaluate_key, format_confusion,
)
from musicintel.analysis.fixtures import synthetic_fixtures           # noqa: E402
from musicintel.eval.provenance import git_state, source_fingerprint  # noqa: E402

SCHEMA_VERSION = 1
ANALYSIS_SOURCES = (
    "musicintel/analysis/keys.py",
    "musicintel/analysis/fixtures.py",
    "musicintel/analysis/evaluation.py",
    "scripts/eval_bpm_key.py",
)


def load_detector(spec: str | None):
    """Import `module:function`. Returns None when no detector is configured."""
    if not spec:
        return None
    module_name, _, attr = spec.partition(":")
    if not attr:
        raise SystemExit(f"detector spec must be 'module:function', got {spec!r}")
    return getattr(importlib.import_module(module_name), attr)


def _percentiles(values: list[float]) -> dict:
    if not values:
        return {}
    a = np.asarray(values, dtype=float)
    return {"n": len(values),
            **{f"p{p}": round(float(np.percentile(a, p)), 3)
               for p in (50, 95, 99)},
            "max": round(float(a.max()), 3)}


def run(detector, items: list[dict], *, warmup: int = 3) -> dict:
    """Evaluate and time. `items` carry samples plus ground truth."""
    bpm_preds: list[Prediction] = []
    key_preds: list[Prediction] = []
    timings = {"total_ms": [], "bpm_ms": [], "key_ms": [], "features_ms": []}

    # Warm the detector so first-call import/JIT cost is not in the percentiles.
    for item in items[:warmup]:
        try:
            detector(item["samples"], item["sample_rate"])
        except Exception:
            break

    for item in items:
        tid = item["id"]
        started = time.perf_counter()
        try:
            result = detector(item["samples"], item["sample_rate"]) or {}
            elapsed = (time.perf_counter() - started) * 1000.0
            timings["total_ms"].append(elapsed)
            for k in ("bpm_ms", "key_ms", "features_ms"):
                v = (result.get("timings") or {}).get(k)
                if v is not None:
                    timings[k].append(float(v))
            bpm_pred, key_pred, err = result.get("bpm"), result.get("key"), None
        except Exception as exc:                      # a crash is a failure
            bpm_pred = key_pred = None
            err = f"{type(exc).__name__}: {exc}"

        if item.get("bpm_truth") is not None:
            bpm_preds.append(Prediction(tid, item["bpm_truth"], bpm_pred, err))
        if item.get("key_truth") is not None:
            key_preds.append(Prediction(tid, item["key_truth"], key_pred, err))

    return {
        "bpm": evaluate_bpm(bpm_preds) if bpm_preds else None,
        "key": evaluate_key(key_preds) if key_preds else None,
        "timing_ms": {k: _percentiles(v) for k, v in timings.items() if v},
    }


def synthetic_items(sample_rate: int | None = None) -> list[dict]:
    out = []
    for f in synthetic_fixtures(*( [sample_rate] if sample_rate else [] )):
        out.append({"id": f.fixture_id, "samples": f.render(),
                    "sample_rate": f.sample_rate,
                    "bpm_truth": f.bpm, "key_truth": f.key,
                    "tags": list(f.tags)})
    return out


def annotated_items(manifest_path: Path) -> tuple[list[dict], dict]:
    """Load ONLY rows a human has actually annotated. Never invents a label."""
    data = json.loads(manifest_path.read_text())
    rows = data.get("tracks", [])
    pending = [r for r in rows if r.get("annotation_status") != "annotated"]
    ready = [r for r in rows if r.get("annotation_status") == "annotated"]
    return ready, {"total": len(rows), "annotated": len(ready),
                   "pending": len(pending),
                   "content_hash": data.get("content_hash"),
                   "fixture_set": data.get("fixture_set")}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--detector", help="dotted path 'module:function'")
    ap.add_argument("--synthetic", action="store_true",
                    help="evaluate the deterministic synthetic fixture set")
    ap.add_argument("--manifest", type=Path,
                    default=REPO_ROOT / "eval/fixtures/bpm_key_annotation_manifest.json")
    ap.add_argument("--report-dir", type=Path, default=REPO_ROOT / "eval/reports")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    ready, manifest_stats = ([], {}) if not args.manifest.exists() else \
        annotated_items(args.manifest)

    print("Stage 4 BPM/key evaluation harness")
    print(f"  synthetic fixtures      : {len(synthetic_fixtures())}")
    if manifest_stats:
        print(f"  real-audio manifest     : {manifest_stats['total']} tracks, "
              f"{manifest_stats['annotated']} annotated, "
              f"{manifest_stats['pending']} pending")
    detector = load_detector(args.detector)
    print(f"  detector                : {args.detector or 'NONE CONFIGURED'}")

    if detector is None:
        print("\n  No detector supplied, so no accuracy can be reported.")
        print("  The harness is ready; the detector slot is deliberately empty.")
        return 0
    if manifest_stats and manifest_stats["annotated"] == 0 and not args.synthetic:
        print("\n  No annotated real-audio tracks. Refusing to report accuracy "
              "against an unlabelled manifest.")
        return 0

    items = synthetic_items() if args.synthetic else []
    if ready:
        print(f"  (real-audio evaluation would load {len(ready)} annotated tracks)")

    results = run(detector, items)
    report = {
        "schema_version": SCHEMA_VERSION,
        "stage": "4-bpm-key",
        "title": "BPM and key detection benchmark",
        "provenance": {
            "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "git": git_state(REPO_ROOT),
            "analysis_sources_sha256": source_fingerprint(REPO_ROOT, ANALYSIS_SOURCES),
            "detector": args.detector,
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "dataset": {"synthetic_fixtures": len(items), **manifest_stats},
        "results": results,
        "limitations": [
            "Synthetic fixtures are functional validation only. They CANNOT "
            "substantiate the >=90% BPM or >=75% key acceptance targets, which "
            "are claims about real music.",
            "Real-audio accuracy requires human annotation; unannotated rows are "
            "never evaluated and never guessed.",
            "Timing excludes audio decode, which is already counted in the "
            "Stage 3 latency budget.",
        ],
    }
    if args.dry_run:
        print("\n  dry run: no report written")
    else:
        args.report_dir.mkdir(parents=True, exist_ok=True)
        out = args.report_dir / "stage4_bpm_key_benchmark.json"
        out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(f"\n  written: {out}")

    if results["bpm"]:
        b = results["bpm"]
        print(f"\n  BPM  raw {b['raw_accuracy']}  octave-tolerant "
              f"{b['octave_tolerant_accuracy']}  2x {b['double_errors']}  "
              f"1/2x {b['half_errors']}  excluded {b['excluded_no_stable_tempo']}  "
              f"failed {b['failed']}")
    if results["key"]:
        k = results["key"]
        print(f"  KEY  exact {k['exact_accuracy']}  MIREX {k['mirex_weighted_score']}")
        print(f"       {k['relation_breakdown']}")
        print(format_confusion(k["confusion_matrix"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
