#!/usr/bin/env python
"""Phase 1E — definitive benchmark of the landmark recognizer.

Runs the complete Phase 1 pipeline (fingerprint -> index -> matcher -> decision)
over the same 1,854-query degradation corpus that produced the Phase 0 baseline,
and reports a condition-by-condition comparison against it.

The Phase 0 report is READ ONLY here. Output goes to
eval/reports/phase1e_benchmark.{json,md}.

Threshold discipline is unchanged from Phase 1D: catalog and held-out tracks are
each split in two by track id, the threshold is chosen on the CALIBRATION half,
and the headline numbers are reported on the EVALUATION half.

    python scripts/eval_phase1e.py
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from musicintel.eval.manifest import Manifest  # noqa: E402
from musicintel.eval.metrics import (  # noqa: E402
    QueryOutcome,
    by_condition_and_duration,
    group_by,
    summarize,
)
from musicintel.eval.provenance import (  # noqa: E402
    ALGORITHM_SOURCES,
    HARNESS_SOURCES,
    PHASE1_SOURCES,
    git_state,
    source_fingerprint,
)
from musicintel.recognition.decision import DEFAULT_DECISION_CONFIG  # noqa: E402
from musicintel.recognition.fingerprint import (  # noqa: E402
    FORMAT_VERSION,
    FingerprintConfig,
    fingerprint,
    load_audio,
)
from musicintel.recognition.index import INDEX_FORMAT_VERSION, build_index  # noqa: E402
from musicintel.recognition.matcher import MatchConfig, match  # noqa: E402

DEFAULT_MANIFEST = REPO_ROOT / "eval/fixtures/manifest.json"
DEFAULT_QUERY_INDEX = REPO_ROOT / "data/eval/queries/index.jsonl"
DEFAULT_REPORT_DIR = REPO_ROOT / "eval/reports"
PHASE0_REPORT = REPO_ROOT / "eval/reports/baseline.json"

MIN_ALIGNED = DEFAULT_DECISION_CONFIG.min_aligned_landmarks  # fixed during the sweep
SWEEP_POINTS = 120
# Sentinel swapped for a single-line JSON dump after serialization. The sweep is
# ~1,000 numbers; pretty-printing it costs thousands of lines and buys nothing,
# because nobody reads a sweep row by row.
_COMPACT = "@@COMPACT_SWEEP@@"


# ------------------------------------------------------------------ running --
def run_queries(records, index, fp_cfg, match_cfg, *, verbose=True):
    rows, n = [], len(records)
    for i, rec in enumerate(records, start=1):
        path = REPO_ROOT / rec["rendered_path"]
        row = {
            "query_id": rec["query_id"], "condition": rec["condition"],
            "family": rec["family"], "duration": rec["duration"],
            "position": rec["position"], "is_negative": rec["is_negative"],
            "truth_track_id": rec.get("track_id"),
            "source_track": rec.get("params", {}).get("source_track"),
            "synthetic": bool(rec.get("params", {}).get("synthetic")), "error": None,
        }
        try:
            t0 = time.perf_counter()
            y, sr = load_audio(path, fp_cfg)
            q = fingerprint(y, sr, fp_cfg)
            t_fp = time.perf_counter() - t0
            t1 = time.perf_counter()
            r = match(q, index, config=match_cfg)
            top = r.best
            t2 = time.perf_counter()
            aligned = top.score if top else 0
            evidence = (aligned / len(q)) if len(q) else 0.0
            t_dec = time.perf_counter() - t2
            row.update(
                query_landmarks=len(q), aligned=aligned, evidence=evidence,
                top_id=top.track_id if top else None,
                top3=[c.track_id for c in r.top(3)],
                best_offset=top.best_offset if top else None,
                concentration=top.concentration if top else 0.0,
                ms_fingerprint=t_fp * 1000, ms_lookup=r.timing.lookup * 1000,
                ms_histogram=r.timing.histogram * 1000,
                ms_rank=r.timing.ranking * 1000, ms_decision=t_dec * 1000,
            )
        except Exception as e:  # noqa: BLE001
            row.update(
                query_landmarks=0, aligned=0, evidence=0.0, top_id=None, top3=[],
                best_offset=None, concentration=0.0, ms_fingerprint=0.0,
                ms_lookup=0.0, ms_histogram=0.0, ms_rank=0.0, ms_decision=0.0,
                error=f"{type(e).__name__}: {e}",
            )
        rows.append(row)
        if verbose and i % 250 == 0:
            print(f"    {i}/{n}")
    return rows


def assign_splits(rows, catalog_ids, heldout_ids):
    """Split by TRACK so no recording informs both the threshold and the score."""
    import hashlib

    cal_cat = {t for i, t in enumerate(sorted(catalog_ids)) if i % 2 == 0}
    cal_held = {t for i, t in enumerate(sorted(heldout_ids)) if i % 2 == 0}
    for row in rows:
        if row["is_negative"]:
            if row["synthetic"]:
                h = int.from_bytes(
                    hashlib.sha256(row["query_id"].encode()).digest()[:2], "big"
                )
                side = "calibration" if h % 2 == 0 else "evaluation"
            else:
                side = "calibration" if row["source_track"] in cal_held else "evaluation"
        else:
            side = "calibration" if row["truth_track_id"] in cal_cat else "evaluation"
        row["split"] = side
    return rows


# ------------------------------------------------------------------ scoring --
def accepts(row, threshold):
    return row["evidence"] >= threshold and row["aligned"] >= MIN_ALIGNED


def confusion(rows, threshold):
    pos = [r for r in rows if not r["is_negative"]]
    neg = [r for r in rows if r["is_negative"]]
    tp = sum(1 for r in pos if accepts(r, threshold) and r["top_id"] == r["truth_track_id"])
    wrong = sum(1 for r in pos if accepts(r, threshold) and r["top_id"] != r["truth_track_id"])
    fp = sum(1 for r in neg if accepts(r, threshold))
    accepted = tp + wrong + fp
    return {
        "threshold": float(threshold), "positives": len(pos), "negatives": len(neg),
        "true_positives": tp, "wrong_accepts": wrong,
        "false_negatives": len(pos) - tp - wrong,
        "true_negatives": len(neg) - fp, "false_positives": fp,
        "recall_at_1": round(tp / len(pos), 4) if pos else None,
        "far": round(fp / len(neg), 6) if neg else None,
        "precision": round(tp / accepted, 4) if accepted else None,
        "correct_rejection_rate": round((len(neg) - fp) / len(neg), 4) if neg else None,
    }


def sweep_columnar(rows, n_points=SWEEP_POINTS):
    """Sweep stored as parallel arrays -- same information, a fraction of the lines."""
    scores = sorted({r["evidence"] for r in rows})
    if len(scores) > n_points:
        scores = [scores[i] for i in np.linspace(0, len(scores) - 1, n_points).astype(int)]
    pts = [confusion(rows, t) for t in scores]
    cols = ("threshold", "true_positives", "wrong_accepts", "false_negatives",
            "true_negatives", "false_positives", "recall_at_1", "far",
            "precision", "correct_rejection_rate")
    return {
        "note": "columnar; arrays are index-aligned, one entry per operating point",
        "points": len(pts), "columns": list(cols),
        **{c: [p[c] for p in pts] for c in cols},
    }


def pick_threshold(rows, far_target):
    scores = sorted({r["evidence"] for r in rows})
    best = None
    for t in scores:
        p = confusion(rows, t)
        if p["far"] is not None and p["far"] <= far_target:
            if best is None or (p["recall_at_1"], -p["threshold"]) > (
                best["recall_at_1"], -best["threshold"]
            ):
                best = p
    return best


def to_outcomes(rows, threshold):
    """Phase 0 QueryOutcome objects, so metrics come from the harness's own code."""
    return [
        QueryOutcome(
            query_id=r["query_id"], condition=r["condition"], family=r["family"],
            duration=r["duration"], position=r["position"],
            is_negative=r["is_negative"],
            latency_ms=(r["ms_fingerprint"] + r["ms_lookup"] + r["ms_histogram"]
                        + r["ms_rank"] + r["ms_decision"]),
            returned_ids=r["top3"] if accepts(r, threshold) else [],
            truth_track_id=r["truth_track_id"],
            top_distance=-r["evidence"], error=r["error"],
        )
        for r in rows
    ]


def pctl(v, p):
    return round(float(np.percentile(v, p)), 3) if v else None


# --------------------------------------------------------------- comparison --
def phase0_comparison(rows_all_outcomes):
    """Join Phase 1 per-condition results onto Phase 0's, reading the baseline.

    Every condition Phase 0 measured appears, improvement or regression alike.
    """
    p0 = json.loads(PHASE0_REPORT.read_text())
    p0_rows = {}
    for fam_rows in p0["tables"].values():
        for row in fam_rows:
            p0_rows[row["condition"]] = row
    p1_rows = {r["condition"]: r for r in by_condition_and_duration(rows_all_outcomes)}

    out = []
    for cond in sorted(set(p0_rows) | set(p1_rows)):
        a, b = p0_rows.get(cond), p1_rows.get(cond)
        if a is None or b is None:
            continue
        fam = b["family"]
        if fam == "negative":
            entry = {"condition": cond, "family": fam, "queries": b["queries"],
                     "phase0_far": a.get("far"), "phase1_far": b.get("far"),
                     "delta_far": None}
            if a.get("far") is not None and b.get("far") is not None:
                entry["delta_far"] = round(b["far"] - a["far"], 4)
        else:
            entry = {"condition": cond, "family": fam, "queries": b["queries"],
                     "phase0_recall_at_1": a.get("recall_at_1"),
                     "phase1_recall_at_1": b.get("recall_at_1"),
                     "phase0_recall_at_3": a.get("recall_at_3"),
                     "phase1_recall_at_3": b.get("recall_at_3")}
            if a.get("recall_at_1") is not None and b.get("recall_at_1") is not None:
                d = round(b["recall_at_1"] - a["recall_at_1"], 4)
                entry["delta_recall_at_1"] = d
                entry["verdict"] = ("improves" if d > 0.005 else
                                    "regresses" if d < -0.005 else "matches")
        out.append(entry)
    return out, p0


# ---------------------------------------------------------------------- main --
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    ap.add_argument("--query-index", default=str(DEFAULT_QUERY_INDEX))
    ap.add_argument("--report-dir", default=str(DEFAULT_REPORT_DIR))
    ap.add_argument("--holdout", type=int, default=12)
    ap.add_argument("--far-target", type=float, default=0.001)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args(argv)

    manifest = Manifest.load(Path(args.manifest))
    problems = manifest.verify(REPO_ROOT)
    if problems:
        print(f"ERROR: corpus incomplete ({len(problems)} problems)")
        return 2
    manifest.assign_holdout(args.holdout)
    catalog, heldout = manifest.catalog, manifest.held_out
    print(f"Corpus: {len(manifest)} tracks | catalog {len(catalog)} | held out {len(heldout)}")

    qi = Path(args.query_index)
    if not qi.exists():
        print(f"ERROR: no query index at {qi}")
        return 2
    records = [json.loads(x) for x in qi.read_text().splitlines()]
    missing = [r for r in records if not (REPO_ROOT / r["rendered_path"]).exists()]
    if missing:
        print(f"ERROR: {len(missing)}/{len(records)} rendered queries missing.")
        return 2
    if args.limit:
        records = records[: args.limit]
    print(f"Query set: {len(records)} (the Phase 0 corpus, verbatim)")

    fp_cfg, match_cfg = FingerprintConfig(), MatchConfig()

    print(f"Building index over {len(catalog)} catalog tracks...")
    t0 = time.perf_counter()
    items = [(t.track_id, fingerprint(*load_audio(REPO_ROOT / t.path, fp_cfg), fp_cfg))
             for t in catalog]
    index = build_index(items, config=fp_cfg)
    t_index = time.perf_counter() - t0
    print(f"  {len(index):,} postings, {index.nbytes/1e6:.1f} MB, {t_index:.1f}s")

    print("Running queries...")
    t1 = time.perf_counter()
    rows = run_queries(records, index, fp_cfg, match_cfg)
    t_run = time.perf_counter() - t1
    print(f"  {len(rows)} queries in {t_run:.1f}s")

    rows = assign_splits(rows, [t.track_id for t in catalog], [t.track_id for t in heldout])
    cal = [r for r in rows if r["split"] == "calibration"]
    ev = [r for r in rows if r["split"] == "evaluation"]

    print("Selecting threshold on the calibration split...")
    chosen = pick_threshold(cal, args.far_target)
    met_cal = chosen is not None
    if chosen is None:
        pts = [confusion(cal, t) for t in sorted({r["evidence"] for r in cal})]
        zero = [p for p in pts if p["far"] == 0.0]
        chosen = max(zero, key=lambda p: p["recall_at_1"]) if zero else pts[-1]
    threshold = chosen["threshold"]
    print(f"  threshold = {threshold:.6f}")

    eval_pt, all_pt = confusion(ev, threshold), confusion(rows, threshold)
    out_eval, out_all = to_outcomes(ev, threshold), to_outcomes(rows, threshold)
    comparison, p0 = phase0_comparison(out_all)

    ranking_only = {
        s: round(sum(1 for r in g if not r["is_negative"] and r["top_id"] == r["truth_track_id"])
                 / max(1, sum(1 for r in g if not r["is_negative"])), 4)
        for s, g in (("evaluation", ev), ("all_queries", rows))
    }
    fa_detail = [
        {"query_id": r["query_id"], "condition": r["condition"], "split": r["split"],
         "source_track": r["source_track"], "matched": r["top_id"],
         "evidence": round(r["evidence"], 6), "aligned": r["aligned"],
         "query_landmarks": r["query_landmarks"]}
        for r in rows if r["is_negative"] and accepts(r, threshold)
    ]

    ok = [r for r in rows if r["error"] is None]
    stages = {k: [r[f"ms_{k}"] for r in ok] for k in
              ("fingerprint", "lookup", "histogram", "rank", "decision")}
    stages["total"] = [sum(r[f"ms_{k}"] for k in
                           ("fingerprint", "lookup", "histogram", "rank", "decision"))
                       for r in ok]
    timing = {k: {"p50": pctl(v, 50), "p95": pctl(v, 95), "p99": pctl(v, 99),
                  "mean": round(float(np.mean(v)), 3)} for k, v in stages.items()}

    git = git_state(REPO_ROOT)
    phase1_paths = tuple(PHASE1_SOURCES) + ("scripts/eval_phase1e.py",)
    n_neg_ev = eval_pt["negatives"]

    results = {
        "schema_version": 1,
        "phase": "1E",
        "title": "Landmark fingerprint recognizer — definitive Phase 1 benchmark",
        "phase0_reference": "eval/reports/baseline.json (not modified by this run)",
        "provenance": {
            "git_commit": git["commit"],
            "git_dirty": git["dirty"],
            "git_dirty_path_count": len(git["dirty_paths"]),
            "git_dirty_paths": git["dirty_paths"],
            # The Phase 1D audit found a report that fingerprinted only the
            # harness. All three subsystems are pinned here.
            "phase1_sources": list(phase1_paths),
            "phase1_source_sha256": source_fingerprint(REPO_ROOT, phase1_paths),
            "harness_sha256": source_fingerprint(REPO_ROOT, HARNESS_SOURCES),
            "algorithm_sha256": source_fingerprint(REPO_ROOT, ALGORITHM_SOURCES),
            "fingerprint_format_version": FORMAT_VERSION,
            "index_format_version": INDEX_FORMAT_VERSION,
            "recognizer_version": f"landmark@{git['commit_short']}"
                                  + ("+dirty" if git["dirty"] else ""),
            "python": sys.version.split()[0],
            "executable": sys.executable,
            "platform": platform.platform(),
            "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        },
        "dataset": {
            "manifest_path": "eval/fixtures/manifest.json",
            "manifest_hash": manifest.content_hash(),
            "split_hash": manifest.split_hash(),
            "track_count": len(manifest), "catalog_count": len(catalog),
            "heldout_count": len(heldout), "queries": len(rows),
            "query_set": "Phase 0 corpus, reused verbatim",
        },
        "configuration": {
            "fingerprint": {k: getattr(fp_cfg, k) for k in (
                "sample_rate", "n_fft", "hop_length", "freq_min_hz", "freq_max_hz",
                "peak_neighborhood_freq_bins", "peak_neighborhood_time_frames",
                "threshold_percentile", "max_peaks_per_frame", "target_peak_density",
                "fan_out", "min_delta_frames", "max_delta_frames")},
            "matcher": {"offset_tolerance_frames": match_cfg.offset_tolerance_frames,
                        "max_candidates": match_cfg.max_candidates},
            "decision": {
                "rule": "MATCH iff evidence_score >= threshold and aligned >= min_aligned",
                "evidence_score": "aligned_landmarks / query_landmarks",
                "score_is_probability": False,
                "threshold": threshold, "min_aligned_landmarks": MIN_ALIGNED,
            },
            "index": {"tracks": index.n_tracks, "postings": len(index),
                      "unique_hashes": index.n_unique_hashes,
                      "bytes": index.nbytes, "build_seconds": round(t_index, 2)},
        },
        "splits": {
            "policy": "by track id, interleaved; no track appears on both sides",
            "calibration": {"queries": len(cal),
                            "positives": sum(1 for r in cal if not r["is_negative"]),
                            "negatives": sum(1 for r in cal if r["is_negative"])},
            "evaluation": {"queries": len(ev),
                           "positives": sum(1 for r in ev if not r["is_negative"]),
                           "negatives": sum(1 for r in ev if r["is_negative"])},
        },
        "threshold_selection": {
            "far_target": args.far_target,
            "met_on_calibration": met_cal,
            "met_on_evaluation": bool(eval_pt["far"] is not None
                                      and eval_pt["far"] <= args.far_target),
            "selected_on": "calibration split only",
            "calibration_operating_point": chosen,
        },
        "results": {
            "evaluation_holdout": eval_pt,
            "calibration": chosen,
            "all_queries": all_pt,
            "recall_at_1_ranking_only": ranking_only,
            "false_accepts": fa_detail,
            "negative_resolution": {
                "evaluation_negatives": n_neg_ev,
                "one_false_accept_pct_points":
                    round(100.0 / n_neg_ev, 4) if n_neg_ev else None,
                "smallest_measurable_nonzero_far":
                    round(1.0 / n_neg_ev, 6) if n_neg_ev else None,
                "negatives_needed_for_0.1pct_far": 1000,
                "note": "3.17%-class figures are corpus-limited, not production FAR",
            },
        },
        "sweep_calibration": _COMPACT,
        "by_condition_all_queries": by_condition_and_duration(out_all),
        "by_condition_evaluation": by_condition_and_duration(out_eval),
        "by_family_all_queries": {k: summarize(v) for k, v in
                                  group_by(out_all, "family").items()},
        "by_position_all_queries": {k: summarize(v) for k, v in group_by(
            [o for o in out_all if not o.is_negative], "position").items()},
        "by_duration_all_queries": {k: summarize(v) for k, v in group_by(
            [o for o in out_all if not o.is_negative], "duration").items()},
        "negatives_by_category": {k: summarize(v) for k, v in group_by(
            [o for o in out_all if o.is_negative], "condition").items()},
        "phase0_vs_phase1": comparison,
        "phase0_headline": {
            "recall_at_1": p0["overall"]["recall_at_1"],
            "recall_at_3": p0["overall"]["recall_at_3"],
            "far": p0["overall"]["far"],
            "correct_rejection_rate": p0["overall"]["correct_rejection_rate"],
            "p50_ms": p0["overall"]["p50_ms"], "p95_ms": p0["overall"]["p95_ms"],
        },
        "timing_ms": timing,
        "limitations": [
            f"FAR is measured against {n_neg_ev} held-out negatives, so one false "
            f"accept moves it by {100.0/n_neg_ev if n_neg_ev else float('nan'):.4f} "
            f"percentage points and the smallest resolvable non-zero FAR is "
            f"{100.0/n_neg_ev if n_neg_ev else float('nan'):.4f}%. A 0.1% "
            f"target needs roughly 1,000 negatives and cannot be demonstrated here.",
            "Speed and pitch conditions fail completely: the fingerprint key packs "
            "exact integer frequency-bin indices, and resampling or pitch shifting "
            "moves every bin. This is a representation limitation, confirmed at the "
            "hash level, not a matcher or threshold problem.",
            "The catalog is 32 tracks. Recognition difficulty grows sharply with "
            "catalog size, so these numbers are an OPTIMISTIC upper bound.",
            "The all-queries column shares half its positives with the calibration "
            "split. It exists for like-for-like comparison with Phase 0 denominators; "
            "the evaluation-split column is the unbiased result.",
            "Degradations are synthetic; no real acoustic captures are included.",
            "The decision score is a rate, not a calibrated probability.",
        ],
    }

    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    text = json.dumps(results, indent=2)
    text = text.replace(
        f'"{_COMPACT}"', json.dumps(sweep_columnar(cal), separators=(",", ":"))
    )
    (report_dir / "phase1e_benchmark.json").write_text(text + "\n")
    results["sweep_calibration"] = sweep_columnar(cal)
    (report_dir / "phase1e_benchmark.md").write_text(build_markdown(results))

    print("\n" + "=" * 70)
    print(f"  threshold            : {threshold:.6f}")
    print(f"  EVAL  Recall@1       : {eval_pt['recall_at_1']:.4f}   (Phase 0: 0.2946)")
    print(f"  EVAL  FAR            : {eval_pt['far']:.4f} "
          f"({eval_pt['false_positives']}/{eval_pt['negatives']})   (Phase 0: 1.0)")
    print(f"  ALL   Recall@1       : {all_pt['recall_at_1']:.4f}")
    print(f"  ALL   FAR            : {all_pt['far']:.4f} "
          f"({all_pt['false_positives']}/{all_pt['negatives']})")
    print(f"  ranking-only R@1     : {ranking_only['all_queries']:.4f}")
    print("=" * 70)
    print(f"Wrote {report_dir/'phase1e_benchmark.json'}")
    print(f"Wrote {report_dir/'phase1e_benchmark.md'}")
    return 0


def _p(v, nd=2):
    return "—" if v is None else f"{v*100:.{nd}f}%"


def build_markdown(r: dict) -> str:
    L = []
    A = L.append
    pv, ds, cfg = r["provenance"], r["dataset"], r["configuration"]
    res, sel, sp = r["results"], r["threshold_selection"], r["splits"]
    p0 = r["phase0_headline"]
    ev, allq = res["evaluation_holdout"], res["all_queries"]

    A("# Phase 1E — Landmark Recognizer Benchmark\n")
    A(f"**Recognizer:** `{pv['recognizer_version']}` "
      f"(fingerprint format v{pv['fingerprint_format_version']}, "
      f"index format v{pv['index_format_version']})  ")
    A(f"**Generated:** {pv['generated_utc']}  ")
    A(f"**Repo commit:** `{pv['git_commit'][:12]}`  ")
    A(f"**Working tree:** {'**DIRTY** — ' + str(pv['git_dirty_path_count']) + ' uncommitted paths; the commit above does not contain the exact code that ran' if pv['git_dirty'] else 'clean'}  ")
    A(f"**Phase 1 pipeline fingerprint:** `{pv['phase1_source_sha256']}`\n")
    A("> The Phase 0 baseline (`eval/reports/baseline.md`, Recall@1 29.46%, FAR 100%)\n"
      "> is **not modified** by this run and remains the reference point.\n")

    A("## Provenance\n")
    A("| Field | Value |")
    A("|---|---|")
    A(f"| Git commit | `{pv['git_commit']}` |")
    A(f"| Git dirty | `{pv['git_dirty']}` ({pv['git_dirty_path_count']} paths) |")
    A(f"| **Phase 1 source fingerprint** | `{pv['phase1_source_sha256']}` |")
    A(f"| Harness fingerprint | `{pv['harness_sha256']}` |")
    A(f"| Phase 0 algorithm fingerprint | `{pv['algorithm_sha256']}` |")
    A(f"| Fingerprint format version | {pv['fingerprint_format_version']} |")
    A(f"| Index format version | {pv['index_format_version']} |")
    A(f"| Manifest hash | `{ds['manifest_hash']}` |")
    A(f"| Split hash | `{ds['split_hash']}` |")
    A(f"| Catalog / held out | {ds['catalog_count']} / {ds['heldout_count']} |")
    A(f"| Queries evaluated | {ds['queries']} |")
    A(f"| Python | {pv['python']} on {pv['platform']} |")
    A("\nFiles covered by the Phase 1 fingerprint:\n")
    for s in pv["phase1_sources"]:
        A(f"- `{s}`")
    A("")

    A("## Configuration\n")
    d = cfg["decision"]
    A("```")
    A(f"fingerprint : {cfg['fingerprint']['sample_rate']} Hz, n_fft "
      f"{cfg['fingerprint']['n_fft']}, hop {cfg['fingerprint']['hop_length']}, "
      f"band {cfg['fingerprint']['freq_min_hz']:.0f}-{cfg['fingerprint']['freq_max_hz']:.0f} Hz,")
    A(f"              {cfg['fingerprint']['target_peak_density']} peaks/s, "
      f"fan_out {cfg['fingerprint']['fan_out']}, dt "
      f"{cfg['fingerprint']['min_delta_frames']}-{cfg['fingerprint']['max_delta_frames']} frames")
    A(f"matcher     : offset tolerance {cfg['matcher']['offset_tolerance_frames']} frames")
    A(f"decision    : {d['evidence_score']}")
    A(f"              MATCH iff score >= {d['threshold']:.6f} and aligned >= "
      f"{d['min_aligned_landmarks']}")
    A(f"index       : {cfg['index']['tracks']} tracks, {cfg['index']['postings']:,} "
      f"postings, {cfg['index']['bytes']/1e6:.1f} MB, built in "
      f"{cfg['index']['build_seconds']}s")
    A("```")
    A("\nThe decision score is a **rate, not a probability**.\n")

    A("## Splits and threshold\n")
    A(f"- Policy: {sp['policy']}")
    A(f"- Calibration: {sp['calibration']['queries']} queries "
      f"({sp['calibration']['positives']} pos / {sp['calibration']['negatives']} neg) "
      f"— threshold selected here")
    A(f"- Evaluation: {sp['evaluation']['queries']} queries "
      f"({sp['evaluation']['positives']} pos / {sp['evaluation']['negatives']} neg) "
      f"— headline numbers reported here")
    A(f"- FAR target {sel['far_target']}: "
      f"{'met' if sel['met_on_calibration'] else 'not met'} on calibration, "
      f"**{'met' if sel['met_on_evaluation'] else 'NOT met'}** on held-out data\n")

    A("## Phase 0 vs Phase 1 — headline\n")
    A("| Metric | Phase 0 (MFCC/FAISS) | **Phase 1 held-out** | Phase 1 all queries |")
    A("|---|---:|---:|---:|")
    A(f"| Recall@1 | {_p(p0['recall_at_1'])} | **{_p(ev['recall_at_1'])}** | {_p(allq['recall_at_1'])} |")
    A(f"| FAR | {_p(p0['far'])} | **{_p(ev['far'])}** | {_p(allq['far'])} |")
    A(f"| Correct rejection | {_p(p0['correct_rejection_rate'])} | "
      f"**{_p(ev['correct_rejection_rate'])}** | {_p(allq['correct_rejection_rate'])} |")
    A(f"| Precision | — | **{_p(ev['precision'])}** | {_p(allq['precision'])} |")
    A(f"| p50 latency | {p0['p50_ms']:.2f} ms | **{r['timing_ms']['total']['p50']:.2f} ms** | |")
    A(f"| p95 latency | {p0['p95_ms']:.2f} ms | **{r['timing_ms']['total']['p95']:.2f} ms** | |")
    A("")
    A(f"Ranking-only Recall@1 (matcher top-1, before the decision layer): "
      f"**{_p(res['recall_at_1_ranking_only']['all_queries'])}** over all queries. "
      f"The gap to the accepted figure is the cost of rejection.\n")

    A("### By family — every family, improvement or regression\n")
    A("| Family | Phase 0 R@1 | Phase 1 R@1 | Δ | Verdict |")
    A("|---|---:|---:|---:|---|")
    fams = {}
    for c in r["phase0_vs_phase1"]:
        if c["family"] == "negative" or "delta_recall_at_1" not in c:
            continue
        f = fams.setdefault(c["family"], {"n": 0, "p0": 0.0, "p1": 0.0})
        f["n"] += c["queries"]
        f["p0"] += c["phase0_recall_at_1"] * c["queries"]
        f["p1"] += c["phase1_recall_at_1"] * c["queries"]
    for name, f in sorted(fams.items()):
        a, b = f["p0"] / f["n"], f["p1"] / f["n"]
        d = b - a
        v = "improves" if d > 0.005 else "regresses" if d < -0.005 else "matches"
        A(f"| {name} | {_p(a)} | {_p(b)} | {d*100:+.2f} pp | {'**' + v + '**' if v == 'regresses' else v} |")
    A("")

    A("## Per-condition results (all queries; Phase 0 denominators)\n")
    A("| Condition | n | P0 R@1 | P1 R@1 | Δ | P1 R@3 | Verdict |")
    A("|---|---:|---:|---:|---:|---:|---|")
    for c in r["phase0_vs_phase1"]:
        if c["family"] == "negative" or "delta_recall_at_1" not in c:
            continue
        A(f"| `{c['condition']}` | {c['queries']} | {_p(c['phase0_recall_at_1'])} | "
          f"{_p(c['phase1_recall_at_1'])} | {c['delta_recall_at_1']*100:+.2f} pp | "
          f"{_p(c['phase1_recall_at_3'])} | {c['verdict']} |")
    A("")

    A("### By position and duration (all queries)\n")
    A("| Slice | Queries | Recall@1 | Recall@3 | No-match |")
    A("|---|---:|---:|---:|---:|")
    for k, v in r["by_position_all_queries"].items():
        A(f"| position = {k} | {v['queries']} | {_p(v['recall_at_1'])} | "
          f"{_p(v['recall_at_3'])} | {_p(v['no_match_rate'])} |")
    for k, v in sorted(r["by_duration_all_queries"].items(), key=lambda kv: float(kv[0])):
        A(f"| duration = {float(k):g} s | {v['queries']} | {_p(v['recall_at_1'])} | "
          f"{_p(v['recall_at_3'])} | {_p(v['no_match_rate'])} |")
    A("")

    A("## Negatives and rejection\n")
    A("| Category | Queries | False accepts | FAR | Correct rejection |")
    A("|---|---:|---:|---:|---:|")
    for k, v in sorted(r["negatives_by_category"].items()):
        A(f"| `{k.replace('negative_','')}` | {v['queries']} | "
          f"{v.get('false_accepts', 0)} | {_p(v.get('far'))} | "
          f"{_p(v.get('correct_rejection_rate'))} |")
    A("")
    nr = res["negative_resolution"]
    A(f"**Resolution limit.** The held-out split has **{nr['evaluation_negatives']} "
      f"negatives**, so a single false accept is worth "
      f"**{nr['one_false_accept_pct_points']} percentage points** and the smallest "
      f"resolvable non-zero FAR is **{nr['smallest_measurable_nonzero_far']*100:.4f}%**. "
      f"Observing a 0.1% FAR at all needs on the order of "
      f"**{nr['negatives_needed_for_0.1pct_far']:,} negatives**; the corpus has 126 in "
      f"total. **No figure here should be read as a production FAR.** Expanding the "
      f"negative fixture set is a future requirement, deliberately not done inside "
      f"this run because it would change the corpus methodology mid-comparison.\n")
    if res["false_accepts"]:
        A("Every false accept, itemized:\n")
        A("| Query | Category | Split | Matched | Evidence | Aligned / landmarks |")
        A("|---|---|---|---|---:|---:|")
        for f in res["false_accepts"]:
            A(f"| `{f['query_id'][:40]}` | {f['condition'].replace('negative_','')} | "
              f"{f['split']} | `{(f['matched'] or '-')[:26]}` | {f['evidence']:.4f} | "
              f"{f['aligned']} / {f['query_landmarks']} |")
        A("")

    A("## Latency\n")
    A("| Stage | p50 ms | p95 ms | p99 ms | mean ms |")
    A("|---|---:|---:|---:|---:|")
    for k, v in r["timing_ms"].items():
        A(f"| {k} | {v['p50']:.2f} | {v['p95']:.2f} | {v['p99']:.2f} | {v['mean']:.2f} |")
    A(f"\nPhase 0 measured p50 {p0['p50_ms']:.2f} ms / p95 {p0['p95_ms']:.2f} ms, but its "
      f"latency excluded feature extraction on the reference side and was taken on a "
      f"differently loaded machine; treat the comparison as indicative only.\n")

    A("## Threshold sweep (calibration split)\n")
    sw = r["sweep_calibration"]
    A(f"Stored columnar in the JSON ({sw['points']} operating points, "
      f"index-aligned arrays). Selected rows:\n")
    A("| Threshold | Recall@1 | FAR | Precision | TP | FP |")
    A("|---:|---:|---:|---:|---:|---:|")
    step = max(1, sw["points"] // 14)
    for i in range(0, sw["points"], step):
        A(f"| {sw['threshold'][i]:.4f} | {_p(sw['recall_at_1'][i])} | "
          f"{_p(sw['far'][i])} | {_p(sw['precision'][i])} | "
          f"{sw['true_positives'][i]} | {sw['false_positives'][i]} |")
    A("")

    A("## Limitations\n")
    for lim in r["limitations"]:
        A(f"- {lim}")
    A("")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
