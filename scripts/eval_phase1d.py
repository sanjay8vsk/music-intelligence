#!/usr/bin/env python
"""Evaluate the Phase 1D landmark recognizer and calibrate its decision threshold.

Reuses the Phase 0 query set verbatim -- same manifest, same catalog/held-out
split, same rendered degradations, same metrics code -- so the numbers are
directly comparable to eval/reports/baseline.md. That report is READ ONLY here;
this writes eval/reports/phase1d_baseline.{json,md}.

Threshold discipline: catalog tracks and held-out tracks are each split in two
by track id. Thresholds are chosen on the CALIBRATION half and reported on the
EVALUATION half, so the headline numbers are not fitted on the data that
produced them.

    python scripts/eval_phase1d.py
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
    HARNESS_SOURCES,
    git_state,
    source_fingerprint,
)
from musicintel.recognition.decision import (  # noqa: E402
    DecisionConfig,
    decide,
)
from musicintel.recognition.fingerprint import (  # noqa: E402
    FORMAT_VERSION,
    FingerprintConfig,
    fingerprint,
    load_audio,
)
from musicintel.recognition.index import build_index  # noqa: E402
from musicintel.recognition.matcher import MatchConfig, match  # noqa: E402

DEFAULT_MANIFEST = REPO_ROOT / "eval/fixtures/manifest.json"
DEFAULT_QUERY_DIR = REPO_ROOT / "data/eval/queries"
DEFAULT_REPORT_DIR = REPO_ROOT / "eval/reports"
MIN_ALIGNED = 5  # held fixed so the sweep stays one-dimensional


# ------------------------------------------------------------------ running --
def run_all_queries(records, index, fp_cfg, match_cfg, *, verbose=True):
    """Fingerprint + match every query. Returns raw evidence rows (no decision)."""
    rows = []
    n = len(records)
    for i, rec in enumerate(records, start=1):
        path = REPO_ROOT / rec["rendered_path"]
        row = {
            "query_id": rec["query_id"],
            "condition": rec["condition"],
            "family": rec["family"],
            "duration": rec["duration"],
            "position": rec["position"],
            "is_negative": rec["is_negative"],
            "truth_track_id": rec.get("track_id"),
            "source_track": rec.get("params", {}).get("source_track"),
            "synthetic": bool(rec.get("params", {}).get("synthetic")),
            "error": None,
        }
        try:
            t0 = time.perf_counter()
            y, sr = load_audio(path, fp_cfg)
            q = fingerprint(y, sr, fp_cfg)
            t_fp = time.perf_counter() - t0
            r = match(q, index, config=match_cfg)
            top = r.best
            row.update(
                query_landmarks=len(q),
                aligned=top.score if top else 0,
                evidence=(top.score / len(q)) if (top and len(q)) else 0.0,
                raw_score=top.score if top else 0,
                top_id=top.track_id if top else None,
                top3=[c.track_id for c in r.top(3)],
                best_offset=top.best_offset if top else None,
                concentration=top.concentration if top else 0.0,
                runner_up=r.candidates[1].track_id if len(r.candidates) > 1 else None,
                runner_up_score=r.candidates[1].score if len(r.candidates) > 1 else 0,
                ms_fingerprint=t_fp * 1000,
                ms_lookup=r.timing.lookup * 1000,
                ms_histogram=r.timing.histogram * 1000,
                ms_rank=r.timing.ranking * 1000,
            )
        except Exception as e:  # noqa: BLE001
            row.update(
                query_landmarks=0, aligned=0, evidence=0.0, raw_score=0, top_id=None,
                top3=[], best_offset=None, concentration=0.0, runner_up=None,
                runner_up_score=0, ms_fingerprint=0.0, ms_lookup=0.0,
                ms_histogram=0.0, ms_rank=0.0, error=f"{type(e).__name__}: {e}",
            )
        rows.append(row)
        if verbose and i % 200 == 0:
            print(f"    {i}/{n} queries")
    return rows


# ------------------------------------------------------------------ splitting -
def assign_splits(rows, catalog_ids, heldout_ids):
    """Interleave tracks into calibration/evaluation halves.

    Split by TRACK, not by query: every degradation of one recording lands on
    the same side, so no audio informs both the threshold and the number the
    threshold is judged by. Interleaved by sorted id rather than cut in half so
    both sides get a comparable spread of the corpus.
    """
    cal_cat = {t for i, t in enumerate(sorted(catalog_ids)) if i % 2 == 0}
    cal_held = {t for i, t in enumerate(sorted(heldout_ids)) if i % 2 == 0}
    for row in rows:
        if row["is_negative"]:
            if row["synthetic"]:
                # 18 synthetic negatives: split deterministically by query id.
                side = "calibration" if hash_id(row["query_id"]) % 2 == 0 else "evaluation"
            else:
                side = "calibration" if row["source_track"] in cal_held else "evaluation"
        else:
            side = "calibration" if row["truth_track_id"] in cal_cat else "evaluation"
        row["split"] = side
    return rows


def hash_id(s: str) -> int:
    import hashlib

    return int.from_bytes(hashlib.sha256(s.encode()).digest()[:2], "big")


# ------------------------------------------------------------------- sweeping -
def confusion(rows, threshold, score_key="evidence", min_aligned=MIN_ALIGNED):
    """Counts at one operating point."""
    pos = [r for r in rows if not r["is_negative"]]
    neg = [r for r in rows if r["is_negative"]]
    accept = lambda r: r[score_key] >= threshold and r["aligned"] >= min_aligned  # noqa: E731

    tp = sum(1 for r in pos if accept(r) and r["top_id"] == r["truth_track_id"])
    wrong = sum(1 for r in pos if accept(r) and r["top_id"] != r["truth_track_id"])
    fn = len(pos) - tp - wrong
    fp = sum(1 for r in neg if accept(r))
    tn = len(neg) - fp
    accepted = tp + wrong + fp
    return {
        "threshold": float(threshold),
        "positives": len(pos),
        "negatives": len(neg),
        "true_positives": tp,
        "wrong_accepts": wrong,
        "false_negatives": fn,
        "true_negatives": tn,
        "false_positives": fp,
        "recall_at_1": round(tp / len(pos), 4) if pos else None,
        "far": round(fp / len(neg), 6) if neg else None,
        "precision": round(tp / accepted, 4) if accepted else None,
        "correct_rejection_rate": round(tn / len(neg), 4) if neg else None,
    }


def sweep(rows, score_key="evidence", n_points=400):
    """Exact sweep over the observed score values."""
    scores = sorted({r[score_key] for r in rows})
    if len(scores) > n_points:
        idx = np.linspace(0, len(scores) - 1, n_points).astype(int)
        scores = [scores[i] for i in idx]
    return [confusion(rows, t, score_key) for t in scores]


def pick_threshold(curve, far_target):
    """Highest-recall operating point whose FAR is within target."""
    ok = [p for p in curve if p["far"] is not None and p["far"] <= far_target]
    if not ok:
        return None
    return max(ok, key=lambda p: (p["recall_at_1"], -p["threshold"]))


# ------------------------------------------------------------------ outcomes --
def to_outcomes(rows, threshold, min_aligned=MIN_ALIGNED):
    """Convert evidence rows into Phase 0 QueryOutcome objects.

    Going through the harness's own dataclass means recall, FAR and the
    per-condition tables are computed by exactly the code that produced the
    Phase 0 numbers -- no reimplementation to disagree with.
    """
    out = []
    for r in rows:
        accepted = r["evidence"] >= threshold and r["aligned"] >= min_aligned
        returned = r["top3"] if accepted else []
        out.append(
            QueryOutcome(
                query_id=r["query_id"],
                condition=r["condition"],
                family=r["family"],
                duration=r["duration"],
                position=r["position"],
                is_negative=r["is_negative"],
                latency_ms=r["ms_fingerprint"] + r["ms_lookup"] + r["ms_histogram"]
                + r["ms_rank"],
                returned_ids=returned,
                truth_track_id=r["truth_track_id"],
                top_distance=-r["evidence"],  # higher evidence == lower "cost"
                error=r["error"],
            )
        )
    return out


def _num(v, nd: int = 4) -> str:
    """Console formatter that survives a None (e.g. a slice with no negatives)."""
    return "n/a" if v is None else f"{v:.{nd}f}"


def pctl(values, p):
    return float(np.percentile(values, p)) if values else None


# ---------------------------------------------------------------------- main --
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    ap.add_argument("--query-dir", default=str(DEFAULT_QUERY_DIR))
    ap.add_argument("--report-dir", default=str(DEFAULT_REPORT_DIR))
    ap.add_argument("--holdout", type=int, default=12)
    ap.add_argument("--far-target", type=float, default=0.001)
    ap.add_argument("--limit", type=int, default=0, help="cap queries, for smoke runs")
    args = ap.parse_args(argv)

    manifest_path = Path(args.manifest)
    if not manifest_path.is_absolute():
        manifest_path = (REPO_ROOT / manifest_path).resolve()
    manifest = Manifest.load(manifest_path)
    problems = manifest.verify(REPO_ROOT)
    if problems:
        print(f"ERROR: corpus incomplete ({len(problems)} problems)")
        return 2
    manifest.assign_holdout(args.holdout)
    catalog, heldout = manifest.catalog, manifest.held_out
    print(f"Corpus: {len(manifest)} tracks | catalog {len(catalog)} | held out {len(heldout)}")

    index_path = Path(args.query_dir).parent / "queries" / "index.jsonl"
    if not index_path.exists():
        print(f"ERROR: no query index at {index_path}. Run the Phase 0 benchmark first.")
        return 2
    records = [json.loads(line) for line in index_path.read_text().splitlines()]
    missing = [r for r in records if not (REPO_ROOT / r["rendered_path"]).exists()]
    if missing:
        print(f"ERROR: {len(missing)} of {len(records)} rendered queries are missing.")
        print("Re-run the Phase 0 benchmark to regenerate them (no --reuse-queries).")
        return 2
    if args.limit:
        records = records[: args.limit]
    print(f"Query set: {len(records)} (reused verbatim from the Phase 0 run)")

    fp_cfg, match_cfg = FingerprintConfig(), MatchConfig()

    print(f"Fingerprinting catalog ({len(catalog)} tracks)...")
    t0 = time.perf_counter()
    items = []
    for i, t in enumerate(catalog, start=1):
        items.append((t.track_id, fingerprint(*load_audio(REPO_ROOT / t.path, fp_cfg), fp_cfg)))
        if i % 8 == 0:
            print(f"    {i}/{len(catalog)}")
    index = build_index(items, config=fp_cfg)
    t_index = time.perf_counter() - t0
    print(f"  index: {len(index):,} postings, {index.nbytes/1e6:.1f} MB, {t_index:.1f}s")

    print("Running queries...")
    t1 = time.perf_counter()
    rows = run_all_queries(records, index, fp_cfg, match_cfg)
    t_queries = time.perf_counter() - t1
    print(f"  {len(rows)} queries in {t_queries:.1f}s")

    rows = assign_splits(rows, [t.track_id for t in catalog], [t.track_id for t in heldout])
    cal = [r for r in rows if r["split"] == "calibration"]
    ev = [r for r in rows if r["split"] == "evaluation"]
    print(f"  calibration: {len(cal)} ({sum(1 for r in cal if r['is_negative'])} neg) | "
          f"evaluation: {len(ev)} ({sum(1 for r in ev if r['is_negative'])} neg)")

    # -- sweep + threshold selection, on CALIBRATION only ------------------
    print("Sweeping thresholds on the calibration split...")
    cal_curve = sweep(cal, "evidence")
    raw_curve = sweep(cal, "raw_score")  # control: unnormalized, see report
    chosen = pick_threshold(cal_curve, args.far_target)
    far_target_met = chosen is not None
    if not far_target_met:
        # Report honestly, then fall back to the strictest measurable point:
        # the lowest threshold achieving zero observed false accepts.
        zero = [p for p in cal_curve if p["far"] == 0.0]
        chosen = max(zero, key=lambda p: p["recall_at_1"]) if zero else cal_curve[-1]
    threshold = chosen["threshold"]
    print(f"  selected threshold = {threshold:.6f} "
          f"(calibration FAR {_num(chosen['far'])}, Recall@1 {_num(chosen['recall_at_1'])})")

    # -- final numbers on the EVALUATION split ------------------------------
    eval_point = confusion(ev, threshold)
    accepted = lambda r: r["evidence"] >= threshold and r["aligned"] >= MIN_ALIGNED  # noqa: E731
    false_accepts_detail = [
        {"query_id": r["query_id"], "condition": r["condition"], "split": r["split"],
         "source_track": r["source_track"], "matched_track": r["top_id"],
         "evidence_score": round(r["evidence"], 6), "aligned_landmarks": r["aligned"],
         "query_landmarks": r["query_landmarks"], "concentration": round(r["concentration"], 4)}
        for r in rows if r["is_negative"] and accepted(r)
    ]
    # What a zero-false-accept threshold would cost, measured ON the evaluation
    # split. Reported for diagnosis only: choosing it would fit the threshold to
    # the data it is judged on, which is exactly what the split exists to prevent.
    posthoc = None
    full_point = confusion(rows, threshold)
    eval_curve = sweep(ev, "evidence")
    zero_fp = [p for p in eval_curve if p["false_positives"] == 0]
    if zero_fp:
        best_zero = max(zero_fp, key=lambda p: p["recall_at_1"])
        posthoc = {"note": "BIASED -- fitted on the evaluation split, shown for "
                           "diagnosis only, not an operating point",
                   "threshold": best_zero["threshold"],
                   "recall_at_1": best_zero["recall_at_1"],
                   "false_positives": 0, "negatives": best_zero["negatives"]}

    outcomes_eval = to_outcomes(ev, threshold)
    outcomes_all = to_outcomes(rows, threshold)
    ranked_recall = {
        "evaluation": round(
            sum(1 for r in ev if not r["is_negative"] and r["top_id"] == r["truth_track_id"])
            / max(1, sum(1 for r in ev if not r["is_negative"])), 4),
        "all": round(
            sum(1 for r in rows if not r["is_negative"] and r["top_id"] == r["truth_track_id"])
            / max(1, sum(1 for r in rows if not r["is_negative"])), 4),
    }

    # rule of three: with 0 observed events in n trials, the 95% upper bound on
    # the true rate is ~3/n. States what the negative set can and cannot show.
    n_neg_eval = eval_point["negatives"]
    far_upper95 = 3.0 / n_neg_eval if eval_point["false_positives"] == 0 else None

    lat = {
        "fingerprint_ms": [r["ms_fingerprint"] for r in rows if r["error"] is None],
        "lookup_ms": [r["ms_lookup"] for r in rows if r["error"] is None],
        "histogram_ms": [r["ms_histogram"] for r in rows if r["error"] is None],
        "rank_ms": [r["ms_rank"] for r in rows if r["error"] is None],
    }
    lat["total_ms"] = [
        a + b + c + d for a, b, c, d in zip(
            lat["fingerprint_ms"], lat["lookup_ms"], lat["histogram_ms"], lat["rank_ms"])
    ]
    timing = {k: {"p50": pctl(v, 50), "p95": pctl(v, 95), "mean": float(np.mean(v)) if v else None}
              for k, v in lat.items()}

    neg_breakdown = {}
    for cond, group in group_by(
        [o for o in outcomes_all if o.is_negative], "condition"
    ).items():
        s = summarize(group)
        neg_breakdown[cond] = {
            "queries": s["queries"], "false_accepts": s.get("false_accepts", 0),
            "far": s.get("far"), "correct_rejection_rate": s.get("correct_rejection_rate"),
        }

    git = git_state(REPO_ROOT)
    results = {
        "schema_version": 1,
        "phase": "1D",
        "supersedes": None,
        "phase0_reference": "eval/reports/baseline.json",
        "recognizer": {
            "name": "landmark_fingerprint",
            "fingerprint_format_version": FORMAT_VERSION,
            "fingerprint_config": {k: getattr(fp_cfg, k) for k in (
                "sample_rate", "n_fft", "hop_length", "freq_min_hz", "freq_max_hz",
                "target_peak_density", "fan_out", "min_delta_frames", "max_delta_frames")},
            "match_config": {"offset_tolerance_frames": match_cfg.offset_tolerance_frames},
            "decision": {
                "score": "aligned_landmarks / query_landmarks",
                "score_is_probability": False,
                "threshold": threshold,
                "min_aligned_landmarks": MIN_ALIGNED,
            },
            "index": {"tracks": index.n_tracks, "postings": len(index),
                      "bytes": index.nbytes, "build_seconds": round(t_index, 2)},
        },
        "environment": {
            "python": sys.version.split()[0],
            "executable": sys.executable,
            "platform": platform.platform(),
            "git_commit": git["commit"],
            "git_dirty": git["dirty"],
            "harness_sha256": source_fingerprint(REPO_ROOT, HARNESS_SOURCES),
            "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        },
        "dataset": {
            "manifest_path": "eval/fixtures/manifest.json",
            "manifest_hash": manifest.content_hash(),
            "split_hash": manifest.split_hash(),
            "catalog_count": len(catalog),
            "heldout_count": len(heldout),
            "query_set": "reused verbatim from the Phase 0 run",
            "queries": len(rows),
        },
        "splits": {
            "policy": "by track id, interleaved; no track appears in both",
            "calibration": {"queries": len(cal),
                            "positives": sum(1 for r in cal if not r["is_negative"]),
                            "negatives": sum(1 for r in cal if r["is_negative"])},
            "evaluation": {"queries": len(ev),
                           "positives": sum(1 for r in ev if not r["is_negative"]),
                           "negatives": sum(1 for r in ev if r["is_negative"])},
        },
        "threshold_selection": {
            "far_target": args.far_target,
            "far_target_met_on_calibration": far_target_met,
            "far_target_met_on_evaluation": bool(
                eval_point["far"] is not None and eval_point["far"] <= args.far_target),
            "selected_threshold": threshold,
            "selected_on": "calibration",
            "calibration_operating_point": chosen,
            "smallest_measurable_far_calibration": 1.0 / max(1, chosen["negatives"]),
        },
        "results": {
            "calibration": chosen,
            "evaluation": eval_point,
            "all_queries": full_point,
            "recall_at_1_ranking_only": ranked_recall,
            "far_95pct_upper_bound_evaluation": far_upper95,
            "smallest_measurable_far_evaluation": 1.0 / max(1, n_neg_eval),
            "false_accepts_detail": false_accepts_detail,
            "posthoc_zero_false_accept_point": posthoc,
        },
        "curves": {"calibration_evidence": cal_curve,
                   "calibration_raw_count": raw_curve,
                   "evaluation_evidence": eval_curve},
        "by_condition_evaluation": by_condition_and_duration(outcomes_eval),
        "by_family_evaluation": {k: summarize(v) for k, v in
                                 group_by(outcomes_eval, "family").items()},
        "by_duration_evaluation": {k: summarize(v) for k, v in group_by(
            [o for o in outcomes_eval if not o.is_negative], "duration").items()},
        "negatives_by_condition_all": neg_breakdown,
        "timing_ms": timing,
        "phase0_comparison": {
            "note": "Phase 0 numbers are quoted from eval/reports/baseline.json, "
                    "which this run does not modify.",
            "phase0_recall_at_1": 0.2946, "phase0_recall_at_3": 0.4485,
            "phase0_far": 1.0, "phase0_correct_rejection_rate": 0.0,
        },
    }

    report_dir = Path(args.report_dir)
    if not report_dir.is_absolute():
        report_dir = (REPO_ROOT / report_dir).resolve()
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "phase1d_baseline.json").write_text(json.dumps(results, indent=2) + "\n")
    (report_dir / "phase1d_baseline.md").write_text(build_markdown(results))

    print("\n" + "=" * 66)
    print(f"  threshold           : {threshold:.6f}  (FAR target {args.far_target} "
          f"{'MET' if far_target_met else 'NOT MET'} on calibration)")
    print(f"  EVAL Recall@1       : {_num(eval_point['recall_at_1'])}")
    print(f"  EVAL FAR            : {_num(eval_point['far'])} "
          f"({eval_point['false_positives']}/{eval_point['negatives']})")
    print(f"  EVAL correct reject : {_num(eval_point['correct_rejection_rate'])}")
    print(f"  EVAL precision      : {_num(eval_point['precision'])}")
    print(f"  ranking-only R@1    : {_num(ranked_recall['evaluation'])}")
    print(f"  Phase 0 was         : R@1 0.2946, FAR 1.0")
    print("=" * 66)
    print(f"Wrote {report_dir/'phase1d_baseline.json'}")
    print(f"Wrote {report_dir/'phase1d_baseline.md'}")
    return 0


def _pct(v):
    return "—" if v is None else f"{v*100:.2f}%"


def build_markdown(r: dict) -> str:
    L, A = [], None
    L = []
    A = L.append
    rec, res, sel = r["recognizer"], r["results"], r["threshold_selection"]
    env, ds, sp = r["environment"], r["dataset"], r["splits"]

    A("# Phase 1D — Match Decision Baseline\n")
    A(f"**Recognizer:** `{rec['name']}` (fingerprint format v{rec['fingerprint_format_version']})  ")
    A(f"**Generated:** {env['generated_utc']}  ")
    A(f"**Repo commit:** `{env['git_commit'][:12]}`"
      + ("  \n**Working tree: DIRTY** — the commit above does not contain the exact code that ran."
         if env["git_dirty"] else "  \n**Working tree:** clean"))
    A(f"  \n**Harness fingerprint:** `{env['harness_sha256'][:16]}…`\n")
    A("> This report does **not** replace `eval/reports/baseline.md`. The Phase 0\n"
      "> baseline (Recall@1 29.46%, FAR 100%) stands unchanged as the reference.\n")

    A("## Decision rule\n")
    A("```")
    A("evidence_score = aligned_landmarks / query_landmarks")
    A(f"MATCH  iff  evidence_score >= {sel['selected_threshold']:.6f}")
    A(f"       and  aligned_landmarks >= {rec['decision']['min_aligned_landmarks']}")
    A("```")
    A("\nThe score is a **rate, not a probability**. It is bounded by [0,1] because "
      "it is a fraction of landmarks, not because it is calibrated against outcome "
      "frequencies.\n")

    A("## Dataset and splits\n")
    A(f"- Manifest hash `{ds['manifest_hash'][:16]}…`, split hash `{ds['split_hash'][:16]}…`")
    A(f"- Catalog **{ds['catalog_count']}** tracks, held out **{ds['heldout_count']}**")
    A(f"- Queries **{ds['queries']}** — {ds['query_set']}")
    A(f"- Index: {rec['index']['tracks']} tracks, {rec['index']['postings']:,} postings, "
      f"{rec['index']['bytes']/1e6:.1f} MB, built in {rec['index']['build_seconds']}s")
    A(f"- Split policy: {sp['policy']}")
    A(f"- Calibration: {sp['calibration']['queries']} queries "
      f"({sp['calibration']['positives']} pos / {sp['calibration']['negatives']} neg)")
    A(f"- Evaluation: {sp['evaluation']['queries']} queries "
      f"({sp['evaluation']['positives']} pos / {sp['evaluation']['negatives']} neg)\n")

    A("## Threshold selection\n")
    A(f"- FAR target: **{sel['far_target']}** ({sel['far_target']*100:.1f}%)")
    A(f"  - on the calibration split: "
      f"{'**met**' if sel['far_target_met_on_calibration'] else '**not met**'}")
    A(f"  - on the held-out evaluation split: "
      f"{'**met**' if sel['far_target_met_on_evaluation'] else '**NOT MET**'} "
      f"— this is the number that counts")
    A(f"- Selected on the calibration split only: **{sel['selected_threshold']:.6f}**")
    A(f"- Smallest FAR the calibration negatives can resolve: "
      f"**{sel['smallest_measurable_far_calibration']*100:.2f}%** "
      f"({sel['calibration_operating_point']['negatives']} negatives)")
    A(f"- Smallest FAR the evaluation negatives can resolve: "
      f"**{res['smallest_measurable_far_evaluation']*100:.2f}%**\n")

    A("## Headline results (evaluation split — threshold NOT fitted on this data)\n")
    ev, cal, allq = res["evaluation"], res["calibration"], res["all_queries"]
    A("| Metric | Calibration | **Evaluation** | All queries | Phase 0 |")
    A("|---|---:|---:|---:|---:|")
    p0 = r["phase0_comparison"]
    A(f"| Recall@1 | {_pct(cal['recall_at_1'])} | **{_pct(ev['recall_at_1'])}** | "
      f"{_pct(allq['recall_at_1'])} | {_pct(p0['phase0_recall_at_1'])} |")
    A(f"| FAR | {_pct(cal['far'])} | **{_pct(ev['far'])}** | {_pct(allq['far'])} | "
      f"{_pct(p0['phase0_far'])} |")
    A(f"| Correct rejection | {_pct(cal['correct_rejection_rate'])} | "
      f"**{_pct(ev['correct_rejection_rate'])}** | {_pct(allq['correct_rejection_rate'])} | "
      f"{_pct(p0['phase0_correct_rejection_rate'])} |")
    A(f"| Precision | {_pct(cal['precision'])} | **{_pct(ev['precision'])}** | "
      f"{_pct(allq['precision'])} | — |")
    A(f"| True positives | {cal['true_positives']} | **{ev['true_positives']}** | "
      f"{allq['true_positives']} | — |")
    A(f"| Wrong accepts | {cal['wrong_accepts']} | **{ev['wrong_accepts']}** | "
      f"{allq['wrong_accepts']} | — |")
    A(f"| False negatives | {cal['false_negatives']} | **{ev['false_negatives']}** | "
      f"{allq['false_negatives']} | — |")
    A(f"| False accepts | {cal['false_positives']} | **{ev['false_positives']}** | "
      f"{allq['false_positives']} | — |")
    A("")
    A(f"Ranking-only Recall@1 (matcher top-1 before the decision layer): "
      f"**{_pct(res['recall_at_1_ranking_only']['evaluation'])}** on the evaluation split. "
      f"The gap to the post-decision figure is what rejection costs.\n")
    if res["far_95pct_upper_bound_evaluation"] is not None:
        A(f"Zero false accepts were observed on {ev['negatives']} evaluation negatives. "
          f"By the rule of three the 95% upper bound on the true FAR is "
          f"**{res['far_95pct_upper_bound_evaluation']*100:.2f}%** — the data cannot "
          f"support a stronger claim than that.\n")

    if r["results"].get("false_accepts_detail"):
        A("### Every false accept, itemized\n")
        A("| Query | Category | Split | Matched | Evidence | Aligned | Concentration |")
        A("|---|---|---|---|---:|---:|---:|")
        for f in r["results"]["false_accepts_detail"]:
            A(f"| `{f['query_id'][:44]}` | {f['condition'].replace('negative_','')} | "
              f"{f['split']} | `{(f['matched_track'] or '-')[:28]}` | "
              f"{f['evidence_score']:.4f} | {f['aligned_landmarks']} | {f['concentration']:.3f} |")
        A("")
    ph = r["results"].get("posthoc_zero_false_accept_point")
    if ph:
        A(f"> **Diagnosis only, not an operating point.** A threshold of "
          f"{ph['threshold']:.4f} would give 0/{ph['negatives']} false accepts at "
          f"Recall@1 {ph['recall_at_1']*100:.2f}% — but that threshold is read off "
          f"the evaluation split itself, so quoting it as a result would be exactly "
          f"the leakage the calibration split exists to prevent.\n")

    A("## Threshold trade-off (calibration split)\n")
    A("| Threshold | Recall@1 | FAR | Precision | Correct rejection | TP | FP |")
    A("|---:|---:|---:|---:|---:|---:|---:|")
    curve = r["curves"]["calibration_evidence"]
    step = max(1, len(curve) // 18)
    for p in curve[::step]:
        A(f"| {p['threshold']:.4f} | {_pct(p['recall_at_1'])} | {_pct(p['far'])} | "
          f"{_pct(p['precision'])} | {_pct(p['correct_rejection_rate'])} | "
          f"{p['true_positives']} | {p['false_positives']} |")
    A("")

    A("## Positive results by condition (evaluation split)\n")
    A("| Condition | Queries | Recall@1 | Recall@3 | No-match | p50 ms |")
    A("|---|---:|---:|---:|---:|---:|")
    for row in r["by_condition_evaluation"]:
        if row["family"] == "negative":
            continue
        A(f"| `{row['condition']}` | {row['queries']} | {_pct(row['recall_at_1'])} | "
          f"{_pct(row['recall_at_3'])} | {_pct(row['no_match_rate'])} | {row['p50_ms']:.1f} |")
    A("")

    A("### By family and duration (evaluation split)\n")
    A("| Slice | Queries | Recall@1 | Recall@3 |")
    A("|---|---:|---:|---:|")
    for k, v in sorted(r["by_family_evaluation"].items()):
        if v.get("recall_at_1") is None:
            continue
        A(f"| family = {k} | {v['queries']} | {_pct(v['recall_at_1'])} | {_pct(v['recall_at_3'])} |")
    for k, v in sorted(r["by_duration_evaluation"].items(), key=lambda kv: float(kv[0])):
        A(f"| duration = {k} s | {v['queries']} | {_pct(v['recall_at_1'])} | {_pct(v['recall_at_3'])} |")
    A("")

    A("## Negative results (all negatives, by category)\n")
    A("| Category | Queries | False accepts | FAR | Correct rejection |")
    A("|---|---:|---:|---:|---:|")
    for k, v in sorted(r["negatives_by_condition_all"].items()):
        A(f"| `{k}` | {v['queries']} | {v['false_accepts']} | {_pct(v['far'])} | "
          f"{_pct(v['correct_rejection_rate'])} |")
    A("")

    A("## Performance (all queries)\n")
    A("| Stage | p50 ms | p95 ms | mean ms |")
    A("|---|---:|---:|---:|")
    for k, v in r["timing_ms"].items():
        A(f"| {k.replace('_ms','')} | {v['p50']:.2f} | {v['p95']:.2f} | {v['mean']:.2f} |")
    A("")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
