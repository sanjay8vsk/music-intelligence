#!/usr/bin/env python
"""Phase 1H — speed-tolerant cascade benchmark.

Runs the frozen Phase 1 recognizer as stage 1 and a query-side playback-rate
sweep as stage 2, over the same corpus Phase 1G measured: 1,728 positives and
1,361 negatives (the original 126 plus the expanded Phase 1G set).

Nothing inside fingerprint/index/matcher/decision changes. Stage 1 keeps the
frozen Phase 1G threshold, so the stage-1-only column here is a control that
should reproduce Phase 1G. Stage 2 gets its OWN threshold, calibrated on the
calibration split alone.

Each query is swept once and every hypothesis is retained, so candidate stage-2
thresholds are replayed arithmetically rather than by re-running the recognizer.
The escalation set is fixed by the stage-1 threshold, which never moves, so the
replay is exact.

    python scripts/eval_phase1h.py
"""

from __future__ import annotations

import argparse
import hashlib
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
from musicintel.eval.metrics import QueryOutcome, by_condition_and_duration  # noqa: E402
from musicintel.eval.negatives import NegativeSet  # noqa: E402
from musicintel.eval.provenance import (  # noqa: E402
    ALGORITHM_SOURCES, HARNESS_SOURCES, PHASE1_SOURCES, git_state, source_fingerprint,
)
from musicintel.recognition.cascade import (  # noqa: E402
    DEFAULT_RATE_GRID, CascadeConfig, identify_cascade,
)
from musicintel.recognition.fingerprint import (  # noqa: E402
    FORMAT_VERSION, FingerprintConfig, fingerprint, load_audio,
)
from musicintel.recognition.index import INDEX_FORMAT_VERSION, build_index  # noqa: E402
from musicintel.recognition.matcher import MatchConfig  # noqa: E402

PHASE1G_REPORT = REPO_ROOT / "eval/reports/phase1g_benchmark.json"
STAGE1_THRESHOLD = 0.026316  # frozen Phase 1G operating point
MIN_ALIGNED = 5
_COMPACT = "@@COMPACT_SWEEP@@"


# ------------------------------------------------------------ replay logic --
def cascade_outcome(row, stage2_threshold):
    """Cascade verdict for a stored row at a candidate stage-2 threshold."""
    if row["stage1_accept"]:
        return True, 1, row["stage1_top"], 0.0
    best = row["best_hyp"]
    if best is None:
        return False, None, None, 0.0
    ok = best["evidence"] >= stage2_threshold and best["aligned"] >= MIN_ALIGNED
    return ok, (2 if ok else None), (best["track_id"] if ok else None), best["rate"]


def confusion(rows, stage2_threshold, *, stage1_only=False):
    tp = wrong = fp = 0
    pos = [r for r in rows if not r["is_negative"]]
    neg = [r for r in rows if r["is_negative"]]
    for r in pos:
        if stage1_only:
            acc, tid = r["stage1_accept"], r["stage1_top"]
        else:
            acc, _, tid, _ = cascade_outcome(r, stage2_threshold)
        if acc:
            if tid == r["truth_track_id"]:
                tp += 1
            else:
                wrong += 1
    for r in neg:
        acc = r["stage1_accept"] if stage1_only else cascade_outcome(r, stage2_threshold)[0]
        fp += bool(acc)
    accd = tp + wrong + fp
    return {
        "threshold": None if stage1_only else float(stage2_threshold),
        "positives": len(pos), "negatives": len(neg),
        "true_positives": tp, "wrong_accepts": wrong,
        "false_negatives": len(pos) - tp - wrong,
        "true_negatives": len(neg) - fp, "false_positives": fp,
        "recall_at_1": round(tp / len(pos), 4) if pos else None,
        "far": round(fp / len(neg), 6) if neg else None,
        "precision": round(tp / accd, 4) if accd else None,
        "correct_rejection_rate": round((len(neg) - fp) / len(neg), 4) if neg else None,
    }


def pick_stage2_threshold(cal_rows, far_target):
    """Highest-recall stage-2 threshold whose CASCADE FAR is within target.

    Judged on the whole cascade, not per hypothesis: ten extra attempts per
    query is ten extra chances to false-accept, and only the end-to-end rate
    reflects that.
    """
    cands = sorted({r["best_hyp"]["evidence"] for r in cal_rows
                    if not r["stage1_accept"] and r["best_hyp"]})
    cands = [c for c in cands if c > 0] or [1.0]
    best = None
    for t in cands:
        p = confusion(cal_rows, t)
        if p["far"] is not None and p["far"] <= far_target:
            if best is None or (p["recall_at_1"], -p["threshold"]) > (
                best["recall_at_1"], -best["threshold"]):
                best = p
    return best


def pctl(v, p):
    return round(float(np.percentile(v, p)), 3) if v else None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", default="eval/fixtures/manifest.json")
    ap.add_argument("--negatives", default="eval/fixtures/negatives_manifest.json")
    ap.add_argument("--query-index", default="data/eval/queries/index.jsonl")
    ap.add_argument("--report-dir", default="eval/reports")
    ap.add_argument("--holdout", type=int, default=12)
    ap.add_argument("--far-target", type=float, default=0.001)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args(argv)

    corpus = Manifest.load(REPO_ROOT / args.manifest)
    corpus.assign_holdout(args.holdout)
    catalog, heldout = corpus.catalog, corpus.held_out
    ns = NegativeSet.load(REPO_ROOT / args.negatives)
    fp_cfg, m_cfg = FingerprintConfig(), MatchConfig()
    cas_cfg = CascadeConfig(rate_grid=DEFAULT_RATE_GRID,
                            stage1_threshold=STAGE1_THRESHOLD,
                            stage2_threshold=0.0,  # replayed later; keeps all hypotheses
                            min_aligned_landmarks=MIN_ALIGNED)
    print(f"Corpus {len(corpus)} | catalog {len(catalog)} | held out {len(heldout)}")
    print(f"Rate grid: {list(DEFAULT_RATE_GRID)}  (stage 1 supplies rate 0)")

    t0 = time.perf_counter()
    index = build_index([(t.track_id, fingerprint(*load_audio(REPO_ROOT / t.path, fp_cfg), fp_cfg))
                         for t in catalog], config=fp_cfg)
    t_index = time.perf_counter() - t0
    print(f"Index: {len(index):,} postings in {t_index:.0f}s")

    cal_cat = {t for i, t in enumerate(sorted(x.track_id for x in catalog)) if i % 2 == 0}
    cal_held = {t for i, t in enumerate(sorted(x.track_id for x in heldout)) if i % 2 == 0}

    def legacy_split(rec):
        if not rec["is_negative"]:
            return "calibration" if rec.get("track_id") in cal_cat else "evaluation"
        st = rec.get("params", {}).get("source_track")
        if st is None:
            h = int.from_bytes(hashlib.sha256(rec["query_id"].encode()).digest()[:2], "big")
            return "calibration" if h % 2 == 0 else "evaluation"
        return "calibration" if st in cal_held else "evaluation"

    legacy = [json.loads(x) for x in (REPO_ROOT / args.query_index).read_text().splitlines()]
    queries = []
    for rec in legacy:
        queries.append({
            "query_id": rec["query_id"], "path": REPO_ROOT / rec["rendered_path"],
            "is_negative": rec["is_negative"], "truth_track_id": rec.get("track_id"),
            "condition": rec["condition"], "family": rec["family"],
            "duration": rec["duration"], "position": rec["position"],
            "category": rec["condition"] if rec["is_negative"] else None,
            "split": legacy_split(rec), "cohort": "phase1e"})
    for e in ns.excerpts:
        queries.append({
            "query_id": e.query_id, "path": REPO_ROOT / e.rendered_path,
            "is_negative": True, "truth_track_id": None, "condition": e.category,
            "family": "negative", "duration": e.duration, "position": "beginning",
            "category": e.category, "split": e.split, "cohort": "phase1g"})
    if args.limit:
        queries = queries[: args.limit]
    print(f"Queries: {len(queries)} "
          f"({sum(1 for q in queries if not q['is_negative'])} pos / "
          f"{sum(1 for q in queries if q['is_negative'])} neg)")

    print("Running cascade...")
    rows = []
    for i, q in enumerate(queries, 1):
        try:
            y, sr = load_audio(q["path"], fp_cfg)
            res = identify_cascade(y, sr, index, config=cas_cfg,
                                   match_config=m_cfg, fingerprint_config=fp_cfg)
            d1 = res.stage1_decision
            hyps = [{"rate": h.rate_percent, "evidence": h.evidence,
                     "track_id": h.track_id, "aligned": h.aligned,
                     "landmarks": h.query_landmarks, "error": h.error}
                    for h in res.hypotheses]
            usable = [h for h in hyps if h["error"] is None and h["track_id"]]
            best = min(usable, key=lambda h: (-h["evidence"], abs(h["rate"]), h["rate"])) if usable else None
            rows.append({**q,
                         "stage1_accept": d1.is_match,
                         "stage1_top": d1.track_id if d1.is_match else (
                             d1.candidates[0].track_id if d1.candidates else None),
                         "stage1_evidence": d1.evidence_score,
                         "escalated": res.escalated, "best_hyp": best,
                         "hypotheses": hyps,
                         "ms_stage1": res.timing.stage1_ms,
                         "ms_stage2": res.timing.stage2_ms,
                         "n_hyp": res.timing.hypotheses_evaluated, "error": None})
        except Exception as ex:  # noqa: BLE001
            rows.append({**q, "stage1_accept": False, "stage1_top": None,
                         "stage1_evidence": 0.0, "escalated": False, "best_hyp": None,
                         "hypotheses": [], "ms_stage1": 0.0, "ms_stage2": 0.0,
                         "n_hyp": 0, "error": f"{type(ex).__name__}: {ex}"})
        if i % 250 == 0:
            esc = sum(1 for r in rows if r["escalated"])
            print(f"    {i}/{len(queries)}  (escalated {esc})")

    cal = [r for r in rows if r["split"] == "calibration"]
    ev = [r for r in rows if r["split"] == "evaluation"]
    print(f"  calibration {len(cal)} | evaluation {len(ev)}")

    # -- stage-2 threshold, calibration split only --------------------------
    chosen = pick_stage2_threshold(cal, args.far_target)
    if chosen is None:
        chosen = confusion(cal, 1.0)
    s2 = chosen["threshold"]
    print(f"  stage-2 threshold = {s2:.6f} (stage 1 stays {STAGE1_THRESHOLD})")

    s1_ev = confusion(ev, s2, stage1_only=True)
    cas_ev = confusion(ev, s2)
    s1_all = confusion(rows, s2, stage1_only=True)
    cas_all = confusion(rows, s2)

    # -- per condition: stage-1 vs cascade ----------------------------------
    def outcomes(subset, stage1_only):
        out = []
        for r in subset:
            if stage1_only:
                acc, tid = r["stage1_accept"], r["stage1_top"]
            else:
                acc, _, tid, _ = cascade_outcome(r, s2)
            out.append(QueryOutcome(
                query_id=r["query_id"], condition=r["condition"], family=r["family"],
                duration=r["duration"], position=r["position"],
                is_negative=r["is_negative"],
                latency_ms=r["ms_stage1"] + r["ms_stage2"],
                returned_ids=[tid] if (acc and tid) else [],
                truth_track_id=r["truth_track_id"], error=r["error"]))
        return out

    pos_all = [r for r in rows if not r["is_negative"]]
    s1_rows = {x["condition"]: x for x in by_condition_and_duration(outcomes(pos_all, True))}
    cs_rows = {x["condition"]: x for x in by_condition_and_duration(outcomes(pos_all, False))}
    per_condition = []
    for cond in sorted(set(s1_rows) | set(cs_rows)):
        a, b = s1_rows.get(cond), cs_rows.get(cond)
        if not a or not b:
            continue
        per_condition.append({
            "condition": cond, "family": b["family"], "queries": b["queries"],
            "stage1_recall_at_1": a["recall_at_1"], "cascade_recall_at_1": b["recall_at_1"],
            "delta": round((b["recall_at_1"] or 0) - (a["recall_at_1"] or 0), 4)})

    fam = {}
    for c in per_condition:
        f = fam.setdefault(c["family"], {"n": 0, "s1": 0.0, "cs": 0.0})
        f["n"] += c["queries"]
        f["s1"] += (c["stage1_recall_at_1"] or 0) * c["queries"]
        f["cs"] += (c["cascade_recall_at_1"] or 0) * c["queries"]
    by_family = {k: {"queries": v["n"],
                     "stage1_recall_at_1": round(v["s1"] / v["n"], 4),
                     "cascade_recall_at_1": round(v["cs"] / v["n"], 4),
                     "delta": round((v["cs"] - v["s1"]) / v["n"], 4)}
                 for k, v in sorted(fam.items())}

    # -- cascade behaviour ---------------------------------------------------
    n = len(rows)
    esc = [r for r in rows if r["escalated"]]
    s2_acc = [r for r in rows if not r["stage1_accept"] and cascade_outcome(r, s2)[0]]
    winners = {}
    for r in s2_acc:
        k = f"{r['best_hyp']['rate']:+g}%"
        winners[k] = winners.get(k, 0) + 1
    behaviour = {
        "stage1_match_rate": round(sum(1 for r in rows if r["stage1_accept"]) / n, 4),
        "stage2_escalation_rate": round(len(esc) / n, 4),
        "stage2_acceptance_rate_of_escalated": round(len(s2_acc) / len(esc), 4) if esc else None,
        "stage2_acceptances": len(s2_acc),
        "winning_rate_histogram": dict(sorted(winners.items(),
                                              key=lambda kv: float(kv[0].rstrip('%')))),
    }

    # -- exploratory: is the full grid needed? (CALIBRATION ONLY) -----------
    cal_s2 = [r for r in cal if not r["stage1_accept"] and r["best_hyp"]]
    grids = {"full ±5% @1%": DEFAULT_RATE_GRID,
             "±5% @2% (-4,-2,2,4)": (-4.0, -2.0, 2.0, 4.0),
             "±5% coarse (-5,-2,2,5)": (-5.0, -2.0, 2.0, 5.0),
             "±2% only (-2,-1,1,2)": (-2.0, -1.0, 1.0, 2.0)}
    grid_study = {}
    for name, g in grids.items():
        allowed = set(g) | {0.0}
        tp = 0
        for r in cal:
            if r["stage1_accept"]:
                tp += int(r["stage1_top"] == r["truth_track_id"] and not r["is_negative"])
                continue
            us = [h for h in r["hypotheses"]
                  if h["error"] is None and h["track_id"] and h["rate"] in allowed]
            if not us:
                continue
            b = min(us, key=lambda h: (-h["evidence"], abs(h["rate"]), h["rate"]))
            if (b["evidence"] >= s2 and b["aligned"] >= MIN_ALIGNED
                    and not r["is_negative"] and b["track_id"] == r["truth_track_id"]):
                tp += 1
        npos = sum(1 for r in cal if not r["is_negative"])
        grid_study[name] = {"hypotheses_per_escalation": len(g),
                            "calibration_recall_at_1": round(tp / npos, 4)}

    # unimodality of the evidence-vs-rate profile: bears on whether a future
    # continuous estimator could replace the grid. Measured, not implemented.
    unimodal = 0
    profiled = 0
    for r in cal_s2:
        prof = sorted([(h["rate"], h["evidence"]) for h in r["hypotheses"]
                       if h["error"] is None], key=lambda x: x[0])
        vals = [v for _, v in prof]
        if max(vals) <= 0:
            continue
        profiled += 1
        peak = int(np.argmax(vals))
        if all(vals[i] <= vals[i + 1] + 1e-12 for i in range(peak)) and \
           all(vals[i] >= vals[i + 1] - 1e-12 for i in range(peak, len(vals) - 1)):
            unimodal += 1
    rate_profile = {"escalated_calibration_queries_with_signal": profiled,
                    "unimodal_profiles": unimodal,
                    "unimodal_fraction": round(unimodal / profiled, 4) if profiled else None,
                    "note": "A single-peaked evidence-vs-rate profile is what a "
                            "coarse-to-fine or continuous rate estimator would need. "
                            "Measured here only; nothing of the sort is implemented."}

    # -- timing --------------------------------------------------------------
    ok = [r for r in rows if r["error"] is None]
    t_s1 = [r["ms_stage1"] for r in ok]
    t_tot = [r["ms_stage1"] + r["ms_stage2"] for r in ok]
    t_s2 = [r["ms_stage2"] for r in ok if r["escalated"]]
    timing = {
        "stage1_ms": {"p50": pctl(t_s1, 50), "p95": pctl(t_s1, 95),
                      "mean": round(float(np.mean(t_s1)), 3)},
        "stage2_ms_when_escalated": {"p50": pctl(t_s2, 50), "p95": pctl(t_s2, 95),
                                     "mean": round(float(np.mean(t_s2)), 3) if t_s2 else None},
        "total_cascade_ms": {"p50": pctl(t_tot, 50), "p95": pctl(t_tot, 95),
                             "p99": pctl(t_tot, 99), "mean": round(float(np.mean(t_tot)), 3)},
    }
    # POST-HOC DIAGNOSTIC, not a re-scored criterion. Criterion 4 is judged on the
    # whole-corpus p50 above. This split exists because the corpus is 44% negatives
    # and every negative escalates by design, so the corpus mix -- not the cascade
    # alone -- drives the median. A production stream has a different mix.
    def _lat(sel):
        v = [r["ms_stage1"] + r["ms_stage2"] for r in ok if sel(r)]
        e = [r for r in rows if sel(r) and r["escalated"]]
        tot = [r for r in rows if sel(r)]
        return {"queries": len(tot), "escalation_rate": round(len(e)/len(tot), 4) if tot else None,
                "p50": pctl(v, 50), "p95": pctl(v, 95),
                "mean": round(float(np.mean(v)), 3) if v else None}
    timing["by_query_type_posthoc"] = {
        "positives": _lat(lambda r: not r["is_negative"]),
        "negatives": _lat(lambda r: r["is_negative"]),
    }

    # -- falsification criteria ---------------------------------------------
    p1g = json.loads(PHASE1G_REPORT.read_text())
    speed_conds = [c for c in per_condition if c["family"] == "speed"]
    speed_recall = (sum(c["cascade_recall_at_1"] * c["queries"] for c in speed_conds)
                    / sum(c["queries"] for c in speed_conds)) if speed_conds else 0.0
    preserved = {}
    for f in ("clean", "noise", "codec", "filter"):
        if f in by_family:
            preserved[f] = round(by_family[f]["cascade_recall_at_1"]
                                 - by_family[f]["stage1_recall_at_1"], 4)
    neg_ev = [r for r in ev if r["is_negative"]]
    criteria = {
        "1_speed_recall_ge_60pct": {"value": round(speed_recall, 4), "target": 0.60,
                                    "pass": bool(speed_recall >= 0.60)},
        "2_preserved_families_within_-1pp": {
            "deltas_vs_stage1": preserved, "target": -0.01,
            "pass": all(v >= -0.01 for v in preserved.values())},
        "3_holdout_far_le_5pct": {"value": cas_ev["far"], "target": 0.05,
                                  "pass": bool(cas_ev["far"] is not None and cas_ev["far"] <= 0.05)},
        "4_p50_latency_le_40ms": {"value": timing["total_cascade_ms"]["p50"], "target": 40.0,
                                  "pass": bool(timing["total_cascade_ms"]["p50"] <= 40.0)},
    }
    criteria["all_passed"] = all(v["pass"] for k, v in criteria.items() if k != "all_passed")

    def far_by(key, subset):
        out = {}
        for r in subset:
            out.setdefault(r[key] or "-", []).append(r)
        return {k: confusion(v, s2) for k, v in sorted(out.items())}

    git = git_state(REPO_ROOT)
    p1_paths = tuple(PHASE1_SOURCES) + ("musicintel/recognition/cascade.py",
                                        "scripts/eval_phase1h.py")
    src_ev = len({r["source_track"] for r in []}) or len(
        {e.source_track for e in ns.excerpts if e.split == "evaluation" and e.source_track})

    results = {
        "schema_version": 1, "phase": "1H",
        "title": "Speed-tolerant recognition cascade",
        "what_changed": "an orchestration layer only: stage-1 short-circuit plus a "
                        "query-side playback-rate sweep on rejection",
        "what_did_not_change": [
            "fingerprint.py, index.py, matcher.py, decision.py — byte-identical",
            "the 1,728 positive queries and the 1,361-negative corpus",
            "the stage-1 threshold (frozen Phase 1G operating point)",
        ],
        "provenance": {
            "git_commit": git["commit"], "git_dirty": git["dirty"],
            "git_dirty_path_count": len(git["dirty_paths"]),
            "phase1_sources": list(p1_paths),
            "phase1_source_sha256": source_fingerprint(REPO_ROOT, p1_paths),
            "harness_sha256": source_fingerprint(REPO_ROOT, HARNESS_SOURCES),
            "algorithm_sha256": source_fingerprint(REPO_ROOT, ALGORITHM_SOURCES),
            "fingerprint_format_version": FORMAT_VERSION,
            "index_format_version": INDEX_FORMAT_VERSION,
            "recognizer_version": f"landmark-cascade@{git['commit_short']}"
                                  + ("+dirty" if git["dirty"] else ""),
            "rate_grid_percent": list(DEFAULT_RATE_GRID),
            "rate_convention": "apply_rate(+p%) plays faster and higher; a recording "
                               "captured at +p% is corrected by about -p%",
            "stage1_threshold": STAGE1_THRESHOLD, "stage2_threshold": s2,
            "min_aligned_landmarks": MIN_ALIGNED,
            "benchmark_command": "python scripts/eval_phase1h.py",
            "python": sys.version.split()[0], "executable": sys.executable,
            "platform": platform.platform(),
            "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        },
        "dataset": {
            "manifest_hash": corpus.content_hash(), "split_hash": corpus.split_hash(),
            "negative_set_hash": ns.content_hash(),
            "catalog_count": len(catalog), "heldout_count": len(heldout),
            "positive_queries": sum(1 for q in queries if not q["is_negative"]),
            "negative_queries": sum(1 for q in queries if q["is_negative"]),
            "negative_sources": len(ns.sources),
            "evaluation_negatives": len(neg_ev),
            "evaluation_negative_source_recordings": src_ev,
        },
        "threshold_selection": {
            "stage1_threshold": STAGE1_THRESHOLD,
            "stage1_threshold_origin": "frozen Phase 1G operating point, not re-fitted",
            "stage2_threshold": s2,
            "stage2_selected_on": "calibration split only, judged on END-TO-END cascade FAR",
            "far_target": args.far_target,
            "calibration_operating_point": chosen,
        },
        "results": {
            "evaluation_stage1_only": s1_ev, "evaluation_cascade": cas_ev,
            "all_queries_stage1_only": s1_all, "all_queries_cascade": cas_all,
            "far_by_category_evaluation": far_by("category", neg_ev),
            "far_by_cohort_evaluation": far_by("cohort", neg_ev),
            "false_accepts": [
                {"query_id": r["query_id"], "category": r["category"],
                 "cohort": r["cohort"], "stage": cascade_outcome(r, s2)[1],
                 "rate": cascade_outcome(r, s2)[3],
                 "matched": cascade_outcome(r, s2)[2]}
                for r in rows if r["is_negative"] and cascade_outcome(r, s2)[0]],
        },
        "cascade_behaviour": behaviour,
        "by_family": by_family,
        "by_condition": per_condition,
        "falsification_criteria": criteria,
        "grid_study_calibration_only": {
            "note": "EXPLORATORY. The evaluation used the full +/-5% @1% grid, fixed "
                    "in advance. These alternatives are scored on the CALIBRATION "
                    "split only and were not used to choose anything.",
            "grids": grid_study},
        "rate_profile_study": rate_profile,
        "timing_ms": timing,
        "phase1g_reference": {
            "note": "Phase 1G is the stage-1 control. Same corpus, same threshold.",
            "phase1g_evaluation": p1g["results"]["evaluation_holdout"]},
        "limitations": [
            f"FAR rests on {len(neg_ev)} evaluation negatives from {src_ev} source "
            f"recordings. Excerpt count is not sample size: clips from one recording "
            f"fail together.",
            "Pitch is unaddressed by design; a rate sweep cannot invert a "
            "duration-preserving pitch shift, and pitch is not an acceptance criterion.",
            "Stage 2 multiplies false-accept exposure by the grid size, which is why it "
            "carries its own stricter threshold.",
            "The catalog is 32 tracks; difficulty grows with catalog size.",
        ],
        "sweep_calibration": _COMPACT,
    }

    sweep_pts = sorted({r["best_hyp"]["evidence"] for r in cal
                        if not r["stage1_accept"] and r["best_hyp"] and r["best_hyp"]["evidence"] > 0})
    if len(sweep_pts) > 120:
        sweep_pts = [sweep_pts[i] for i in np.linspace(0, len(sweep_pts) - 1, 120).astype(int)]
    pts = [confusion(cal, t) for t in sweep_pts]
    cols = ("threshold", "true_positives", "false_positives", "recall_at_1", "far",
            "precision", "correct_rejection_rate")
    sweep = {"note": "stage-2 threshold sweep on the calibration split, columnar",
             "points": len(pts), "columns": list(cols),
             **{c: [p[c] for p in pts] for c in cols}}

    rd = REPO_ROOT / args.report_dir
    rd.mkdir(parents=True, exist_ok=True)
    text = json.dumps(results, indent=2).replace(
        f'"{_COMPACT}"', json.dumps(sweep, separators=(",", ":")))
    (rd / "phase1h_benchmark.json").write_text(text + "\n")
    results["sweep_calibration"] = sweep
    (rd / "phase1h_benchmark.md").write_text(build_md(results))

    print("\n" + "=" * 72)
    print(f"  stage-2 threshold      : {s2:.6f}")
    print(f"  stage-1 match rate     : {behaviour['stage1_match_rate']*100:.2f}%")
    print(f"  escalation rate        : {behaviour['stage2_escalation_rate']*100:.2f}%")
    print(f"  EVAL Recall@1  stage1  : {s1_ev['recall_at_1']:.4f}")
    print(f"  EVAL Recall@1  cascade : {cas_ev['recall_at_1']:.4f}")
    _far = "n/a" if cas_ev["far"] is None else f"{cas_ev['far']:.6f}"
    print(f"  EVAL FAR       cascade : {_far} ({cas_ev['false_positives']}/{len(neg_ev)})")
    print(f"  p50 / p95 latency      : {timing['total_cascade_ms']['p50']:.1f} / "
          f"{timing['total_cascade_ms']['p95']:.1f} ms")
    print("  --- falsification criteria ---")
    for k, v in criteria.items():
        if k == "all_passed":
            continue
        print(f"    [{'PASS' if v['pass'] else 'FAIL'}] {k}: {v.get('value', v.get('deltas_vs_stage1'))}")
    print(f"  ALL CRITERIA: {'PASS' if criteria['all_passed'] else 'FAIL'}")
    print("=" * 72)
    return 0


def _p(v, nd=2):
    return "—" if v is None else f"{v*100:.{nd}f}%"


def build_md(r):
    L = []; A = L.append
    pv, ds, res = r["provenance"], r["dataset"], r["results"]
    beh, crit, tm = r["cascade_behaviour"], r["falsification_criteria"], r["timing_ms"]
    s1, cs = res["evaluation_stage1_only"], res["evaluation_cascade"]
    A("# Phase 1H — Speed-Tolerant Recognition Cascade\n")
    A(f"**Recognizer:** `{pv['recognizer_version']}`  ")
    A(f"**Generated:** {pv['generated_utc']}  ")
    A(f"**Repo commit:** `{pv['git_commit'][:12]}`  ")
    A(f"**Working tree:** {'DIRTY (' + str(pv['git_dirty_path_count']) + ' paths)' if pv['git_dirty'] else 'clean'}  ")
    A(f"**Phase 1 fingerprint:** `{pv['phase1_source_sha256']}`\n")
    A("> An orchestration layer only. `fingerprint.py`, `index.py`, `matcher.py` and\n"
      "> `decision.py` are byte-identical to Phase 1E/1G, and the Phase 0, 1D, 1E and\n"
      "> 1G reports are untouched.\n")

    A("## Verdict\n")
    A(f"**{'ALL FOUR CRITERIA PASS' if crit['all_passed'] else 'FALSIFICATION CRITERIA NOT ALL MET'}**\n")
    A("| # | Criterion | Target | Measured | Result |")
    A("|---|---|---:|---:|---|")
    c1, c2, c3, c4 = (crit["1_speed_recall_ge_60pct"], crit["2_preserved_families_within_-1pp"],
                      crit["3_holdout_far_le_5pct"], crit["4_p50_latency_le_40ms"])
    A(f"| 1 | Speed recall (4 conditions) | ≥ 60% | {_p(c1['value'])} | {'**PASS**' if c1['pass'] else '**FAIL**'} |")
    worst = min(c2["deltas_vs_stage1"].values()) if c2["deltas_vs_stage1"] else 0
    A(f"| 2 | Clean/noise/codec/filter vs stage 1 | ≥ −1 pp | worst {worst*100:+.2f} pp | {'**PASS**' if c2['pass'] else '**FAIL**'} |")
    A(f"| 3 | Held-out FAR | ≤ 5% | {_p(c3['value'],4)} | {'**PASS**' if c3['pass'] else '**FAIL**'} |")
    A(f"| 4 | p50 latency | ≤ 40 ms | {c4['value']:.2f} ms | {'**PASS**' if c4['pass'] else '**FAIL**'} |")
    A("")

    A("## Cascade configuration\n")
    A("```")
    A(f"rate grid   : {pv['rate_grid_percent']}   (stage 1 supplies rate 0)")
    A(f"convention  : {pv['rate_convention']}")
    A(f"stage 1     : threshold {pv['stage1_threshold']}  ({r['threshold_selection']['stage1_threshold_origin']})")
    A(f"stage 2     : threshold {pv['stage2_threshold']:.6f}  ({r['threshold_selection']['stage2_selected_on']})")
    A(f"min aligned : {pv['min_aligned_landmarks']}")
    A("```\n")

    A("## Headline (evaluation split)\n")
    A("| Metric | Stage 1 only | **Cascade** | Δ |")
    A("|---|---:|---:|---:|")
    for lab, k in (("Recall@1", "recall_at_1"), ("FAR", "far"),
                   ("Precision", "precision"), ("Correct rejection", "correct_rejection_rate")):
        d = (cs[k] or 0) - (s1[k] or 0)
        A(f"| {lab} | {_p(s1[k],4)} | **{_p(cs[k],4)}** | {d*100:+.2f} pp |")
    for lab, k in (("TP", "true_positives"), ("FP", "false_positives"),
                   ("TN", "true_negatives"), ("FN", "false_negatives")):
        A(f"| {lab} | {s1[k]} | **{cs[k]}** | {cs[k]-s1[k]:+d} |")
    A("")

    A("## Cascade behaviour\n")
    A(f"- Stage-1 match rate: **{_p(beh['stage1_match_rate'])}** of all queries")
    A(f"- Stage-2 escalation rate: **{_p(beh['stage2_escalation_rate'])}**")
    A(f"- Stage-2 acceptance rate among escalated: "
      f"**{_p(beh['stage2_acceptance_rate_of_escalated'])}** "
      f"({beh['stage2_acceptances']} acceptances)")
    A("\n**Which rate hypothesis won:**\n")
    A("| Correction | Stage-2 acceptances |")
    A("|---|---:|")
    for k, v in beh["winning_rate_histogram"].items():
        A(f"| {k} | {v} |")
    A("")

    A("## Per-family: stage 1 vs cascade\n")
    A("| Family | Queries | Stage 1 | Cascade | Δ |")
    A("|---|---:|---:|---:|---:|")
    for k, v in r["by_family"].items():
        A(f"| {k} | {v['queries']} | {_p(v['stage1_recall_at_1'])} | "
          f"**{_p(v['cascade_recall_at_1'])}** | {v['delta']*100:+.2f} pp |")
    A("")

    A("## Per-condition: stage 1 vs cascade\n")
    A("| Condition | n | Stage 1 | Cascade | Δ |")
    A("|---|---:|---:|---:|---:|")
    for c in r["by_condition"]:
        A(f"| `{c['condition']}` | {c['queries']} | {_p(c['stage1_recall_at_1'])} | "
          f"{_p(c['cascade_recall_at_1'])} | {c['delta']*100:+.2f} pp |")
    A("")

    A("## False accepts and rejection\n")
    A(f"Evaluation negatives: **{ds['evaluation_negatives']}** excerpts from "
      f"**{ds['evaluation_negative_source_recordings']}** source recordings. "
      f"One false accept ≈ **{(100.0/ds['evaluation_negatives']) if ds['evaluation_negatives'] else float('nan'):.4f} pp**. "
      f"Excerpt count is not statistical sample size — clips from one recording fail "
      f"together, so the effective diversity is the recording count.\n")
    A("| Category | Negatives | False accepts | FAR |")
    A("|---|---:|---:|---:|")
    for k, v in res["far_by_category_evaluation"].items():
        A(f"| `{k.replace('negative_','')}` | {v['negatives']} | {v['false_positives']} | {_p(v['far'],4)} |")
    A("")
    if res["false_accepts"]:
        A("| Query | Category | Stage | Correction | Matched |")
        A("|---|---|---:|---:|---|")
        for f in res["false_accepts"][:40]:
            A(f"| `{f['query_id'][:38]}` | {f['category'].replace('negative_','')} | "
              f"{f['stage']} | {f['rate']:+g}% | `{(f['matched'] or '-')[:24]}` |")
        A("")

    A("## Latency\n")
    A("| Stage | p50 ms | p95 ms | mean ms |")
    A("|---|---:|---:|---:|")
    A(f"| stage 1 (every query) | {tm['stage1_ms']['p50']:.2f} | {tm['stage1_ms']['p95']:.2f} | {tm['stage1_ms']['mean']:.2f} |")
    s2t = tm["stage2_ms_when_escalated"]
    A(f"| stage 2 (when escalated) | {s2t['p50']:.2f} | {s2t['p95']:.2f} | {s2t['mean']:.2f} |")
    t = tm["total_cascade_ms"]
    A(f"| **total cascade** | **{t['p50']:.2f}** | **{t['p95']:.2f}** | {t['mean']:.2f} |")
    A(f"\np99 total: {t['p99']:.2f} ms.\n")
    bt = tm.get("by_query_type_posthoc")
    if bt:
        A("**Post-hoc diagnostic — not a re-scored criterion.** Criterion 4 is judged on\n"
          "the whole-corpus p50 above and stands as measured. This split is reported because\n"
          "the benchmark corpus is 44% negatives and every negative escalates by design, so\n"
          "the corpus mix drives the median as much as the cascade does.\n")
        A("| Query type | Queries | Escalation rate | p50 ms | p95 ms |")
        A("|---|---:|---:|---:|---:|")
        for k, v in bt.items():
            A(f"| {k} | {v['queries']} | {_p(v['escalation_rate'])} | {v['p50']:.2f} | {v['p95']:.2f} |")
        A("")

    gs = r["grid_study_calibration_only"]
    A("## Is the full grid necessary? (exploratory, calibration only)\n")
    A(f"> {gs['note']}\n")
    A("| Grid | Hypotheses | Calibration Recall@1 |")
    A("|---|---:|---:|")
    for k, v in gs["grids"].items():
        A(f"| {k} | {v['hypotheses_per_escalation']} | {_p(v['calibration_recall_at_1'])} |")
    rp = r["rate_profile_study"]
    A(f"\n**Toward a continuous rate estimator.** Of {rp['escalated_calibration_queries_with_signal']} "
      f"escalated calibration queries with any signal, **{_p(rp['unimodal_fraction'])}** had a "
      f"single-peaked evidence-vs-rate profile. {rp['note']}\n")

    A("## Limitations\n")
    for x in r["limitations"]:
        A(f"- {x}")
    A("")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
