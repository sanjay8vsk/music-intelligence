#!/usr/bin/env python
"""Phase 1G — higher-resolution false-accept benchmark.

Runs the CURRENT Phase 1 recognizer, entirely unmodified, against:
  * the existing 1,728 positive queries, reused verbatim;
  * the existing 126 negatives, reused verbatim;
  * the expanded negative set built by scripts/build_negative_set.py.

Only the negative sample size changes. The recognizer, the positive corpus and
the fingerprint/index/matcher/decision code are identical to Phase 1E, so any
difference in FAR is a property of the measurement, not of the system.

The frozen Phase 1E report is READ ONLY here. Output goes to
eval/reports/phase1g_benchmark.{json,md}.

    python scripts/eval_phase1g.py
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
from musicintel.eval.negatives import NegativeSet  # noqa: E402
from musicintel.eval.provenance import (  # noqa: E402
    ALGORITHM_SOURCES, HARNESS_SOURCES, PHASE1_SOURCES, git_state, source_fingerprint,
)
from musicintel.recognition.decision import DEFAULT_DECISION_CONFIG  # noqa: E402
from musicintel.recognition.fingerprint import (  # noqa: E402
    FORMAT_VERSION, FingerprintConfig, fingerprint, load_audio,
)
from musicintel.recognition.index import INDEX_FORMAT_VERSION, build_index  # noqa: E402
from musicintel.recognition.matcher import MatchConfig, match  # noqa: E402

PHASE1E_REPORT = REPO_ROOT / "eval/reports/phase1e_benchmark.json"
MIN_ALIGNED = DEFAULT_DECISION_CONFIG.min_aligned_landmarks
SWEEP_POINTS = 120
_COMPACT = "@@COMPACT_SWEEP@@"


def run_one(path, index, fp_cfg, match_cfg):
    t0 = time.perf_counter()
    y, sr = load_audio(path, fp_cfg)
    q = fingerprint(y, sr, fp_cfg)
    t_fp = time.perf_counter() - t0
    r = match(q, index, config=match_cfg)
    top = r.best
    aligned = top.score if top else 0
    return {
        "query_landmarks": len(q), "aligned": aligned,
        "evidence": (aligned / len(q)) if len(q) else 0.0,
        "top_id": top.track_id if top else None,
        "ms_fingerprint": t_fp * 1000, "ms_lookup": r.timing.lookup * 1000,
        "ms_histogram": r.timing.histogram * 1000, "ms_rank": r.timing.ranking * 1000,
    }


def accepts(r, threshold):
    return r["evidence"] >= threshold and r["aligned"] >= MIN_ALIGNED


def confusion(rows, threshold):
    pos = [r for r in rows if not r["is_negative"]]
    neg = [r for r in rows if r["is_negative"]]
    tp = sum(1 for r in pos if accepts(r, threshold) and r["top_id"] == r["truth_track_id"])
    wrong = sum(1 for r in pos if accepts(r, threshold) and r["top_id"] != r["truth_track_id"])
    fp = sum(1 for r in neg if accepts(r, threshold))
    acc = tp + wrong + fp
    return {
        "threshold": float(threshold), "positives": len(pos), "negatives": len(neg),
        "true_positives": tp, "wrong_accepts": wrong,
        "false_negatives": len(pos) - tp - wrong,
        "true_negatives": len(neg) - fp, "false_positives": fp,
        "recall_at_1": round(tp / len(pos), 4) if pos else None,
        "far": round(fp / len(neg), 6) if neg else None,
        "precision": round(tp / acc, 4) if acc else None,
        "correct_rejection_rate": round((len(neg) - fp) / len(neg), 4) if neg else None,
    }


def sweep_columnar(rows, n=SWEEP_POINTS):
    scores = sorted({r["evidence"] for r in rows})
    if len(scores) > n:
        scores = [scores[i] for i in np.linspace(0, len(scores) - 1, n).astype(int)]
    pts = [confusion(rows, t) for t in scores]
    cols = ("threshold", "true_positives", "false_negatives", "true_negatives",
            "false_positives", "recall_at_1", "far", "precision", "correct_rejection_rate")
    return {"note": "columnar; index-aligned arrays", "points": len(pts),
            "columns": list(cols), **{c: [p[c] for p in pts] for c in cols}}


def pick_threshold(rows, far_target):
    best = None
    for t in sorted({r["evidence"] for r in rows}):
        p = confusion(rows, t)
        if p["far"] is not None and p["far"] <= far_target:
            if best is None or (p["recall_at_1"], -p["threshold"]) > (best["recall_at_1"], -best["threshold"]):
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
    args = ap.parse_args(argv)

    corpus = Manifest.load(REPO_ROOT / args.manifest)
    corpus.assign_holdout(args.holdout)
    catalog, heldout = corpus.catalog, corpus.held_out
    ns = NegativeSet.load(REPO_ROOT / args.negatives)
    problems = ns.verify({t.track_id for t in catalog})
    if problems:
        print(f"ERROR: negative set failed verification: {problems[:5]}")
        return 2
    print(f"Corpus {len(corpus)} | catalog {len(catalog)} | held out {len(heldout)}")
    print(f"Expanded negatives: {len(ns.excerpts)} from {len(ns.sources)} source recordings")

    legacy = [json.loads(x) for x in (REPO_ROOT / args.query_index).read_text().splitlines()]
    missing = [r for r in legacy if not (REPO_ROOT / r["rendered_path"]).exists()]
    if missing:
        print(f"ERROR: {len(missing)} legacy queries missing")
        return 2

    fp_cfg, match_cfg = FingerprintConfig(), MatchConfig()
    print(f"Indexing {len(catalog)} catalog tracks...")
    t0 = time.perf_counter()
    index = build_index([(t.track_id, fingerprint(*load_audio(REPO_ROOT / t.path, fp_cfg), fp_cfg))
                         for t in catalog], config=fp_cfg)
    t_index = time.perf_counter() - t0
    print(f"  {len(index):,} postings in {t_index:.0f}s")

    # -- assemble the query list -------------------------------------------
    cal_cat = {t for i, t in enumerate(sorted(x.track_id for x in catalog)) if i % 2 == 0}
    cal_held = {t for i, t in enumerate(sorted(x.track_id for x in heldout)) if i % 2 == 0}
    import hashlib

    def legacy_split(rec):
        if not rec["is_negative"]:
            return "calibration" if rec.get("track_id") in cal_cat else "evaluation"
        st = rec.get("params", {}).get("source_track")
        if st is None:
            h = int.from_bytes(hashlib.sha256(rec["query_id"].encode()).digest()[:2], "big")
            return "calibration" if h % 2 == 0 else "evaluation"
        return "calibration" if st in cal_held else "evaluation"

    queries = []
    for rec in legacy:
        queries.append({
            "query_id": rec["query_id"], "path": REPO_ROOT / rec["rendered_path"],
            "is_negative": rec["is_negative"], "truth_track_id": rec.get("track_id"),
            "category": rec["condition"] if rec["is_negative"] else None,
            "source_track": rec.get("params", {}).get("source_track"),
            "split": legacy_split(rec), "cohort": "phase1e",
        })
    for e in ns.excerpts:
        queries.append({
            "query_id": e.query_id, "path": REPO_ROOT / e.rendered_path,
            "is_negative": True, "truth_track_id": None, "category": e.category,
            "source_track": e.source_track, "split": e.split, "cohort": "phase1g",
        })
    print(f"Total queries: {len(queries)} "
          f"({sum(1 for q in queries if not q['is_negative'])} pos / "
          f"{sum(1 for q in queries if q['is_negative'])} neg)")

    print("Running...")
    rows = []
    for i, q in enumerate(queries, 1):
        try:
            rows.append({**q, **run_one(q["path"], index, fp_cfg, match_cfg), "error": None})
        except Exception as ex:  # noqa: BLE001
            rows.append({**q, "query_landmarks": 0, "aligned": 0, "evidence": 0.0,
                         "top_id": None, "ms_fingerprint": 0.0, "ms_lookup": 0.0,
                         "ms_histogram": 0.0, "ms_rank": 0.0,
                         "error": f"{type(ex).__name__}: {ex}"})
        if i % 400 == 0:
            print(f"    {i}/{len(queries)}")

    cal = [r for r in rows if r["split"] == "calibration"]
    ev = [r for r in rows if r["split"] == "evaluation"]
    print(f"  calibration {len(cal)} ({sum(1 for r in cal if r['is_negative'])} neg) | "
          f"evaluation {len(ev)} ({sum(1 for r in ev if r['is_negative'])} neg)")

    chosen = pick_threshold(cal, args.far_target)
    met_cal = chosen is not None
    if chosen is None:
        pts = [confusion(cal, t) for t in sorted({r["evidence"] for r in cal})]
        zero = [p for p in pts if p["far"] == 0.0]
        chosen = max(zero, key=lambda p: p["recall_at_1"]) if zero else pts[-1]
    threshold = chosen["threshold"]
    print(f"  threshold = {threshold:.6f}")

    eval_pt, all_pt = confusion(ev, threshold), confusion(rows, threshold)

    def by(key, subset):
        out = {}
        for r in subset:
            out.setdefault(r[key] or "-", []).append(r)
        return {k: confusion(v, threshold) for k, v in sorted(out.items())}

    neg_ev = [r for r in ev if r["is_negative"]]
    n_neg_ev = len(neg_ev)
    src_ev = len({r["source_track"] for r in neg_ev if r["source_track"]})
    fa = [{"query_id": r["query_id"], "category": r["category"], "cohort": r["cohort"],
           "source_track": r["source_track"], "matched": r["top_id"],
           "evidence": round(r["evidence"], 6), "aligned": r["aligned"],
           "query_landmarks": r["query_landmarks"]}
          for r in rows if r["is_negative"] and accepts(r, threshold)]

    ok = [r for r in rows if r["error"] is None]
    stages = {k: [r[f"ms_{k}"] for r in ok] for k in ("fingerprint", "lookup", "histogram", "rank")}
    stages["total"] = [sum(r[f"ms_{k}"] for k in ("fingerprint", "lookup", "histogram", "rank")) for r in ok]
    timing = {k: {"p50": pctl(v, 50), "p95": pctl(v, 95), "mean": round(float(np.mean(v)), 3)}
              for k, v in stages.items()}

    p1e = json.loads(PHASE1E_REPORT.read_text())
    git = git_state(REPO_ROOT)
    p1_paths = tuple(PHASE1_SOURCES) + ("scripts/eval_phase1g.py", "musicintel/eval/negatives.py")

    results = {
        "schema_version": 1, "phase": "1G",
        "title": "Higher-resolution false-accept benchmark (expanded negative set)",
        "what_changed": "negative sample size only",
        "what_did_not_change": [
            "recognizer (fingerprint/index/matcher/decision) — byte-identical to Phase 1E",
            "the 1,728 positive queries — reused verbatim",
            "the original 126 negatives — reused verbatim, still part of the set",
            "the positive calibration/evaluation split policy",
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
            "recognizer_version": f"landmark@{git['commit_short']}" + ("+dirty" if git["dirty"] else ""),
            "benchmark_command": "python scripts/eval_phase1g.py",
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
            "negatives_by_category": ns.counts_by_category(),
            "negative_source_recordings_by_split": ns.source_counts(),
        },
        "configuration": {
            "decision": {"evidence_score": "aligned_landmarks / query_landmarks",
                         "score_is_probability": False, "threshold": threshold,
                         "min_aligned_landmarks": MIN_ALIGNED},
            "index": {"tracks": index.n_tracks, "postings": len(index),
                      "build_seconds": round(t_index, 2)},
        },
        "threshold_selection": {
            "far_target": args.far_target, "met_on_calibration": met_cal,
            "met_on_evaluation": bool(eval_pt["far"] is not None and eval_pt["far"] <= args.far_target),
            "selected_on": "calibration split only",
            "calibration_operating_point": chosen,
        },
        "results": {
            "evaluation_holdout": eval_pt, "calibration": chosen, "all_queries": all_pt,
            "far_by_category_evaluation": by("category", neg_ev),
            "far_by_category_all": by("category", [r for r in rows if r["is_negative"]]),
            "far_by_cohort_evaluation": by("cohort", neg_ev),
            "false_accepts": fa,
            "negative_resolution": {
                "evaluation_negatives": n_neg_ev,
                "evaluation_source_recordings": src_ev,
                "one_false_accept_pct_points": round(100.0 / n_neg_ev, 4) if n_neg_ev else None,
                "smallest_measurable_nonzero_far": round(1.0 / n_neg_ev, 6) if n_neg_ev else None,
                "note": "FAR resolution is set by the NUMBER OF EVALUATION NEGATIVES. "
                        "Excerpt count is not sample size: clips from one recording fail "
                        "together, so the effective diversity is the source-recording count.",
            },
        },
        "phase1e_comparison": {
            "note": "Phase 1E and Phase 1G measure DIFFERENT negative populations. "
                    "Phase 1E's 63 evaluation negatives came from 6 held-out recordings; "
                    "Phase 1G adds newly fetched sources and disjoint excerpting. The FAR "
                    "figures are therefore NOT directly equivalent — what improved is "
                    "resolution, not the system.",
            "phase1e_report": "eval/reports/phase1e_benchmark.json (unmodified)",
            "phase1e_evaluation": p1e["results"]["evaluation_holdout"],
            "phase1e_negative_resolution": p1e["results"]["negative_resolution"],
        },
        "sweep_calibration": _COMPACT,
        "timing_ms": timing,
        "limitations": [
            "Excerpt count is not statistical sample size. Clips from one recording share "
            "mastering and instrumentation and fail together; the effective diversity is "
            "the source-recording count, reported alongside every FAR figure.",
            "Aggregate FAR depends on category mix. Silence and noise are trivially "
            "rejected, so adding them lowers aggregate FAR without improving the system. "
            "Out-of-catalog MUSIC is the number that matters and is reported separately.",
            "The catalog is still 32 tracks; difficulty grows with catalog size.",
            "Negative sources are CC-licensed netlabel music, not a genre-balanced sample "
            "of commercial music.",
        ],
    }

    rd = REPO_ROOT / args.report_dir
    rd.mkdir(parents=True, exist_ok=True)
    text = json.dumps(results, indent=2).replace(
        f'"{_COMPACT}"', json.dumps(sweep_columnar(cal), separators=(",", ":")))
    (rd / "phase1g_benchmark.json").write_text(text + "\n")
    results["sweep_calibration"] = sweep_columnar(cal)
    (rd / "phase1g_benchmark.md").write_text(build_md(results))

    print("\n" + "=" * 70)
    print(f"  threshold           : {threshold:.6f}")
    print(f"  EVAL negatives      : {n_neg_ev} from {src_ev} source recordings")
    print(f"  EVAL Recall@1       : {eval_pt['recall_at_1']:.4f}")
    print(f"  EVAL FAR            : {eval_pt['far']:.6f} ({eval_pt['false_positives']}/{n_neg_ev})")
    print(f"  1 false accept      = {100.0/n_neg_ev:.4f} pp   (Phase 1E: 1.5873 pp)")
    print(f"  EVAL precision      : {eval_pt['precision']}")
    print("=" * 70)
    print(f"Wrote {rd/'phase1g_benchmark.json'}\nWrote {rd/'phase1g_benchmark.md'}")
    return 0


def _p(v, nd=2):
    return "—" if v is None else f"{v*100:.{nd}f}%"


def build_md(r):
    L = []; A = L.append
    pv, ds, res = r["provenance"], r["dataset"], r["results"]
    ev, cal, allq = res["evaluation_holdout"], res["calibration"], res["all_queries"]
    nr = res["negative_resolution"]; cmp_ = r["phase1e_comparison"]
    A("# Phase 1G — Higher-Resolution False-Accept Benchmark\n")
    A(f"**Recognizer:** `{pv['recognizer_version']}` — unchanged from Phase 1E  ")
    A(f"**Generated:** {pv['generated_utc']}  ")
    A(f"**Repo commit:** `{pv['git_commit'][:12]}`  ")
    A(f"**Working tree:** {'DIRTY (' + str(pv['git_dirty_path_count']) + ' paths)' if pv['git_dirty'] else 'clean'}  ")
    A(f"**Phase 1 fingerprint:** `{pv['phase1_source_sha256']}`\n")
    A("> **What changed: the negative sample size only.** The recognizer, the 1,728\n"
      "> positive queries and the original 126 negatives are all reused unchanged.\n"
      "> `eval/reports/baseline.md`, `phase1d_baseline.md` and `phase1e_benchmark.md`\n"
      "> are untouched.\n")

    A("## Negative set\n")
    A(f"- **{ds['negative_queries']} negatives** from **{ds['negative_sources']} source recordings**")
    A(f"- Evaluation split: **{nr['evaluation_negatives']} negatives** from "
      f"**{nr['evaluation_source_recordings']} recordings**")
    A(f"- **One false accept = {nr['one_false_accept_pct_points']} percentage points** "
      f"(Phase 1E: 1.5873 pp)")
    A(f"- Smallest resolvable non-zero FAR: **{nr['smallest_measurable_nonzero_far']*100:.4f}%** "
      f"(Phase 1E: 1.5873%)\n")
    A("| Category | Count |")
    A("|---|---:|")
    for k, v in ds["negatives_by_category"].items():
        A(f"| `{k.replace('negative_','')}` | {v} |")
    A("")
    A("**Excerpt count is not sample size.** Clips from one recording share mastering and\n"
      "instrumentation and fail together, so the effective diversity is the source-recording\n"
      "count, quoted above alongside every FAR figure.\n")

    A("## Results (evaluation split — threshold fitted only on calibration)\n")
    A("| Metric | Calibration | **Evaluation** | All queries |")
    A("|---|---:|---:|---:|")
    A(f"| Recall@1 | {_p(cal['recall_at_1'])} | **{_p(ev['recall_at_1'])}** | {_p(allq['recall_at_1'])} |")
    A(f"| FAR | {_p(cal['far'],4)} | **{_p(ev['far'],4)}** | {_p(allq['far'],4)} |")
    A(f"| Correct rejection | {_p(cal['correct_rejection_rate'])} | **{_p(ev['correct_rejection_rate'])}** | {_p(allq['correct_rejection_rate'])} |")
    A(f"| Precision | {_p(cal['precision'])} | **{_p(ev['precision'])}** | {_p(allq['precision'])} |")
    A(f"| TP | {cal['true_positives']} | **{ev['true_positives']}** | {allq['true_positives']} |")
    A(f"| FP | {cal['false_positives']} | **{ev['false_positives']}** | {allq['false_positives']} |")
    A(f"| TN | {cal['true_negatives']} | **{ev['true_negatives']}** | {allq['true_negatives']} |")
    A(f"| FN | {cal['false_negatives']} | **{ev['false_negatives']}** | {allq['false_negatives']} |")
    A("")

    A("### FAR by negative category (evaluation split)\n")
    A("| Category | Negatives | False accepts | FAR |")
    A("|---|---:|---:|---:|")
    for k, v in res["far_by_category_evaluation"].items():
        A(f"| `{k.replace('negative_','')}` | {v['negatives']} | {v['false_positives']} | {_p(v['far'],4)} |")
    A("\n**Aggregate FAR depends on category mix.** Silence and noise are trivially rejected, "
      "so including them lowers the aggregate without improving anything. Out-of-catalog "
      "music is the number that matters.\n")

    A("### FAR by cohort (evaluation split)\n")
    A("| Cohort | Negatives | False accepts | FAR |")
    A("|---|---:|---:|---:|")
    for k, v in res["far_by_cohort_evaluation"].items():
        A(f"| {k} | {v['negatives']} | {v['false_positives']} | {_p(v['far'],4)} |")
    A("")

    A("## Comparison with the frozen Phase 1E benchmark\n")
    p1e = cmp_["phase1e_evaluation"]; p1r = cmp_["phase1e_negative_resolution"]
    A("| | Phase 1E (frozen) | Phase 1G |")
    A("|---|---:|---:|")
    A(f"| Evaluation negatives | {p1r['evaluation_negatives']} | **{nr['evaluation_negatives']}** |")
    A(f"| Source recordings behind them | 6 | **{nr['evaluation_source_recordings']}** |")
    A(f"| 1 false accept = | {p1r['one_false_accept_pct_points']} pp | **{nr['one_false_accept_pct_points']} pp** |")
    A(f"| FAR | {_p(p1e['far'],4)} | **{_p(ev['far'],4)}** |")
    A(f"| Recall@1 | {_p(p1e['recall_at_1'])} | {_p(ev['recall_at_1'])} |")
    A("")
    A(f"> {cmp_['note']}\n")

    A("## Latency\n")
    A("| Stage | p50 ms | p95 ms | mean ms |")
    A("|---|---:|---:|---:|")
    for k, v in r["timing_ms"].items():
        A(f"| {k} | {v['p50']:.2f} | {v['p95']:.2f} | {v['mean']:.2f} |")
    A("")
    if res["false_accepts"]:
        A("## Every false accept, itemized\n")
        A("| Query | Category | Cohort | Matched | Evidence | Aligned |")
        A("|---|---|---|---|---:|---:|")
        for f in res["false_accepts"][:60]:
            A(f"| `{f['query_id'][:38]}` | {f['category'].replace('negative_','')} | {f['cohort']} | "
              f"`{(f['matched'] or '-')[:24]}` | {f['evidence']:.4f} | {f['aligned']} |")
        if len(res["false_accepts"]) > 60:
            A(f"\n_({len(res['false_accepts'])-60} more in the JSON.)_")
        A("")
    A("## Limitations\n")
    for x in r["limitations"]:
        A(f"- {x}")
    A("")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
