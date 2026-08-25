#!/usr/bin/env python
"""Phase 1H-gated — the latency-reduced cascade, scored on Phase 1H methodology.

Same 1,728 positives, same 1,361 negatives, same split, same metrics code, same
four acceptance criteria as the official Phase 1H benchmark. The only change is
the orchestration: a concentration gate and a 2 s probe in front of a 4-point
rate sweep, followed by full-query confirmation.

Two passes, deliberately:
  PASS A  calibration split only, gate open and stage-2 open, so every stage is
          computed. Both thresholds are derived here and nowhere else.
  PASS B  the REAL gated cascade over the whole corpus with the frozen
          thresholds. Latency is wall-clock from this pass, not reconstructed.

    python scripts/eval_phase1h_gated.py
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
from musicintel.recognition.fingerprint import (  # noqa: E402
    FORMAT_VERSION, FingerprintConfig, fingerprint, load_audio,
)
from musicintel.recognition.gated_cascade import (  # noqa: E402
    GATED_RATE_GRID, PROBE_SECONDS, GatedCascadeConfig, identify_gated,
)
from musicintel.recognition.index import INDEX_FORMAT_VERSION, build_index  # noqa: E402
from musicintel.recognition.matcher import MatchConfig  # noqa: E402

PHASE1H_REPORT = REPO_ROOT / "eval/reports/phase1h_benchmark.json"
STAGE1_THRESHOLD = 0.026316
MIN_ALIGNED = 5
# Fixed in advance: the gate may cost at most this much calibration recall.
# Mirrors criterion 2's -1 pp discipline; it is NOT tuned on evaluation data.
GATE_RECALL_TOLERANCE = 0.01
_COMPACT = "@@COMPACT_SWEEP@@"


def confusion(rows, verdict):
    """verdict(row) -> (accepted, track_id)."""
    tp = wrong = fp = 0
    pos = [r for r in rows if not r["is_negative"]]
    neg = [r for r in rows if r["is_negative"]]
    for r in pos:
        acc, tid = verdict(r)
        if acc:
            tp += int(tid == r["truth_track_id"])
            wrong += int(tid != r["truth_track_id"])
    for r in neg:
        fp += int(verdict(r)[0])
    accd = tp + wrong + fp
    return {"positives": len(pos), "negatives": len(neg), "true_positives": tp,
            "wrong_accepts": wrong, "false_negatives": len(pos) - tp - wrong,
            "true_negatives": len(neg) - fp, "false_positives": fp,
            "recall_at_1": round(tp / len(pos), 4) if pos else None,
            "far": round(fp / len(neg), 6) if neg else None,
            "precision": round(tp / accd, 4) if accd else None,
            "correct_rejection_rate": round((len(neg) - fp) / len(neg), 4) if neg else None}


def replay(row, gate_t, s2_t):
    """Exact analytic replay of the gated pipeline at candidate thresholds."""
    if row["stage1_accept"]:
        return True, row["stage1_top"]
    if row["gate_value"] < gate_t:
        return False, None
    if not row["probe_passed"]:
        return False, None
    if row["confirm_id"] is None or row["confirm_aligned"] < MIN_ALIGNED:
        return False, None
    return (row["confirm_evidence"] >= s2_t), (
        row["confirm_id"] if row["confirm_evidence"] >= s2_t else None)


def escalation_rate(rows, gate_t):
    """Fraction that reaches the probe -- what drives the median latency."""
    n = sum(1 for r in rows
            if (not r["stage1_accept"]) and r["gate_value"] >= gate_t)
    return n / len(rows) if rows else 0.0


def pctl(v, p):
    return round(float(np.percentile(v, p)), 3) if v else None


def build_queries(corpus, ns, cal_cat, cal_held, query_index):
    def legacy_split(rec):
        if not rec["is_negative"]:
            return "calibration" if rec.get("track_id") in cal_cat else "evaluation"
        st = rec.get("params", {}).get("source_track")
        if st is None:
            h = int.from_bytes(hashlib.sha256(rec["query_id"].encode()).digest()[:2], "big")
            return "calibration" if h % 2 == 0 else "evaluation"
        return "calibration" if st in cal_held else "evaluation"

    qs = []
    for rec in [json.loads(x) for x in query_index.read_text().splitlines()]:
        qs.append({"query_id": rec["query_id"], "path": REPO_ROOT / rec["rendered_path"],
                   "is_negative": rec["is_negative"], "truth_track_id": rec.get("track_id"),
                   "condition": rec["condition"], "family": rec["family"],
                   "duration": rec["duration"], "position": rec["position"],
                   "category": rec["condition"] if rec["is_negative"] else None,
                   "split": legacy_split(rec), "cohort": "phase1e"})
    for e in ns.excerpts:
        qs.append({"query_id": e.query_id, "path": REPO_ROOT / e.rendered_path,
                   "is_negative": True, "truth_track_id": None, "condition": e.category,
                   "family": "negative", "duration": e.duration, "position": "beginning",
                   "category": e.category, "split": e.split, "cohort": "phase1g"})
    return qs


def run(queries, index, cfg, fp_cfg, m_cfg, label):
    rows = []
    for i, q in enumerate(queries, 1):
        try:
            y, sr = load_audio(q["path"], fp_cfg)
            r = identify_gated(y, sr, index, config=cfg, match_config=m_cfg,
                               fingerprint_config=fp_cfg)
            d1 = r.stage1_decision
            rows.append({**q,
                         "stage1_accept": bool(d1.is_match),
                         "stage1_top": d1.track_id if d1.is_match else (
                             d1.candidates[0].track_id if d1.candidates else None),
                         "gate_value": r.gate_value,
                         "gate_passed": r.gate_passed,
                         "probe_passed": r.probe_passed,
                         "confirm_id": r.track_id if r.stage == 2 else None,
                         "confirm_evidence": r.evidence_score if r.stage == 2 else 0.0,
                         "confirm_aligned": r.aligned_landmarks if r.stage == 2 else 0,
                         "rate": r.rate_percent,
                         "accepted": r.is_match, "stage": r.stage,
                         "track_id": r.track_id,
                         "ms_stage1": r.timing.stage1_ms, "ms_probe": r.timing.probe_ms,
                         "ms_confirm": r.timing.confirm_ms,
                         "ms_total": r.timing.total_ms, "error": None})
        except Exception as ex:  # noqa: BLE001
            rows.append({**q, "stage1_accept": False, "stage1_top": None, "gate_value": 0.0,
                         "gate_passed": False, "probe_passed": False, "confirm_id": None,
                         "confirm_evidence": 0.0, "confirm_aligned": 0, "rate": 0.0,
                         "accepted": False, "stage": None, "track_id": None,
                         "ms_stage1": 0.0, "ms_probe": 0.0, "ms_confirm": 0.0,
                         "ms_total": 0.0, "error": f"{type(ex).__name__}: {ex}"})
        if i % 400 == 0:
            print(f"    [{label}] {i}/{len(queries)}")
    return rows


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
    fp_cfg, m_cfg = FingerprintConfig(), MatchConfig()
    print(f"Corpus {len(corpus)} | catalog {len(catalog)} | held out {len(heldout)}")
    print(f"Grid {list(GATED_RATE_GRID)} | probe {PROBE_SECONDS}s | stage1 {STAGE1_THRESHOLD}")

    t0 = time.perf_counter()
    index = build_index([(t.track_id, fingerprint(*load_audio(REPO_ROOT / t.path, fp_cfg), fp_cfg))
                         for t in catalog], config=fp_cfg)
    t_index = time.perf_counter() - t0
    print(f"Index: {len(index):,} postings in {t_index:.0f}s")

    cal_cat = {t for i, t in enumerate(sorted(x.track_id for x in catalog)) if i % 2 == 0}
    cal_held = {t for i, t in enumerate(sorted(x.track_id for x in heldout)) if i % 2 == 0}
    queries = build_queries(corpus, ns, cal_cat, cal_held,
                            REPO_ROOT / args.query_index)
    cal_q = [q for q in queries if q["split"] == "calibration"]
    print(f"Queries {len(queries)} | calibration {len(cal_q)}")

    # ---------------- PASS A: calibration, everything open ----------------
    print("PASS A — calibration, gate open, stage-2 open...")
    open_cfg = GatedCascadeConfig(rate_grid=GATED_RATE_GRID,
                                  stage1_threshold=STAGE1_THRESHOLD,
                                  gate_threshold=0.0, probe_seconds=PROBE_SECONDS,
                                  stage2_threshold=0.0, min_aligned_landmarks=MIN_ALIGNED)
    cal = run(cal_q, index, open_cfg, fp_cfg, m_cfg, "A")

    # stage-2 threshold: highest calibration recall with end-to-end FAR in target,
    # gate open so the two derivations do not contaminate each other.
    cands = sorted({r["confirm_evidence"] for r in cal if r["confirm_evidence"] > 0})
    best_s2 = None
    for t in cands:
        p = confusion(cal, lambda r, t=t: replay(r, 0.0, t))
        if p["far"] is not None and p["far"] <= args.far_target:
            if best_s2 is None or (p["recall_at_1"], -t) > (best_s2[1]["recall_at_1"], -best_s2[0]):
                best_s2 = (t, p)
    if best_s2 is None:
        best_s2 = (1.0, confusion(cal, lambda r: replay(r, 0.0, 1.0)))
    s2 = best_s2[0]
    ungated = confusion(cal, lambda r: replay(r, 0.0, s2))
    print(f"  stage-2 threshold = {s2:.6f}  (calibration R@1 {ungated['recall_at_1']:.4f}, "
          f"FAR {ungated['far']:.6f})")

    # gate threshold: the most aggressive gate whose calibration recall stays
    # within GATE_RECALL_TOLERANCE of the ungated cascade; among those, the one
    # with the lowest escalation rate. Both quantities are calibration-only.
    floor = (ungated["recall_at_1"] or 0.0) - GATE_RECALL_TOLERANCE
    gate_cands = sorted({0.0} | {round(r["gate_value"], 6) for r in cal
                                 if not r["stage1_accept"]})
    gate_rows = []
    for g in gate_cands:
        p = confusion(cal, lambda r, g=g: replay(r, g, s2))
        e = escalation_rate(cal, g)
        gate_rows.append({"gate": g, "recall_at_1": p["recall_at_1"], "far": p["far"],
                          "escalation": round(e, 4)})
    ok = [x for x in gate_rows if (x["recall_at_1"] or 0) >= floor]
    chosen_gate = min(ok, key=lambda x: (x["escalation"], -x["gate"]))["gate"] if ok else 0.0
    gsel = [x for x in gate_rows if x["gate"] == chosen_gate][0]
    print(f"  gate threshold    = {chosen_gate:.6f}  (calibration R@1 {gsel['recall_at_1']:.4f} "
          f">= floor {floor:.4f}, escalation {gsel['escalation']*100:.2f}%)")

    # ---------------- PASS B: the real gated cascade, whole corpus --------
    print("PASS B — real gated cascade over the whole corpus...")
    final_cfg = GatedCascadeConfig(rate_grid=GATED_RATE_GRID,
                                   stage1_threshold=STAGE1_THRESHOLD,
                                   gate_threshold=chosen_gate,
                                   probe_seconds=PROBE_SECONDS,
                                   stage2_threshold=s2,
                                   min_aligned_landmarks=MIN_ALIGNED)
    rows = run(queries, index, final_cfg, fp_cfg, m_cfg, "B")
    ev = [r for r in rows if r["split"] == "evaluation"]

    live = lambda r: (r["accepted"], r["track_id"])  # noqa: E731
    s1_only = lambda r: (r["stage1_accept"], r["stage1_top"])  # noqa: E731
    ev_cascade, ev_stage1 = confusion(ev, live), confusion(ev, s1_only)
    all_cascade, all_stage1 = confusion(rows, live), confusion(rows, s1_only)

    def outcomes(subset, fn):
        out = []
        for r in subset:
            acc, tid = fn(r)
            out.append(QueryOutcome(
                query_id=r["query_id"], condition=r["condition"], family=r["family"],
                duration=r["duration"], position=r["position"],
                is_negative=r["is_negative"], latency_ms=r["ms_total"],
                returned_ids=[tid] if (acc and tid) else [],
                truth_track_id=r["truth_track_id"], error=r["error"]))
        return out

    pos = [r for r in rows if not r["is_negative"]]
    s1r = {x["condition"]: x for x in by_condition_and_duration(outcomes(pos, s1_only))}
    csr = {x["condition"]: x for x in by_condition_and_duration(outcomes(pos, live))}
    per_condition = [{"condition": c, "family": csr[c]["family"], "queries": csr[c]["queries"],
                      "stage1_recall_at_1": s1r[c]["recall_at_1"],
                      "cascade_recall_at_1": csr[c]["recall_at_1"],
                      "delta": round((csr[c]["recall_at_1"] or 0) - (s1r[c]["recall_at_1"] or 0), 4)}
                     for c in sorted(set(s1r) & set(csr))]
    fam = {}
    for c in per_condition:
        f = fam.setdefault(c["family"], {"n": 0, "s1": 0.0, "cs": 0.0})
        f["n"] += c["queries"]
        f["s1"] += (c["stage1_recall_at_1"] or 0) * c["queries"]
        f["cs"] += (c["cascade_recall_at_1"] or 0) * c["queries"]
    by_family = {k: {"queries": v["n"], "stage1_recall_at_1": round(v["s1"] / v["n"], 4),
                     "cascade_recall_at_1": round(v["cs"] / v["n"], 4),
                     "delta": round((v["cs"] - v["s1"]) / v["n"], 4)}
                 for k, v in sorted(fam.items())}

    n = len(rows)
    ok_rows = [r for r in rows if r["error"] is None]
    t_tot = [r["ms_total"] for r in ok_rows]
    t_s1 = [r["ms_stage1"] for r in ok_rows]
    t_pr = [r["ms_probe"] for r in ok_rows if r["gate_passed"]]
    t_cf = [r["ms_confirm"] for r in ok_rows if r["probe_passed"]]
    timing = {"stage1_ms": {"p50": pctl(t_s1, 50), "p95": pctl(t_s1, 95)},
              "probe_ms_when_gate_passed": {"p50": pctl(t_pr, 50), "p95": pctl(t_pr, 95)},
              "confirm_ms_when_probe_passed": {"p50": pctl(t_cf, 50), "p95": pctl(t_cf, 95)},
              "total_ms": {"p50": pctl(t_tot, 50), "p95": pctl(t_tot, 95),
                           "p99": pctl(t_tot, 99), "mean": round(float(np.mean(t_tot)), 3)}}

    behaviour = {
        "stage1_match_rate": round(sum(1 for r in rows if r["stage1_accept"]) / n, 4),
        "gate_pass_rate": round(sum(1 for r in rows if r["gate_passed"]) / n, 4),
        "escalation_rate": round(sum(1 for r in rows if r["gate_passed"]) / n, 4),
        "probe_pass_rate": round(sum(1 for r in rows if r["probe_passed"]) / n, 4),
        "stage2_acceptances": sum(1 for r in rows if r["stage"] == 2),
        "gate_skipped_negatives": round(
            sum(1 for r in rows if r["is_negative"] and not r["stage1_accept"]
                and not r["gate_passed"]) / max(1, sum(1 for r in rows if r["is_negative"])), 4),
        "winning_rate_histogram": {},
    }
    for r in rows:
        if r["stage"] == 2:
            k = f"{r['rate']:+g}%"
            behaviour["winning_rate_histogram"][k] = behaviour["winning_rate_histogram"].get(k, 0) + 1
    behaviour["winning_rate_histogram"] = dict(sorted(
        behaviour["winning_rate_histogram"].items(), key=lambda kv: float(kv[0].rstrip("%"))))

    speed_c = [c for c in per_condition if c["family"] == "speed"]
    speed_recall = (sum(c["cascade_recall_at_1"] * c["queries"] for c in speed_c)
                    / sum(c["queries"] for c in speed_c)) if speed_c else 0.0
    preserved = {f: by_family[f]["delta"] for f in ("clean", "noise", "codec", "filter")
                 if f in by_family}
    neg_ev = [r for r in ev if r["is_negative"]]
    crit = {
        "1_speed_recall_ge_60pct": {"value": round(speed_recall, 4), "target": 0.60,
                                    "pass": bool(speed_recall >= 0.60)},
        "2_preserved_families_within_-1pp": {"deltas_vs_stage1": preserved, "target": -0.01,
                                             "pass": all(v >= -0.01 for v in preserved.values())},
        "3_holdout_far_le_5pct": {"value": ev_cascade["far"], "target": 0.05,
                                  "pass": bool(ev_cascade["far"] is not None and ev_cascade["far"] <= 0.05)},
        "4_p50_latency_le_40ms": {"value": timing["total_ms"]["p50"], "target": 40.0,
                                  "pass": bool(timing["total_ms"]["p50"] <= 40.0)},
    }
    crit["all_passed"] = all(v["pass"] for k, v in crit.items() if k != "all_passed")

    def far_by(key, subset):
        out = {}
        for r in subset:
            out.setdefault(r[key] or "-", []).append(r)
        return {k: confusion(v, live) for k, v in sorted(out.items())}

    git = git_state(REPO_ROOT)
    p1_paths = tuple(PHASE1_SOURCES) + ("musicintel/recognition/cascade.py",
                                        "musicintel/recognition/gated_cascade.py",
                                        "scripts/eval_phase1h_gated.py")
    src_ev = len({e.source_track for e in ns.excerpts
                  if e.split == "evaluation" and e.source_track})
    p1h = json.loads(PHASE1H_REPORT.read_text())

    results = {
        "schema_version": 1, "phase": "1H-gated",
        "title": "Gated speed-tolerant cascade — latency-reduced",
        "what_changed": "orchestration only: concentration gate + 2 s probe + 4-point grid "
                        "+ full-query confirmation",
        "what_did_not_change": [
            "fingerprint.py, index.py, matcher.py, decision.py — byte-identical",
            "cascade.py — untouched, so Phase 1H stays reproducible",
            "the 1,728 positive queries and the 1,361-negative corpus",
            "the split, the metrics implementation, the four acceptance criteria",
            "the stage-1 threshold (frozen Phase 1G/1H operating point)",
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
            "recognizer_version": f"landmark-gated@{git['commit_short']}"
                                  + ("+dirty" if git["dirty"] else ""),
            "rate_grid_percent": list(GATED_RATE_GRID),
            "probe_seconds": PROBE_SECONDS,
            "rate_convention": "apply_rate(+p%) plays faster and higher; a recording "
                               "captured at +p% is corrected by about -p%",
            "stage1_threshold": STAGE1_THRESHOLD, "gate_threshold": chosen_gate,
            "stage2_threshold": s2, "min_aligned_landmarks": MIN_ALIGNED,
            "benchmark_command": "python scripts/eval_phase1h_gated.py",
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
            "evaluation_negatives": len(neg_ev),
            "evaluation_negative_source_recordings": src_ev,
        },
        "threshold_derivation": {
            "stage1_threshold": STAGE1_THRESHOLD,
            "stage1_origin": "frozen Phase 1G/1H operating point, not re-fitted",
            "stage2_threshold": s2,
            "stage2_rule": "highest calibration Recall@1 with end-to-end cascade FAR "
                           f"<= {args.far_target}, gate open, calibration split only",
            "stage2_calibration_point": ungated,
            "gate_threshold": chosen_gate,
            "gate_rule": "most aggressive gate whose calibration Recall@1 stays within "
                         f"{GATE_RECALL_TOLERANCE} of the ungated cascade; among those, "
                         "lowest calibration escalation. Calibration split only.",
            "gate_recall_tolerance": GATE_RECALL_TOLERANCE,
            "gate_calibration_point": gsel,
            "gate_sweep": gate_rows[:: max(1, len(gate_rows) // 40)],
        },
        "results": {
            "evaluation_cascade": ev_cascade, "evaluation_stage1_only": ev_stage1,
            "all_queries_cascade": all_cascade, "all_queries_stage1_only": all_stage1,
            "far_by_category_evaluation": far_by("category", neg_ev),
            "false_accepts": [
                {"query_id": r["query_id"], "category": r["category"], "cohort": r["cohort"],
                 "stage": r["stage"], "rate": r["rate"], "matched": r["track_id"]}
                for r in rows if r["is_negative"] and r["accepted"]],
        },
        "cascade_behaviour": behaviour,
        "by_family": by_family, "by_condition": per_condition,
        "falsification_criteria": crit,
        "timing_ms": timing,
        "phase1h_reference": {
            "note": "Ungated Phase 1H, same corpus and criteria.",
            "evaluation": p1h["results"]["evaluation_cascade"],
            "p50_ms": p1h["timing_ms"]["total_cascade_ms"]["p50"],
            "escalation_rate": p1h["cascade_behaviour"]["stage2_escalation_rate"],
        },
        "limitations": [
            f"FAR rests on {len(neg_ev)} evaluation negatives from {src_ev} source "
            f"recordings; excerpt count is not statistical sample size.",
            "The gate's discriminative power is weak (AUC 0.639 measured in the Phase 1H "
            "investigation). It works because only ~20% of negatives need skipping, which "
            "makes the design sensitive to the corpus's negative share.",
            "Pitch is unaddressed by design and is not an acceptance criterion.",
            "The catalog is 32 tracks; difficulty grows with catalog size.",
        ],
        "sweep_calibration": _COMPACT,
    }

    pts = [confusion(cal, lambda r, t=t: replay(r, chosen_gate, t)) | {"threshold": t}
           for t in (cands[:: max(1, len(cands) // 100)] if cands else [1.0])]
    cols = ("threshold", "true_positives", "false_positives", "recall_at_1", "far", "precision")
    sweep = {"note": "stage-2 sweep at the chosen gate, calibration split, columnar",
             "points": len(pts), "columns": list(cols),
             **{c: [p[c] for p in pts] for c in cols}}

    rd = REPO_ROOT / args.report_dir
    rd.mkdir(parents=True, exist_ok=True)
    text = json.dumps(results, indent=2).replace(
        f'"{_COMPACT}"', json.dumps(sweep, separators=(",", ":")))
    (rd / "phase1h_gated_benchmark.json").write_text(text + "\n")
    results["sweep_calibration"] = sweep
    (rd / "phase1h_gated_benchmark.md").write_text(build_md(results))

    print("\n" + "=" * 74)
    print(f"  gate {chosen_gate:.6f} | stage2 {s2:.6f}")
    print(f"  escalation rate      : {behaviour['escalation_rate']*100:.2f}%   "
          f"(Phase 1H: {p1h['cascade_behaviour']['stage2_escalation_rate']*100:.2f}%)")
    print(f"  negatives skipped by gate: {behaviour['gate_skipped_negatives']*100:.2f}%")
    print(f"  EVAL Recall@1        : {ev_cascade['recall_at_1']:.4f}  (stage1 {ev_stage1['recall_at_1']:.4f})")
    print(f"  EVAL FAR             : {ev_cascade['far']:.6f} ({ev_cascade['false_positives']}/{len(neg_ev)})")
    print(f"  p50 / p95            : {timing['total_ms']['p50']:.1f} / {timing['total_ms']['p95']:.1f} ms"
          f"   (Phase 1H p50 {p1h['timing_ms']['total_cascade_ms']['p50']:.1f})")
    print("  --- criteria ---")
    for k, v in crit.items():
        if k == "all_passed":
            continue
        print(f"    [{'PASS' if v['pass'] else 'FAIL'}] {k}: {v.get('value', v.get('deltas_vs_stage1'))}")
    print(f"  VERDICT: {'ALL PASS — candidate accepted' if crit['all_passed'] else 'FALSIFIED'}")
    print("=" * 74)
    return 0


def _p(v, nd=2):
    return "—" if v is None else f"{v*100:.{nd}f}%"


def build_md(r):
    L = []; A = L.append
    pv, ds, res = r["provenance"], r["dataset"], r["results"]
    beh, crit, tm, td = r["cascade_behaviour"], r["falsification_criteria"], r["timing_ms"], r["threshold_derivation"]
    cs, s1 = res["evaluation_cascade"], res["evaluation_stage1_only"]
    ref = r["phase1h_reference"]
    A("# Phase 1H (gated) — Latency-Reduced Speed Cascade\n")
    A(f"**Recognizer:** `{pv['recognizer_version']}`  ")
    A(f"**Generated:** {pv['generated_utc']}  ")
    A(f"**Repo commit:** `{pv['git_commit'][:12]}`  ")
    A(f"**Working tree:** {'DIRTY (' + str(pv['git_dirty_path_count']) + ' paths)' if pv['git_dirty'] else 'clean'}  ")
    A(f"**Phase 1 fingerprint:** `{pv['phase1_source_sha256']}`\n")
    A("> Orchestration only. `fingerprint.py`, `index.py`, `matcher.py`, `decision.py` and\n"
      "> `cascade.py` are all byte-identical; Phase 0/1D/1E/1G/1H reports are untouched.\n")

    A("## Verdict\n")
    A(f"**{'ALL FOUR CRITERIA PASS — candidate accepted' if crit['all_passed'] else 'FALSIFIED'}**\n")
    c1, c2, c3, c4 = (crit["1_speed_recall_ge_60pct"], crit["2_preserved_families_within_-1pp"],
                      crit["3_holdout_far_le_5pct"], crit["4_p50_latency_le_40ms"])
    worst = min(c2["deltas_vs_stage1"].values()) if c2["deltas_vs_stage1"] else 0
    A("| # | Criterion | Target | Measured | Result |")
    A("|---|---|---:|---:|---|")
    A(f"| 1 | Speed recall | ≥ 60% | {_p(c1['value'])} | {'**PASS**' if c1['pass'] else '**FAIL**'} |")
    A(f"| 2 | Clean/noise/codec/filter vs stage 1 | ≥ −1 pp | worst {worst*100:+.2f} pp | {'**PASS**' if c2['pass'] else '**FAIL**'} |")
    A(f"| 3 | Held-out FAR | ≤ 5% | {_p(c3['value'],4)} | {'**PASS**' if c3['pass'] else '**FAIL**'} |")
    A(f"| 4 | Whole-corpus p50 latency | ≤ 40 ms | {c4['value']:.2f} ms | {'**PASS**' if c4['pass'] else '**FAIL**'} |")
    A("")

    A("## Threshold derivation (calibration split only)\n")
    A(f"- **Stage 1: {td['stage1_threshold']}** — {td['stage1_origin']}")
    A(f"- **Stage 2: {td['stage2_threshold']:.6f}** — {td['stage2_rule']}")
    A(f"- **Gate: {td['gate_threshold']:.6f}** — {td['gate_rule']}")
    gp = td["gate_calibration_point"]
    A(f"  - at that gate, calibration Recall@1 {_p(gp['recall_at_1'])}, "
      f"escalation {_p(gp['escalation'])}")
    A("\nThe evaluation split was not consulted for either threshold.\n")

    A("## Headline (evaluation split)\n")
    A("| Metric | Stage 1 only | **Gated cascade** | Δ |")
    A("|---|---:|---:|---:|")
    for lab, k in (("Recall@1", "recall_at_1"), ("FAR", "far"), ("Precision", "precision"),
                   ("Correct rejection", "correct_rejection_rate")):
        A(f"| {lab} | {_p(s1[k],4)} | **{_p(cs[k],4)}** | {((cs[k] or 0)-(s1[k] or 0))*100:+.2f} pp |")
    for lab, k in (("TP", "true_positives"), ("FP", "false_positives"),
                   ("TN", "true_negatives"), ("FN", "false_negatives")):
        A(f"| {lab} | {s1[k]} | **{cs[k]}** | {cs[k]-s1[k]:+d} |")
    A("")

    A("## Against ungated Phase 1H\n")
    A("| | Phase 1H (ungated) | **Phase 1H (gated)** |")
    A("|---|---:|---:|")
    A(f"| Escalation rate | {_p(ref['escalation_rate'])} | **{_p(beh['escalation_rate'])}** |")
    A(f"| p50 latency | {ref['p50_ms']:.1f} ms | **{tm['total_ms']['p50']:.1f} ms** |")
    A(f"| Recall@1 | {_p(ref['evaluation']['recall_at_1'])} | **{_p(cs['recall_at_1'])}** |")
    A(f"| FAR | {_p(ref['evaluation']['far'],4)} | **{_p(cs['far'],4)}** |")
    A("")

    A("## Cascade behaviour\n")
    A(f"- Stage-1 match rate: **{_p(beh['stage1_match_rate'])}**")
    A(f"- Gate pass / escalation rate: **{_p(beh['escalation_rate'])}**")
    A(f"- Negatives skipped by the gate: **{_p(beh['gate_skipped_negatives'])}**")
    A(f"- Probe pass rate: **{_p(beh['probe_pass_rate'])}**; stage-2 acceptances: "
      f"**{beh['stage2_acceptances']}**")
    if beh["winning_rate_histogram"]:
        A("\n| Winning correction | Acceptances |")
        A("|---|---:|")
        for k, v in beh["winning_rate_histogram"].items():
            A(f"| {k} | {v} |")
    A("")

    A("## Per-family: stage 1 vs gated cascade\n")
    A("| Family | Queries | Stage 1 | Cascade | Δ |")
    A("|---|---:|---:|---:|---:|")
    for k, v in r["by_family"].items():
        A(f"| {k} | {v['queries']} | {_p(v['stage1_recall_at_1'])} | "
          f"**{_p(v['cascade_recall_at_1'])}** | {v['delta']*100:+.2f} pp |")
    A("")
    A("## Per-condition\n")
    A("| Condition | n | Stage 1 | Cascade | Δ |")
    A("|---|---:|---:|---:|---:|")
    for c in r["by_condition"]:
        A(f"| `{c['condition']}` | {c['queries']} | {_p(c['stage1_recall_at_1'])} | "
          f"{_p(c['cascade_recall_at_1'])} | {c['delta']*100:+.2f} pp |")
    A("")

    A("## Rejection\n")
    A(f"Evaluation negatives: **{ds['evaluation_negatives']}** excerpts from "
      f"**{ds['evaluation_negative_source_recordings']}** source recordings; one false "
      f"accept ≈ {100.0/max(1,ds['evaluation_negatives']):.4f} pp. Excerpt count is not "
      f"statistical sample size.\n")
    A("| Category | Negatives | False accepts | FAR |")
    A("|---|---:|---:|---:|")
    for k, v in res["far_by_category_evaluation"].items():
        A(f"| `{k.replace('negative_','')}` | {v['negatives']} | {v['false_positives']} | {_p(v['far'],4)} |")
    A("")
    if res["false_accepts"]:
        A("| Query | Category | Stage | Correction | Matched |")
        A("|---|---|---:|---:|---|")
        for f in res["false_accepts"][:40]:
            A(f"| `{f['query_id'][:38]}` | {f['category'].replace('negative_','')} | {f['stage']} | "
              f"{f['rate']:+g}% | `{(f['matched'] or '-')[:24]}` |")
        A("")

    A("## Latency (wall-clock, real gated run)\n")
    A("| Stage | p50 ms | p95 ms |")
    A("|---|---:|---:|")
    A(f"| stage 1 (every query) | {tm['stage1_ms']['p50']:.2f} | {tm['stage1_ms']['p95']:.2f} |")
    pr, cf = tm["probe_ms_when_gate_passed"], tm["confirm_ms_when_probe_passed"]
    A(f"| probe (gate passed) | {pr['p50']:.2f} | {pr['p95']:.2f} |")
    A(f"| confirm (probe passed) | {cf['p50']:.2f} | {cf['p95']:.2f} |")
    t = tm["total_ms"]
    A(f"| **total** | **{t['p50']:.2f}** | **{t['p95']:.2f}** |")
    A(f"\np99 {t['p99']:.2f} ms, mean {t['mean']:.2f} ms.\n")
    A("## Limitations\n")
    for x in r["limitations"]:
        A(f"- {x}")
    A("")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
