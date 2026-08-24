"""Metrics for recognition evaluation.

Deliberately keeps four families of number apart, because conflating them is how
recognition systems get overstated:

  accuracy   -- Recall@k on queries that DO have a correct answer
  rejection  -- no-match rate, and False Accept Rate on queries that do NOT
  latency    -- mean / p50 / p95, measured with a monotonic clock
  separability -- what FAR would be achievable IF a score threshold existed

The last one matters when scoring a recognizer that has no rejection stage. Such
a system has FAR = 1.0 by construction, which says nothing about whether its
representation could tell a match from a non-match. The threshold sweep answers
that separately.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np


@dataclass
class QueryOutcome:
    """One executed query and what the recognizer did with it."""

    query_id: str
    condition: str
    family: str
    duration: float
    position: str
    is_negative: bool
    latency_ms: float
    returned_ids: list[str] = field(default_factory=list)
    truth_track_id: str | None = None
    top_distance: float | None = None
    error: str | None = None

    @property
    def returned_any(self) -> bool:
        return len(self.returned_ids) > 0

    @property
    def correct_at_1(self) -> bool:
        return bool(self.returned_ids) and self.returned_ids[0] == self.truth_track_id

    @property
    def correct_at_3(self) -> bool:
        return self.truth_track_id in self.returned_ids[:3]

    def to_dict(self) -> dict:
        return asdict(self)


# ------------------------------------------------------------- helpers ----
def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=float), p))


def latency_stats(outcomes: list[QueryOutcome]) -> dict:
    lat = [o.latency_ms for o in outcomes if o.error is None]
    if not lat:
        return {"n": 0, "mean_ms": None, "p50_ms": None, "p95_ms": None}
    return {
        "n": len(lat),
        "mean_ms": round(float(np.mean(lat)), 2),
        "p50_ms": round(percentile(lat, 50), 2),
        "p95_ms": round(percentile(lat, 95), 2),
        "min_ms": round(float(np.min(lat)), 2),
        "max_ms": round(float(np.max(lat)), 2),
    }


# ------------------------------------------------------- core aggregates ----
def summarize(outcomes: list[QueryOutcome]) -> dict:
    """Metrics for one group of queries.

    Positive-only fields are None for negative groups and vice versa, so a
    caller can never accidentally read a recall number off a negative set.
    """
    if not outcomes:
        return {"queries": 0}

    negatives = [o for o in outcomes if o.is_negative]
    positives = [o for o in outcomes if not o.is_negative]
    ok = [o for o in outcomes if o.error is None]

    out: dict = {
        "queries": len(outcomes),
        "errors": len(outcomes) - len(ok),
        **latency_stats(outcomes),
    }

    if positives:
        p_ok = [o for o in positives if o.error is None]
        n = len(p_ok) or 1
        out["recall_at_1"] = round(sum(o.correct_at_1 for o in p_ok) / n, 4)
        out["recall_at_3"] = round(sum(o.correct_at_3 for o in p_ok) / n, 4)
        out["no_match_rate"] = round(sum(not o.returned_any for o in p_ok) / n, 4)
    else:
        out["recall_at_1"] = None
        out["recall_at_3"] = None
        out["no_match_rate"] = None

    if negatives:
        n_ok = [o for o in negatives if o.error is None]
        n = len(n_ok) or 1
        false_accepts = sum(o.returned_any for o in n_ok)
        out["far"] = round(false_accepts / n, 4)
        out["correct_rejection_rate"] = round(1.0 - false_accepts / n, 4)
        out["false_accepts"] = false_accepts
    else:
        out["far"] = None
        out["correct_rejection_rate"] = None

    return out


def group_by(outcomes: list[QueryOutcome], key: str) -> dict[str, list[QueryOutcome]]:
    groups: dict[str, list[QueryOutcome]] = {}
    for o in outcomes:
        groups.setdefault(str(getattr(o, key)), []).append(o)
    return groups


def by_condition_and_duration(outcomes: list[QueryOutcome]) -> list[dict]:
    """Condition x duration breakdown -- the primary reporting table."""
    buckets: dict[tuple[str, float], list[QueryOutcome]] = {}
    for o in outcomes:
        buckets.setdefault((o.condition, o.duration), []).append(o)
    rows = []
    for (cond, dur), group in sorted(buckets.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        row = {"condition": f"{cond}_duration_{dur:g}s", "base_condition": cond,
               "duration": dur, "family": group[0].family}
        row.update(summarize(group))
        rows.append(row)
    return rows


# ------------------------------------------------------ threshold sweep ----
def threshold_sweep(outcomes: list[QueryOutcome], n_points: int = 200) -> dict:
    """Would a score threshold have separated matches from non-matches?

    Answers a question the as-implemented FAR cannot: is the failure a missing
    rejection stage, or a representation that cannot discriminate at all?

    Accept a query when top_distance <= tau. Correct only if the top-1 id is
    also right. Requires distances on both positives and negatives.
    """
    pos = [
        o
        for o in outcomes
        if not o.is_negative and o.error is None and o.top_distance is not None
    ]
    neg = [
        o
        for o in outcomes
        if o.is_negative and o.error is None and o.top_distance is not None
    ]
    if not pos or not neg:
        return {"available": False, "reason": "need scored positives and negatives"}

    all_d = sorted({o.top_distance for o in pos} | {o.top_distance for o in neg})
    if len(all_d) > n_points:
        idx = np.linspace(0, len(all_d) - 1, n_points).astype(int)
        taus = [all_d[i] for i in idx]
    else:
        taus = all_d

    pos_d = np.array([o.top_distance for o in pos])
    pos_ok = np.array([o.correct_at_1 for o in pos])
    neg_d = np.array([o.top_distance for o in neg])

    curve = []
    for tau in taus:
        accepted_pos = pos_d <= tau
        recall = float(np.sum(accepted_pos & pos_ok) / len(pos))
        far = float(np.sum(neg_d <= tau) / len(neg))
        curve.append({"tau": round(float(tau), 4), "recall_at_1": round(recall, 4),
                      "far": round(far, 4)})

    def best_recall_at_far(limit: float) -> dict | None:
        ok = [c for c in curve if c["far"] <= limit]
        return max(ok, key=lambda c: c["recall_at_1"]) if ok else None

    # Distance distributions -- overlap here is the direct evidence.
    corr = pos_d[pos_ok] if pos_ok.any() else np.array([])
    return {
        "available": True,
        "n_positive": len(pos),
        "n_negative": len(neg),
        "operating_points": {
            "far_le_0.001": best_recall_at_far(0.001),
            "far_le_0.01": best_recall_at_far(0.01),
            "far_le_0.05": best_recall_at_far(0.05),
            "far_le_0.10": best_recall_at_far(0.10),
        },
        "distance_distribution": {
            "correct_positive": _dist_summary(corr),
            "all_positive": _dist_summary(pos_d),
            "negative": _dist_summary(neg_d),
        },
        "curve": curve,
    }


def _dist_summary(a: np.ndarray) -> dict | None:
    if a is None or len(a) == 0:
        return None
    return {
        "n": int(len(a)),
        "min": round(float(np.min(a)), 4),
        "p05": round(float(np.percentile(a, 5)), 4),
        "median": round(float(np.median(a)), 4),
        "p95": round(float(np.percentile(a, 95)), 4),
        "max": round(float(np.max(a)), 4),
    }


def worst_and_best(rows: list[dict], min_queries: int = 10) -> tuple[list[dict], list[dict]]:
    """Rank measured positive conditions by Recall@1."""
    scored = [
        r
        for r in rows
        if r.get("recall_at_1") is not None and r.get("queries", 0) >= min_queries
    ]
    ranked = sorted(scored, key=lambda r: (r["recall_at_1"], r["recall_at_3"]))
    return ranked[:8], list(reversed(ranked[-8:]))
