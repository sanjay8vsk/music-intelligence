"""BPM and key metrics, with the reporting policy the roadmap demands.

DENOMINATOR POLICY -- the thing most accuracy numbers get wrong
--------------------------------------------------------------
Three populations, kept apart:

* **evaluable** -- ground truth exists and the detector returned an answer.
* **excluded** -- ground truth says there is no stable tempo / no tonal centre.
  Removed from the denominator entirely and reported by name and count. They are
  not failures: the correct answer is "there isn't one".
* **failed** -- the detector raised, or returned nothing where an answer was
  expected. Counted in the denominator **as wrong**. A crash is not an excuse.

Accuracy is over `evaluable + failed`. Silently dropping failures would let a
detector improve its score by crashing on hard material.

WHAT IS NEVER REPORTED
----------------------
A bare key accuracy, or a BPM accuracy that has quietly absorbed octave errors.
`evaluate_bpm` returns raw and octave-tolerant as separate fields and the caller
cannot collapse them by accident; `evaluate_key` always carries the confusion
matrix and the relation breakdown alongside the headline.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from musicintel.analysis.keys import (
    ALL_KEYS, KEY_LABELS, Key, mirex_score, parse_key, relation,
)

BPM_TOLERANCE = 0.02              # |pred - truth| / truth
OCTAVE_FACTORS = (0.5, 1.0, 2.0)
TRIPLE_FACTORS = (1.0 / 3.0, 3.0)

NO_TEMPO = "no_stable_tempo"
NO_KEY = "no_tonal_centre"


@dataclass(frozen=True)
class Prediction:
    """One detector answer against one ground truth.

    `truth` is None when the reference says the quantity is undefined, which is
    an exclusion. `predicted` is None when the detector failed, which is not.
    """

    track_id: str
    truth: float | str | None
    predicted: float | str | None
    error: str | None = None          # detector exception text, if any


def _relative_error(truth: float, predicted: float) -> float:
    return abs(predicted - truth) / truth


def _within(truth: float, predicted: float, tol: float = BPM_TOLERANCE) -> bool:
    return _relative_error(truth, predicted) <= tol


def _percentiles(values: list[float]) -> dict:
    if not values:
        return {}
    import numpy as np
    a = np.asarray(values, dtype=float)
    return {f"p{p}": round(float(np.percentile(a, p)), 5)
            for p in (50, 75, 90, 95, 99)} | {
        "min": round(float(a.min()), 5), "max": round(float(a.max()), 5),
        "mean": round(float(a.mean()), 5)}


def evaluate_bpm(predictions: list[Prediction], *,
                 tolerance: float = BPM_TOLERANCE) -> dict:
    """BPM metrics. Raw and octave-tolerant are separate and stay separate."""
    excluded = [p for p in predictions if p.truth is None or p.truth == NO_TEMPO]
    scored = [p for p in predictions if p not in excluded]
    failed = [p for p in scored if p.predicted is None]
    usable = [p for p in scored if p.predicted is not None]

    raw_hits = 0
    octave_hits = 0
    triple_hits = 0
    double_errors = 0
    half_errors = 0
    rel_errors: list[float] = []
    per_track: list[dict] = []

    for p in usable:
        truth, pred = float(p.truth), float(p.predicted)
        rel = _relative_error(truth, pred)
        rel_errors.append(rel)
        raw_ok = _within(truth, pred, tolerance)
        # Octave tolerance compares the PREDICTION against multiples of truth.
        oct_ok = any(_within(truth * f, pred, tolerance) for f in OCTAVE_FACTORS)
        tri_ok = any(_within(truth * f, pred, tolerance) for f in TRIPLE_FACTORS)
        raw_hits += raw_ok
        octave_hits += oct_ok
        triple_hits += tri_ok and not oct_ok
        if not raw_ok:
            if _within(truth * 2.0, pred, tolerance):
                double_errors += 1
            elif _within(truth * 0.5, pred, tolerance):
                half_errors += 1
        per_track.append({"track_id": p.track_id, "truth": truth,
                          "predicted": pred, "relative_error": round(rel, 5),
                          "raw_ok": raw_ok, "octave_ok": oct_ok})

    denominator = len(usable) + len(failed)
    def rate(n: int) -> float | None:
        return round(n / denominator, 4) if denominator else None

    return {
        "tolerance": tolerance,
        "evaluable": len(usable),
        "failed": len(failed),
        "excluded_no_stable_tempo": len(excluded),
        "denominator": denominator,
        # Deliberately two separate numbers. Never add them, never merge them.
        "raw_accuracy": rate(raw_hits),
        "octave_tolerant_accuracy": rate(octave_hits),
        "triple_only_accuracy": rate(triple_hits),
        "double_errors": double_errors,
        "half_errors": half_errors,
        "relative_error_distribution": _percentiles(rel_errors),
        "failures": [{"track_id": p.track_id, "error": p.error} for p in failed],
        "excluded_tracks": [p.track_id for p in excluded],
        "per_track": per_track,
    }


def evaluate_key(predictions: list[Prediction]) -> dict:
    """Key metrics: exact accuracy, MIREX weighting, relations, and a 24x24."""
    excluded = [p for p in predictions if p.truth is None or p.truth == NO_KEY]
    scored = [p for p in predictions if p not in excluded]
    failed = [p for p in scored if p.predicted is None]
    usable = [p for p in scored if p.predicted is not None]

    matrix = [[0] * 24 for _ in range(24)]
    relations = {"exact": 0, "relative": 0, "parallel": 0,
                 "dominant": 0, "subdominant": 0, "other": 0}
    weighted = 0.0
    per_track: list[dict] = []

    for p in usable:
        truth = p.truth if isinstance(p.truth, Key) else parse_key(str(p.truth))
        pred = p.predicted if isinstance(p.predicted, Key) else parse_key(str(p.predicted))
        rel = relation(truth, pred)
        relations[rel] += 1
        score = mirex_score(truth, pred)
        weighted += score
        matrix[truth.index][pred.index] += 1
        per_track.append({"track_id": p.track_id, "truth": str(truth),
                          "predicted": str(pred), "relation": rel,
                          "mirex": score})

    denominator = len(usable) + len(failed)
    return {
        "evaluable": len(usable),
        "failed": len(failed),
        "excluded_no_tonal_centre": len(excluded),
        "denominator": denominator,
        "exact_accuracy": (round(relations["exact"] / denominator, 4)
                           if denominator else None),
        # A failure contributes 0.0, so the weighted score is over the same
        # denominator as the accuracy.
        "mirex_weighted_score": (round(weighted / denominator, 4)
                                 if denominator else None),
        "relation_breakdown": dict(relations),
        "relation_rates": {k: (round(v / denominator, 4) if denominator else None)
                           for k, v in relations.items()},
        "confusion_matrix": matrix,
        "confusion_labels": list(KEY_LABELS),
        "failures": [{"track_id": p.track_id, "error": p.error} for p in failed],
        "excluded_tracks": [p.track_id for p in excluded],
        "per_track": per_track,
    }


def format_confusion(matrix: list[list[int]], *, labels=KEY_LABELS,
                     only_nonzero: bool = True) -> str:
    """Readable 24x24, suppressing all-zero rows so a small run stays legible."""
    lines = []
    for i, row in enumerate(matrix):
        if only_nonzero and not any(row):
            continue
        cells = ", ".join(f"{labels[j]}x{v}" for j, v in enumerate(row) if v)
        lines.append(f"  {labels[i]:<9} -> {cells}")
    return "\n".join(lines) or "  (no evaluable predictions)"


__all__ = [
    "BPM_TOLERANCE", "NO_KEY", "NO_TEMPO", "OCTAVE_FACTORS", "Prediction",
    "evaluate_bpm", "evaluate_key", "format_confusion",
]
