"""Speed-tolerant recognition cascade.

WHAT THIS IS
------------
An ORCHESTRATION layer. It contains no DSP, no hashing, no matching and no
scoring of its own: it calls the frozen Phase 1 pipeline
(fingerprint -> index -> matcher -> decision) and decides how many times to call
it. Every number it reports is produced by that pipeline unmodified.

THE PROBLEM
-----------
Phase 1E measured speed conditions at 3.91%. A playback-rate change scales BOTH
axes -- frequency bins move, and so do frame indices -- so the exact-integer hash
breaks and, separately, the time offset stops being constant:

    offset = ref_anchor - query_anchor = query_anchor * (r - 1) + start

That is linear in the anchor, not flat, so even a rate-invariant key would not
spike in the offset histogram. Phase 1F established the way out: resampling the
QUERY inverts both axes at once, restoring the original fingerprint exactly and
leaving the matcher's constant-offset assumption intact.

RATE CONVENTION -- verified, not assumed
----------------------------------------
`apply_rate(y, sr, p)` plays the audio back at (1 + p/100)x:

    p > 0  ->  FASTER and HIGHER   (4.000 s -> 3.922 s, 1000 Hz -> 1020 Hz at +2%)
    p < 0  ->  SLOWER and LOWER    (4.000 s -> 4.082 s, 1000 Hz ->  980 Hz at -2%)

So a recording captured at +2% is corrected by applying about -2%. Exactly it is
-(p/(100+p))*100 = -1.9608%, but the grid is integral and the tolerance is about
+/-1%, so -2% is comfortably inside it. `test_cascade.py` pins the sign.

WHY A CASCADE AND NOT A SWEEP ON EVERY QUERY
--------------------------------------------
Stage 1 already recognizes clean, noisy, codec-compressed and band-limited audio.
Running ten extra hypotheses on those queries would multiply latency and, worse,
multiply false-accept exposure: each hypothesis is another chance to cross a
threshold. So the sweep runs ONLY when stage 1 declines, and stage 2 carries its
own, separately calibrated threshold.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from musicintel.recognition.decision import (
    Decision,
    DecisionConfig,
    MatchDecision,
    decide,
)
from musicintel.recognition.fingerprint import (
    FingerprintConfig,
    fingerprint,
    load_audio,
)
from musicintel.recognition.index import FingerprintIndex
from musicintel.recognition.matcher import MatchConfig, match

# Fixed in advance, before any evaluation data was looked at: +/-5% at 1% steps.
# Phase 1F measured the correction tolerance at roughly +/-1%, so 1% spacing
# leaves every true rate in the range within half a step of a grid point.
DEFAULT_RATE_GRID: tuple[float, ...] = (
    -5.0, -4.0, -3.0, -2.0, -1.0, 1.0, 2.0, 3.0, 4.0, 5.0,
)


def apply_rate(y: np.ndarray, sr: int, percent: float) -> np.ndarray:
    """Play `y` back at (1 + percent/100)x. Positive = faster and higher.

    Resampling to sr/(1+p/100) and then treating the result as still being at
    `sr` is what makes this a playback-rate change rather than a pitch shift:
    frequency and duration move together, which is precisely the transformation
    a tape, a vinyl deck or a mis-clocked broadcast applies.
    """
    y = np.asarray(y, dtype=np.float32)
    if percent == 0 or y.size == 0:
        return y
    import librosa

    factor = 1.0 + percent / 100.0
    if factor <= 0:
        raise ValueError(f"rate {percent}% is not a playable speed")
    target = int(round(sr / factor))
    if target <= 0:
        raise ValueError(f"rate {percent}% collapses the sample rate")
    return librosa.resample(y, orig_sr=sr, target_sr=target).astype(np.float32)


@dataclass(frozen=True)
class CascadeConfig:
    """Cascade parameters. Stage 1 and stage 2 have SEPARATE thresholds.

    Reusing one threshold for both would be wrong: stage 2 gets ten extra
    attempts per query, so the same bar admits far more false accepts. The two
    are calibrated independently, both on the calibration split only.
    """

    rate_grid: tuple[float, ...] = DEFAULT_RATE_GRID
    # Stage 1 is the frozen, already-calibrated Phase 1G operating point.
    stage1_threshold: float = 0.026316
    # Stage 2 is calibrated separately; this default is a placeholder, and the
    # measured value lives in eval/reports/phase1h_benchmark.md.
    stage2_threshold: float = 0.10
    min_aligned_landmarks: int = 5

    def validate(self) -> None:
        if not 0.0 <= self.stage1_threshold <= 1.0:
            raise ValueError("stage1_threshold must be a rate in [0, 1]")
        if not 0.0 <= self.stage2_threshold <= 1.0:
            raise ValueError("stage2_threshold must be a rate in [0, 1]")
        if self.min_aligned_landmarks < 1:
            raise ValueError("min_aligned_landmarks must be >= 1")
        if 0.0 in self.rate_grid:
            # Rate 0 is stage 1's result; re-running it would be duplicated work.
            raise ValueError("rate_grid must not contain 0 -- stage 1 covers it")
        if len(set(self.rate_grid)) != len(self.rate_grid):
            raise ValueError("rate_grid contains duplicates")

    @property
    def stage1_decision_config(self) -> DecisionConfig:
        return DecisionConfig(threshold=self.stage1_threshold,
                              min_aligned_landmarks=self.min_aligned_landmarks)


DEFAULT_CASCADE_CONFIG = CascadeConfig()


@dataclass(frozen=True)
class RateHypothesis:
    """One rate correction and what the frozen pipeline made of it."""

    rate_percent: float
    track_id: str | None
    aligned: int
    query_landmarks: int
    evidence: float
    best_offset: int | None = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


@dataclass(frozen=True)
class CascadeTiming:
    stage1_ms: float
    stage2_ms: float
    hypotheses_evaluated: int

    @property
    def total_ms(self) -> float:
        return self.stage1_ms + self.stage2_ms


@dataclass(frozen=True)
class CascadeResult:
    """Verdict plus the evidence and the path taken to it."""

    decision: Decision
    track_id: str | None
    stage: int | None  # 1 or 2 when matched; None when rejected
    rate_percent: float  # winning correction; 0.0 for a stage-1 match
    evidence_score: float
    threshold: float
    aligned_landmarks: int
    query_landmarks: int
    best_offset: int | None
    escalated: bool
    stage1_decision: MatchDecision
    hypotheses: tuple[RateHypothesis, ...] = field(default_factory=tuple)
    timing: CascadeTiming = field(
        default_factory=lambda: CascadeTiming(0.0, 0.0, 0)
    )

    @property
    def is_match(self) -> bool:
        return self.decision is Decision.MATCH

    def to_dict(self) -> dict:
        return {
            "decision": self.decision.value, "track_id": self.track_id,
            "stage": self.stage, "rate_percent": self.rate_percent,
            "evidence_score": round(self.evidence_score, 6),
            "threshold": self.threshold,
            "aligned_landmarks": self.aligned_landmarks,
            "query_landmarks": self.query_landmarks,
            "best_offset": self.best_offset, "escalated": self.escalated,
            "hypotheses_evaluated": self.timing.hypotheses_evaluated,
        }


def _hypothesis_from_decision(d: MatchDecision, rate: float) -> RateHypothesis:
    """Reuse stage 1's already-computed result as the rate-0 hypothesis."""
    top = d.candidates[0] if d.candidates else None
    return RateHypothesis(
        rate_percent=rate,
        track_id=top.track_id if top else None,
        aligned=top.score if top else 0,
        query_landmarks=d.query_landmark_count,
        evidence=d.evidence_score,
        best_offset=top.best_offset if top else None,
    )


def _rank_key(h: RateHypothesis) -> tuple:
    """Best evidence wins; ties break toward the smallest correction, then the
    smaller signed rate, so the choice is total and reproducible."""
    return (-h.evidence, abs(h.rate_percent), h.rate_percent)


def identify_cascade(
    y: np.ndarray,
    sr: int,
    index: FingerprintIndex,
    *,
    config: CascadeConfig | None = None,
    match_config: MatchConfig | None = None,
    fingerprint_config: FingerprintConfig | None = None,
) -> CascadeResult:
    """Stage 1, then a query-side rate sweep only if stage 1 declines.

    A stage-1 MATCH short-circuits: no resampling, no extra hypotheses, and the
    latency and false-accept exposure of the frozen recognizer are unchanged.
    """
    cfg = config or DEFAULT_CASCADE_CONFIG
    cfg.validate()
    fp_cfg = fingerprint_config or index.config

    # -- stage 1: the frozen recognizer, exactly as Phase 1G measured it ----
    t0 = time.perf_counter()
    q0 = fingerprint(y, sr, fp_cfg)
    d1 = decide(match(q0, index, config=match_config),
                config=cfg.stage1_decision_config)
    t_stage1 = (time.perf_counter() - t0) * 1000.0

    if d1.is_match:
        return CascadeResult(
            decision=Decision.MATCH, track_id=d1.track_id, stage=1, rate_percent=0.0,
            evidence_score=d1.evidence_score, threshold=cfg.stage1_threshold,
            aligned_landmarks=d1.aligned_landmarks,
            query_landmarks=d1.query_landmark_count, best_offset=d1.best_offset,
            escalated=False, stage1_decision=d1,
            hypotheses=(_hypothesis_from_decision(d1, 0.0),),
            timing=CascadeTiming(t_stage1, 0.0, 1),
        )

    # -- stage 2: rate hypotheses -------------------------------------------
    t1 = time.perf_counter()
    hyps: list[RateHypothesis] = [_hypothesis_from_decision(d1, 0.0)]
    for rate in cfg.rate_grid:
        try:
            yy = apply_rate(y, sr, rate)
            qq = fingerprint(yy, sr, fp_cfg)
            r = match(qq, index, config=match_config)
            top = r.best
            hyps.append(RateHypothesis(
                rate_percent=rate,
                track_id=top.track_id if top else None,
                aligned=top.score if top else 0,
                query_landmarks=len(qq),
                evidence=(top.score / len(qq)) if (top and len(qq)) else 0.0,
                best_offset=top.best_offset if top else None,
            ))
        except Exception as e:  # noqa: BLE001 -- one bad rate must not sink the query
            hyps.append(RateHypothesis(
                rate_percent=rate, track_id=None, aligned=0, query_landmarks=0,
                evidence=0.0, error=f"{type(e).__name__}: {e}",
            ))
    t_stage2 = (time.perf_counter() - t1) * 1000.0

    usable = [h for h in hyps if h.ok and h.track_id is not None]
    timing = CascadeTiming(t_stage1, t_stage2, len(hyps))
    if not usable:
        return _reject(cfg, d1, tuple(hyps), timing)

    best = min(usable, key=_rank_key)
    accepted = (
        best.evidence >= cfg.stage2_threshold
        and best.aligned >= cfg.min_aligned_landmarks
    )
    if not accepted:
        return _reject(cfg, d1, tuple(hyps), timing, best=best)

    return CascadeResult(
        decision=Decision.MATCH, track_id=best.track_id, stage=2,
        rate_percent=best.rate_percent, evidence_score=best.evidence,
        threshold=cfg.stage2_threshold, aligned_landmarks=best.aligned,
        query_landmarks=best.query_landmarks, best_offset=best.best_offset,
        escalated=True, stage1_decision=d1, hypotheses=tuple(hyps), timing=timing,
    )


def _reject(cfg, d1, hyps, timing, best=None) -> CascadeResult:
    """NO_MATCH. The winning track is withheld so a rejected hypothesis can
    never be read as an answer, exactly as the stage-1 decision layer does."""
    return CascadeResult(
        decision=Decision.NO_MATCH, track_id=None, stage=None,
        rate_percent=best.rate_percent if best else 0.0,
        evidence_score=best.evidence if best else d1.evidence_score,
        threshold=cfg.stage2_threshold,
        aligned_landmarks=best.aligned if best else d1.aligned_landmarks,
        query_landmarks=best.query_landmarks if best else d1.query_landmark_count,
        best_offset=None, escalated=True, stage1_decision=d1,
        hypotheses=hyps, timing=timing,
    )


def identify_cascade_file(
    path: str | Path,
    index: FingerprintIndex,
    *,
    config: CascadeConfig | None = None,
    match_config: MatchConfig | None = None,
    fingerprint_config: FingerprintConfig | None = None,
) -> CascadeResult:
    """Decode a file and run the cascade over it."""
    fp_cfg = fingerprint_config or index.config
    y, sr = load_audio(path, fp_cfg)
    return identify_cascade(y, sr, index, config=config, match_config=match_config,
                            fingerprint_config=fp_cfg)


__all__ = [
    "DEFAULT_CASCADE_CONFIG", "DEFAULT_RATE_GRID", "CascadeConfig",
    "CascadeResult", "CascadeTiming", "RateHypothesis", "apply_rate",
    "identify_cascade", "identify_cascade_file",
]
