"""Gated speed-tolerant cascade -- the latency-reduced successor to `cascade.py`.

WHY THIS EXISTS
---------------
Phase 1H passed three of four acceptance criteria and failed p50 latency at
144.5 ms against a 40 ms bar. The investigation established the shape of the
problem precisely:

  * stage 1 alone is ~26 ms, i.e. 65% of the whole budget;
  * a hypothesis costs ~24.6 ms, of which resampling is 2.5% -- matching is 63%
    and fingerprinting 33%, so there is no cheap part to shave;
  * escalation was 55.33%, so the MEDIAN query ran the sweep;
  * consequently even a ONE-hypothesis sweep projects to 50.7 ms and still fails.

Grid reduction therefore cannot fix criterion 4. The only lever that can is
pushing escalation below 50% so the median query never reaches stage 2 -- which
needs roughly 20% of negatives skipped before the sweep begins.

THE PIPELINE
------------
    stage 1 (full query, frozen recognizer)
      |-- MATCH -> return immediately, nothing else runs
      v NO_MATCH
    concentration gate           free: reuses stage 1's own candidate
      |-- fail -> NO_MATCH, no probe, no sweep
      v pass
    2 s probe over (-4, -2, +2, +4)      ~3x cheaper per hypothesis
      |-- no usable candidate -> NO_MATCH, no confirmation
      v pass
    full-query confirmation at the winning rate
      v
    stage-2 decision

WHY CONCENTRATION, AND ONLY CONCENTRATION
------------------------------------------
Of the free stage-1 statistics, three were at or below chance at separating
speed-recoverable queries from out-of-catalog music (evidence AUC 0.458,
aligned 0.427, hits-per-landmark 0.343 -- negatives frequently score HIGHER).
Only concentration carried signal, at AUC 0.639. That weakness is principled: a
speed-shifted recording keeps only ~1-3% of its hashes, which is what incidental
collisions produce anyway, so stage 1 has almost nothing to see. A weak gate is
nonetheless sufficient, because only ~20% of negatives need skipping.

WHY THE GRID IS (-4, -2, +2, +4)
---------------------------------
(-5,-2,+2,+5) tied the full 10-point grid on the benchmark, but the corpus only
contains +/-2% and +/-5% speed changes, so that parity was fitted to the test
set. Against off-grid speeds it collapses -- 1/6 recovered at +3.5%, where
(-4,-2,+2,+4) recovers 6/6. The grid here is the one that generalises, not the
one that scored best.

Nothing in fingerprint.py, index.py, matcher.py or decision.py is touched, and
`cascade.py` is left intact so the frozen Phase 1H benchmark stays reproducible.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from musicintel.recognition.cascade import RateHypothesis, apply_rate
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

# Fixed from the off-grid generalisation experiment, before any evaluation data
# was consulted. Four points at 2% spacing cover +/-5% with <=1% residual, which
# is inside the measured correction tolerance.
GATED_RATE_GRID: tuple[float, ...] = (-4.0, -2.0, 2.0, 4.0)
PROBE_SECONDS: float = 2.0


@dataclass(frozen=True)
class GatedCascadeConfig:
    """Two calibrated thresholds, both derived on the calibration split only.

    `gate_threshold` and `stage2_threshold` are placeholders here; the measured
    values live in eval/reports/phase1h_gated_benchmark.md. The probe carries no
    tuned threshold of its own on purpose -- it only asks whether any usable
    candidate exists, so the calibration stays two-dimensional and the reported
    trade-off means what it says.
    """

    rate_grid: tuple[float, ...] = GATED_RATE_GRID
    stage1_threshold: float = 0.026316  # frozen Phase 1G/1H operating point
    gate_threshold: float = 0.0  # calibrated; 0.0 = gate open
    probe_seconds: float = PROBE_SECONDS
    stage2_threshold: float = 0.10  # calibrated
    min_aligned_landmarks: int = 5

    def validate(self) -> None:
        for name in ("stage1_threshold", "stage2_threshold", "gate_threshold"):
            v = getattr(self, name)
            if not 0.0 <= v <= 1.0:
                raise ValueError(f"{name} must be a rate in [0, 1]")
        if self.min_aligned_landmarks < 1:
            raise ValueError("min_aligned_landmarks must be >= 1")
        if self.probe_seconds <= 0:
            raise ValueError("probe_seconds must be positive")
        if 0.0 in self.rate_grid:
            raise ValueError("rate_grid must not contain 0 -- stage 1 covers it")
        if len(set(self.rate_grid)) != len(self.rate_grid):
            raise ValueError("rate_grid contains duplicates")

    @property
    def stage1_decision_config(self) -> DecisionConfig:
        return DecisionConfig(threshold=self.stage1_threshold,
                              min_aligned_landmarks=self.min_aligned_landmarks)


DEFAULT_GATED_CONFIG = GatedCascadeConfig()


@dataclass(frozen=True)
class GatedTiming:
    stage1_ms: float
    probe_ms: float
    confirm_ms: float
    hypotheses_evaluated: int

    @property
    def total_ms(self) -> float:
        return self.stage1_ms + self.probe_ms + self.confirm_ms


@dataclass(frozen=True)
class GatedResult:
    decision: Decision
    track_id: str | None
    stage: int | None
    rate_percent: float
    evidence_score: float
    threshold: float
    aligned_landmarks: int
    query_landmarks: int
    best_offset: int | None
    gate_value: float
    gate_passed: bool
    probe_passed: bool
    escalated: bool
    stage1_decision: MatchDecision
    probe_hypotheses: tuple[RateHypothesis, ...] = field(default_factory=tuple)
    timing: GatedTiming = field(default_factory=lambda: GatedTiming(0.0, 0.0, 0.0, 0))

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
            "gate_value": round(self.gate_value, 6),
            "gate_passed": self.gate_passed, "probe_passed": self.probe_passed,
            "escalated": self.escalated,
            "hypotheses_evaluated": self.timing.hypotheses_evaluated,
        }


def gate_value_of(d: MatchDecision) -> float:
    """The free stage-1 statistic the gate reads: top candidate concentration.

    Concentration is the fraction of a track's matched landmarks that agreed on
    one time offset. Computed by the matcher already, so reading it costs nothing.
    """
    return d.candidates[0].concentration if d.candidates else 0.0


def _rank(h: RateHypothesis) -> tuple:
    """Best evidence; ties toward the smallest correction, then the smaller
    signed rate, so the winner is a total order and reproducible."""
    return (-h.evidence, abs(h.rate_percent), h.rate_percent)


def _reject(cfg, d1, gate_v, gate_ok, probe_ok, hyps, timing, esc) -> GatedResult:
    """NO_MATCH withholds the winning track, as the stage-1 decision layer does."""
    return GatedResult(
        decision=Decision.NO_MATCH, track_id=None, stage=None, rate_percent=0.0,
        evidence_score=d1.evidence_score, threshold=cfg.stage2_threshold,
        aligned_landmarks=d1.aligned_landmarks,
        query_landmarks=d1.query_landmark_count, best_offset=None,
        gate_value=gate_v, gate_passed=gate_ok, probe_passed=probe_ok,
        escalated=esc, stage1_decision=d1, probe_hypotheses=hyps, timing=timing,
    )


def identify_gated(
    y: np.ndarray,
    sr: int,
    index: FingerprintIndex,
    *,
    config: GatedCascadeConfig | None = None,
    match_config: MatchConfig | None = None,
    fingerprint_config: FingerprintConfig | None = None,
) -> GatedResult:
    """Stage 1, then gate, then a cheap probe sweep, then a full-query confirm.

    Each step can only stop the pipeline, never restart it, so a stage-1 MATCH
    costs exactly what the frozen recognizer costs and nothing more.
    """
    cfg = config or DEFAULT_GATED_CONFIG
    cfg.validate()
    fp_cfg = fingerprint_config or index.config

    # -- stage 1 -----------------------------------------------------------
    t0 = time.perf_counter()
    q0 = fingerprint(y, sr, fp_cfg)
    d1 = decide(match(q0, index, config=match_config),
                config=cfg.stage1_decision_config)
    t_s1 = (time.perf_counter() - t0) * 1000.0

    if d1.is_match:
        return GatedResult(
            decision=Decision.MATCH, track_id=d1.track_id, stage=1, rate_percent=0.0,
            evidence_score=d1.evidence_score, threshold=cfg.stage1_threshold,
            aligned_landmarks=d1.aligned_landmarks,
            query_landmarks=d1.query_landmark_count, best_offset=d1.best_offset,
            gate_value=gate_value_of(d1), gate_passed=False, probe_passed=False,
            escalated=False, stage1_decision=d1,
            timing=GatedTiming(t_s1, 0.0, 0.0, 1),
        )

    # -- concentration gate (free) -----------------------------------------
    gate_v = gate_value_of(d1)
    if gate_v < cfg.gate_threshold:
        return _reject(cfg, d1, gate_v, False, False, (),
                       GatedTiming(t_s1, 0.0, 0.0, 1), esc=False)

    # -- 2 s probe sweep ----------------------------------------------------
    t1 = time.perf_counter()
    n_probe = int(round(cfg.probe_seconds * sr))
    probe = y[:n_probe] if y.size > n_probe else y
    hyps: list[RateHypothesis] = []
    for rate in cfg.rate_grid:
        try:
            pp = fingerprint(apply_rate(probe, sr, rate), sr, fp_cfg)
            r = match(pp, index, config=match_config)
            top = r.best
            hyps.append(RateHypothesis(
                rate_percent=rate, track_id=top.track_id if top else None,
                aligned=top.score if top else 0, query_landmarks=len(pp),
                evidence=(top.score / len(pp)) if (top and len(pp)) else 0.0,
                best_offset=top.best_offset if top else None))
        except Exception as e:  # noqa: BLE001 -- one bad rate must not sink the query
            hyps.append(RateHypothesis(rate_percent=rate, track_id=None, aligned=0,
                                       query_landmarks=0, evidence=0.0,
                                       error=f"{type(e).__name__}: {e}"))
    t_probe = (time.perf_counter() - t1) * 1000.0
    hyps_t = tuple(hyps)

    usable = [h for h in hyps if h.ok and h.track_id
              and h.aligned >= cfg.min_aligned_landmarks]
    if not usable:
        return _reject(cfg, d1, gate_v, True, False, hyps_t,
                       GatedTiming(t_s1, t_probe, 0.0, 1 + len(hyps)), esc=True)
    best_probe = min(usable, key=_rank)

    # -- full-query confirmation at the probe's winning rate ----------------
    t2 = time.perf_counter()
    try:
        qc = fingerprint(apply_rate(y, sr, best_probe.rate_percent), sr, fp_cfg)
        rc = match(qc, index, config=match_config)
        top = rc.best
        conf_ev = (top.score / len(qc)) if (top and len(qc)) else 0.0
        conf_aligned = top.score if top else 0
        conf_id = top.track_id if top else None
        conf_off = top.best_offset if top else None
        conf_lm = len(qc)
    except Exception:  # noqa: BLE001
        conf_ev, conf_aligned, conf_id, conf_off, conf_lm = 0.0, 0, None, None, 0
    t_conf = (time.perf_counter() - t2) * 1000.0
    timing = GatedTiming(t_s1, t_probe, t_conf, 2 + len(hyps))

    if (conf_id is None or conf_ev < cfg.stage2_threshold
            or conf_aligned < cfg.min_aligned_landmarks):
        return _reject(cfg, d1, gate_v, True, True, hyps_t, timing, esc=True)

    return GatedResult(
        decision=Decision.MATCH, track_id=conf_id, stage=2,
        rate_percent=best_probe.rate_percent, evidence_score=conf_ev,
        threshold=cfg.stage2_threshold, aligned_landmarks=conf_aligned,
        query_landmarks=conf_lm, best_offset=conf_off, gate_value=gate_v,
        gate_passed=True, probe_passed=True, escalated=True,
        stage1_decision=d1, probe_hypotheses=hyps_t, timing=timing,
    )


def identify_gated_file(path: str | Path, index: FingerprintIndex, **kw) -> GatedResult:
    """Decode a file and run the gated cascade over it."""
    fp_cfg = kw.get("fingerprint_config") or index.config
    y, sr = load_audio(path, fp_cfg)
    return identify_gated(y, sr, index, **kw)


__all__ = [
    "DEFAULT_GATED_CONFIG", "GATED_RATE_GRID", "PROBE_SECONDS",
    "GatedCascadeConfig", "GatedResult", "GatedTiming", "gate_value_of",
    "identify_gated", "identify_gated_file",
]
