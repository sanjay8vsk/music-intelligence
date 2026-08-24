"""Match decision: turning ranked candidates into MATCH or NO_MATCH.

THE PROBLEM THIS EXISTS TO FIX
------------------------------
The Phase 0 recognizer had a False Accept Rate of 1.0. It returned a catalog
track for speech, for silence and for pure noise, because it had no rejection
stage at all -- it always answered. That single property made it unusable
regardless of its recall, and eval/reports/baseline.md says so plainly.

Phase 1C fixed the ranking but deliberately kept that gap open: it always
returns whatever ranked best. This module closes it. It is the first place in
the system allowed to say "none of these".

SCORE IS NOT PROBABILITY
------------------------
There are three distinct things here and conflating them is how systems come to
report a confident-looking 0.97 that means nothing:

    raw evidence      aligned landmark count from the matcher (Phase 1C)
        |             -- an integer, scales with query length
        v
    decision score    aligned landmarks / query landmarks
        |             -- a RATE in [0, 1], comparable across query lengths
        v
    threshold         a number chosen from measured data
        |
        v
    MATCH / NO_MATCH

The decision score is a **rate**, not a probability. It is bounded by [0, 1]
because it is a fraction of landmarks, not because it is calibrated against
outcome frequencies. A score of 0.4 does not mean "40% likely to be correct".
Nothing in this module is calibrated in that sense, so nothing here is named
confidence, probability or certainty, and `MatchDecision` exposes no such field.

Turning the rate into a genuine posterior would require fitting it against
outcome frequencies on far more negative data than the evaluation corpus holds
(126 negatives cannot resolve a false-accept rate below ~0.8%). Until that
exists, the honest output is a decision plus the evidence behind it.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from musicintel.recognition.fingerprint import FingerprintConfig, FingerprintResult
from musicintel.recognition.index import FingerprintIndex
from musicintel.recognition.matcher import (
    MatchCandidate,
    MatchConfig,
    MatchResult,
    match,
    match_file,
)


class Decision(str, Enum):
    """The verdict. `str` mixin so it serializes as plain text in reports."""

    MATCH = "MATCH"
    NO_MATCH = "NO_MATCH"


@dataclass(frozen=True)
class DecisionConfig:
    """Decision rule parameters.

    `threshold` is the operating point and is the ONLY quantity swept during
    calibration. `min_aligned_landmarks` is held fixed so the sweep stays
    one-dimensional and the reported trade-off curve means what it says.
    """

    # Minimum decision score (aligned landmarks / query landmarks) to accept.
    # The default is a placeholder, NOT a validated operating point -- callers
    # doing real work should pass the threshold selected by calibration and
    # recorded in eval/reports/phase1d_baseline.md.
    threshold: float = 0.05

    # Absolute floor on aligned landmarks, independent of the rate. Guards the
    # degenerate short query: two landmarks, both aligning by chance, is a rate
    # of 1.0 on no evidence at all. Fixed during the sweep.
    min_aligned_landmarks: int = 5

    def validate(self) -> None:
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError("threshold must be a rate in [0, 1]")
        if self.min_aligned_landmarks < 1:
            raise ValueError("min_aligned_landmarks must be >= 1")


DEFAULT_DECISION_CONFIG = DecisionConfig()


def evidence_score(candidate: MatchCandidate, query_landmark_count: int) -> float:
    """Length-normalized evidence: aligned landmarks per query landmark.

    WHY NORMALIZE. The matcher's raw score is a count, so it grows with query
    length: a 10 s query has roughly three times the landmarks of a 3 s query
    and therefore roughly three times the aligned count for the same recording.
    A single count threshold would demand more evidence of a short query than a
    long one purely as an artifact of duration. Dividing by the query's own
    landmark count removes that dependency, giving a quantity that means the
    same thing at 3 s and at 10 s: the fraction of what the query had to offer
    that lined up at one offset.

    WHY THIS NUMERATOR. `candidate.score` counts DISTINCT query landmarks at the
    winning offset, so a hash with many postings cannot inflate it, and offset
    agreement is what separates a recording from a coincidence.

    WHY NOT MORE TERMS. Concentration, margin and total hits are all correlated
    with this quantity and each adds a parameter to justify. A single
    interpretable rate that can be swept in one dimension is worth more than a
    blend that is harder to reason about; the other quantities are reported as
    evidence rather than folded into the score.

    Returns 0.0 for an empty query rather than dividing by zero.
    """
    if query_landmark_count <= 0:
        return 0.0
    return candidate.score / query_landmark_count


@dataclass(frozen=True)
class MatchDecision:
    """A verdict plus every quantity that produced it.

    `evidence_score` is a rate, not a probability -- see the module docstring.
    There is deliberately no `confidence` field.
    """

    decision: Decision
    track_id: str | None  # None whenever the decision is NO_MATCH
    evidence_score: float
    threshold: float
    aligned_landmarks: int
    query_landmark_count: int
    best_offset: int | None
    best_offset_seconds: float | None
    concentration: float
    runner_up_track_id: str | None
    runner_up_score: int
    margin: int  # aligned landmarks minus the runner-up TRACK's aligned count
    candidates: tuple[MatchCandidate, ...]  # ranked, for inspection

    @property
    def is_match(self) -> bool:
        return self.decision is Decision.MATCH

    def to_dict(self) -> dict:
        """Flat, JSON-safe view for reports. Candidates are not included."""
        return {
            "decision": self.decision.value,
            "track_id": self.track_id,
            "evidence_score": round(self.evidence_score, 6),
            "threshold": self.threshold,
            "aligned_landmarks": self.aligned_landmarks,
            "query_landmark_count": self.query_landmark_count,
            "best_offset": self.best_offset,
            "concentration": round(self.concentration, 6),
            "runner_up_track_id": self.runner_up_track_id,
            "runner_up_score": self.runner_up_score,
            "margin": self.margin,
        }


def _no_match(
    threshold: float,
    query_landmark_count: int,
    candidates: tuple[MatchCandidate, ...] = (),
    *,
    score: float = 0.0,
    aligned: int = 0,
) -> MatchDecision:
    return MatchDecision(
        decision=Decision.NO_MATCH,
        track_id=None,
        evidence_score=score,
        threshold=threshold,
        aligned_landmarks=aligned,
        query_landmark_count=query_landmark_count,
        best_offset=None,
        best_offset_seconds=None,
        concentration=0.0,
        runner_up_track_id=None,
        runner_up_score=0,
        margin=0,
        candidates=candidates,
    )


def decide(
    result: MatchResult, *, config: DecisionConfig | None = None
) -> MatchDecision:
    """Accept the top candidate, or return NO_MATCH.

    The rule, in full:

        MATCH  iff  evidence_score >= threshold
               and  aligned_landmarks >= min_aligned_landmarks

    Nothing else. No tie-breaking against the runner-up, no per-condition
    special cases. A rule small enough to state in two lines is a rule whose
    failures can be diagnosed.

    On NO_MATCH the winning track is withheld (`track_id is None`) rather than
    returned alongside a negative verdict, so a caller cannot accidentally read
    a rejected hypothesis as an answer. The ranked candidates stay available on
    `.candidates` for inspection.
    """
    cfg = config or DEFAULT_DECISION_CONFIG
    cfg.validate()

    if not result.candidates:
        return _no_match(cfg.threshold, result.query_landmark_count)

    top = result.candidates[0]
    runner_up = result.candidates[1] if len(result.candidates) > 1 else None
    score = evidence_score(top, result.query_landmark_count)

    accepted = (
        score >= cfg.threshold and top.score >= cfg.min_aligned_landmarks
    )
    if not accepted:
        return _no_match(
            cfg.threshold,
            result.query_landmark_count,
            result.candidates,
            score=score,
            aligned=top.score,
        )

    return MatchDecision(
        decision=Decision.MATCH,
        track_id=top.track_id,
        evidence_score=score,
        threshold=cfg.threshold,
        aligned_landmarks=top.score,
        query_landmark_count=result.query_landmark_count,
        best_offset=top.best_offset,
        best_offset_seconds=top.best_offset_seconds,
        concentration=top.concentration,
        runner_up_track_id=runner_up.track_id if runner_up else None,
        runner_up_score=runner_up.score if runner_up else 0,
        margin=top.score - (runner_up.score if runner_up else 0),
        candidates=result.candidates,
    )


def identify(
    query: FingerprintResult,
    index: FingerprintIndex,
    *,
    match_config: MatchConfig | None = None,
    decision_config: DecisionConfig | None = None,
) -> MatchDecision:
    """Match then decide, in one call."""
    return decide(match(query, index, config=match_config), config=decision_config)


def identify_file(
    path: str | Path,
    index: FingerprintIndex,
    *,
    match_config: MatchConfig | None = None,
    decision_config: DecisionConfig | None = None,
    fingerprint_config: FingerprintConfig | None = None,
) -> MatchDecision:
    """Fingerprint an audio file, match it, and decide."""
    result = match_file(
        path, index, config=match_config, fingerprint_config=fingerprint_config
    )
    return decide(result, config=decision_config)


__all__ = [
    "DEFAULT_DECISION_CONFIG",
    "Decision",
    "DecisionConfig",
    "MatchDecision",
    "decide",
    "evidence_score",
    "identify",
    "identify_file",
]
