"""Offset-histogram matcher over a landmark fingerprint index.

THE QUESTION THIS ANSWERS
-------------------------
Given fingerprints from a query recording, which catalog track shows the
strongest TEMPORAL ALIGNMENT of matching fingerprints?

Not "which track shares the most hashes" -- that is a similarity question and
it is the one the Phase 0 baseline asked, which is why it returned a catalog
track for silence, speech and pure noise alike (FAR 1.0, see
eval/reports/baseline.md). Shared hashes alone are weak evidence: a 28-bit key
over ~150 landmarks per second collides constantly, and 34.5% of the hashes in
a small real index already carry more than one posting.

The evidence that actually identifies a recording is *agreement about time*.

WHY A TRUE MATCH SPIKES
-----------------------
If a query is a recording of catalog track T starting `k` frames into it, then
EVERY landmark the two share satisfies

    db_anchor_frame - query_anchor_frame == k

for the same k. The whole set of true matches votes for one offset. Coincidental
collisions have no reason to agree: their offsets scatter across the whole range
the track spans. So the matcher builds a per-track histogram of

    offset = db_anchor_frame - query_anchor_frame

and asks how tall its tallest cluster is. A spike is structural evidence that
the two recordings are the same audio in the same time order; a flat histogram
with the same total hit count is not.

SCOPE -- READ THIS BEFORE USING THE SCORE
-----------------------------------------
Phase 1C ranks candidates. It does NOT decide whether the winner is real.

`MatchCandidate.score` is a COUNT of aligned landmarks. It is not a probability,
not a confidence, and not calibrated in any way. A query of pure noise against
this matcher will still produce a ranked list with a non-zero top score, because
some offset always wins by chance.

Deciding "this is a match" versus NO_MATCH -- thresholds, calibration, rejection
-- is Phase 1D and is deliberately absent here. Anything that consumes this
module today must treat a top candidate as a hypothesis, not an answer.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from musicintel.recognition.fingerprint import (
    FingerprintConfig,
    FingerprintResult,
    fingerprint_file,
)
from musicintel.recognition.index import FingerprintIndex


@dataclass(frozen=True)
class MatchConfig:
    """Matcher tunables.

    Set from the frame geometry of the fingerprint format, not fitted against
    the evaluation corpus.
    """

    # How far apart two offsets may be and still count as the same alignment,
    # in frames. A frame is hop_length/sample_rate = 11.61 ms at the defaults,
    # so the default window spans offsets [k, k+2] -- about 35 ms.
    #
    # Why not 0 (exact equality): the query and the reference are decoded
    # separately and their STFT frame grids need not line up. An excerpt that
    # starts mid-hop shifts every peak's frame index, and a peak sitting near a
    # neighbourhood boundary can be picked one frame either side. Those move the
    # offset by a frame or two without the audio differing at all.
    #
    # Why not large: widening the window scoops up more unrelated collisions,
    # and a wide enough window makes any track look aligned. Two frames keeps
    # the tolerance at roughly one STFT hop of slack in each direction.
    offset_tolerance_frames: int = 2

    # Candidates returned, ranked. Ranking is over every track with a hit; this
    # only truncates the output.
    max_candidates: int = 10

    def validate(self) -> None:
        if self.offset_tolerance_frames < 0:
            raise ValueError("offset_tolerance_frames must be >= 0")
        if self.max_candidates < 1:
            raise ValueError("max_candidates must be >= 1")


DEFAULT_MATCH_CONFIG = MatchConfig()


# ------------------------------------------------------------------ results --
@dataclass(frozen=True)
class MatchCandidate:
    """One ranked hypothesis, with the evidence behind it.

    `score` is the primary evidence: the number of DISTINCT QUERY LANDMARKS that
    align at the winning offset. Distinct, not raw hits, so a single query
    landmark whose hash happens to occur 221 times in one track contributes one
    unit of evidence rather than 221.

    It is a count. It is not a probability and not a confidence -- see the
    module docstring. Comparing scores across queries of different lengths is
    meaningless on its own; a longer query has more landmarks to align.
    """

    track_id: str
    score: int  # distinct query landmarks aligned at best_offset
    best_offset: int  # frames; db_anchor - query_anchor
    best_offset_seconds: float
    best_offset_count: int  # raw hits inside the winning window
    total_hits: int  # every posting retrieved for this track
    matched_query_landmarks: int  # distinct query landmarks hitting this track at all
    second_best_offset: int | None  # best cluster disjoint from the winner
    second_best_score: int

    @property
    def concentration(self) -> float:
        """Fraction of this track's matched landmarks that share the winning offset.

        A dispersion measure, not a confidence. 1.0 means every landmark that
        touched this track agreed on one alignment; a value near zero means the
        hits were scattered and the "match" is collision noise.
        """
        if self.matched_query_landmarks == 0:
            return 0.0
        return self.score / self.matched_query_landmarks

    @property
    def margin(self) -> int:
        """Aligned landmarks at the winning offset minus the runner-up cluster."""
        return self.score - self.second_best_score


@dataclass(frozen=True)
class MatchTiming:
    """Wall-clock split of one match call, in seconds."""

    lookup: float
    histogram: float
    ranking: float

    @property
    def total(self) -> float:
        return self.lookup + self.histogram + self.ranking


@dataclass(frozen=True)
class MatchResult:
    """Ranked candidates for one query.

    A non-empty candidate list is NOT a recognition decision. Phase 1C always
    returns whatever ranked best; rejection is Phase 1D.
    """

    candidates: tuple[MatchCandidate, ...]
    query_landmark_count: int
    query_duration_sec: float
    matched_query_landmarks: int  # distinct query landmarks hitting ANY track
    total_hits: int
    timing: MatchTiming

    def __len__(self) -> int:
        return len(self.candidates)

    @property
    def best(self) -> MatchCandidate | None:
        return self.candidates[0] if self.candidates else None

    @property
    def top_id(self) -> str | None:
        return self.candidates[0].track_id if self.candidates else None

    def top(self, k: int) -> tuple[MatchCandidate, ...]:
        return self.candidates[:k]

    def top_ids(self, k: int) -> list[str]:
        return [c.track_id for c in self.candidates[:k]]


# ------------------------------------------------------------------ helpers --
def _expand_ranges(starts: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """Concatenate arange(s, s+c) for every (s, c) pair, without a Python loop.

    Used to turn "each query hash matches index rows [lo, hi)" into one flat
    array of row positions. The standard ragged-range idiom: lay out a global
    arange, then subtract each group's starting position within it.
    """
    total = int(counts.sum())
    if total == 0:
        return np.empty(0, dtype=np.int64)
    group_starts = np.cumsum(counts) - counts  # offset of each group in the output
    return (
        np.arange(total, dtype=np.int64)
        - np.repeat(group_starts, counts)
        + np.repeat(starts, counts)
    )


def _best_cluster(
    offsets: np.ndarray, query_idx: np.ndarray, tolerance: int
) -> tuple[int, int, int, int, int]:
    """Densest window of aligned offsets.

    `offsets` must be sorted ascending, with `query_idx` in the same order.
    Returns (score, hits, best_offset, window_lo_value, window_hi_value) where
    `score` counts DISTINCT query landmarks inside the window.

    A sliding window over sorted offsets rather than fixed bins: fixed bins put
    hard edges at arbitrary places, and a cluster straddling an edge gets split
    in half. A window has no edges to straddle.
    """
    n = offsets.size
    if n == 0:
        return 0, 0, 0, 0, 0

    counts: dict[int, int] = {}
    distinct = 0
    left = 0
    best_score = -1
    best_hits = -1
    best_lo = best_hi = 0

    for right in range(n):
        q = int(query_idx[right])
        counts[q] = counts.get(q, 0) + 1
        if counts[q] == 1:
            distinct += 1
        while offsets[right] - offsets[left] > tolerance:
            ql = int(query_idx[left])
            counts[ql] -= 1
            if counts[ql] == 0:
                distinct -= 1
            left += 1
        hits = right - left + 1
        # Ties go to the earlier (lower-offset) window, so the choice is stable.
        if distinct > best_score or (distinct == best_score and hits > best_hits):
            best_score, best_hits, best_lo, best_hi = distinct, hits, left, right

    window = offsets[best_lo : best_hi + 1]
    # Representative offset: the single most common exact offset in the window,
    # ties broken toward the smaller value. Reporting the modal offset rather
    # than the window edge keeps `best_offset` meaningful when tolerance is 0.
    values, freq = np.unique(window, return_counts=True)
    best_offset = int(values[int(np.argmax(freq))])
    return best_score, best_hits, best_offset, int(window[0]), int(window[-1])


# -------------------------------------------------------------------- match --
def match(
    query: FingerprintResult,
    index: FingerprintIndex,
    *,
    config: MatchConfig | None = None,
) -> MatchResult:
    """Rank catalog tracks by temporal alignment with `query`.

    Every posting of every matching hash is considered -- not the first, not the
    nearest. There is no distance metric and no vector search anywhere in here.

    Returns candidates ordered by evidence. An empty list means no query hash
    appeared in the index at all; it does NOT mean "rejected", because Phase 1C
    makes no accept/reject decision.
    """
    cfg = config or DEFAULT_MATCH_CONFIG
    cfg.validate()
    if query.config != index.config:
        # Hashes made under different settings are not comparable; matching them
        # would silently produce nonsense rather than an error.
        raise ValueError(
            "query was fingerprinted with a different config than the index"
        )

    empty_timing = MatchTiming(0.0, 0.0, 0.0)
    if len(query) == 0 or len(index) == 0:
        return MatchResult(
            candidates=(),
            query_landmark_count=len(query),
            query_duration_sec=query.duration_sec,
            matched_query_landmarks=0,
            total_hits=0,
            timing=empty_timing,
        )

    # -- 1. lookup: every posting for every query hash ---------------------
    t0 = time.perf_counter()
    lo = np.searchsorted(index.hashes, query.hashes, side="left")
    hi = np.searchsorted(index.hashes, query.hashes, side="right")
    counts = (hi - lo).astype(np.int64)
    rows = _expand_ranges(lo.astype(np.int64), counts)
    post_ords = index.track_ords[rows].astype(np.int64)
    post_frames = index.anchor_frames[rows].astype(np.int64)
    query_anchor = np.repeat(query.anchor_frames.astype(np.int64), counts)
    query_index = np.repeat(np.arange(len(query), dtype=np.int64), counts)
    t_lookup = time.perf_counter() - t0

    if rows.size == 0:
        return MatchResult(
            candidates=(),
            query_landmark_count=len(query),
            query_duration_sec=query.duration_sec,
            matched_query_landmarks=0,
            total_hits=0,
            timing=MatchTiming(t_lookup, 0.0, 0.0),
        )

    # -- 2. histogram: offsets, grouped by track ---------------------------
    t1 = time.perf_counter()
    deltas = post_frames - query_anchor
    # Sort by (track, offset) so each track's offsets arrive already ordered,
    # which is what the sliding window needs.
    order = np.lexsort((deltas, post_ords))
    post_ords = post_ords[order]
    deltas = deltas[order]
    query_index = query_index[order]

    track_ids = index.track_ids
    frame_seconds = index.config.hop_length / index.config.sample_rate
    tol = cfg.offset_tolerance_frames

    boundaries = np.flatnonzero(np.diff(post_ords)) + 1
    starts = np.concatenate(([0], boundaries))
    ends = np.concatenate((boundaries, [post_ords.size]))

    candidates: list[MatchCandidate] = []
    for s, e in zip(starts.tolist(), ends.tolist()):
        ordinal = int(post_ords[s])
        offs = deltas[s:e]
        qidx = query_index[s:e]

        score, hits, best_offset, win_lo, win_hi = _best_cluster(offs, qidx, tol)

        # Runner-up cluster, excluding everything within tolerance of the winner
        # so the "second peak" is a genuine alternative alignment and not the
        # same peak shifted by one frame.
        keep = (offs < win_lo - tol) | (offs > win_hi + tol)
        if keep.any():
            second_score, _, second_offset, _, _ = _best_cluster(
                offs[keep], qidx[keep], tol
            )
        else:
            second_score, second_offset = 0, None

        candidates.append(
            MatchCandidate(
                track_id=track_ids[ordinal],
                score=score,
                best_offset=best_offset,
                best_offset_seconds=best_offset * frame_seconds,
                best_offset_count=hits,
                total_hits=int(e - s),
                matched_query_landmarks=int(np.unique(qidx).size),
                second_best_offset=second_offset,
                second_best_score=second_score,
            )
        )
    t_histogram = time.perf_counter() - t1

    # -- 3. rank -----------------------------------------------------------
    t2 = time.perf_counter()
    candidates.sort(
        key=lambda c: (
            -c.score,
            -c.best_offset_count,
            -c.matched_query_landmarks,
            c.track_id,  # total order: equal evidence still ranks deterministically
        )
    )
    ranked = tuple(candidates[: cfg.max_candidates])
    t_rank = time.perf_counter() - t2

    return MatchResult(
        candidates=ranked,
        query_landmark_count=len(query),
        query_duration_sec=query.duration_sec,
        matched_query_landmarks=int(np.unique(query_index).size),
        total_hits=int(rows.size),
        timing=MatchTiming(t_lookup, t_histogram, t_rank),
    )


def match_file(
    path: str | Path,
    index: FingerprintIndex,
    *,
    config: MatchConfig | None = None,
    fingerprint_config: FingerprintConfig | None = None,
) -> MatchResult:
    """Fingerprint an audio file with the index's own config, then match it."""
    query = fingerprint_file(path, fingerprint_config or index.config)
    return match(query, index, config=config)


__all__ = [
    "DEFAULT_MATCH_CONFIG",
    "MatchCandidate",
    "MatchConfig",
    "MatchResult",
    "MatchTiming",
    "match",
    "match_file",
]
