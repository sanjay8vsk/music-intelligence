"""Negative-set construction for false-accept measurement.

WHY THIS EXISTS
---------------
Phase 1E measured FAR against 63 held-out negatives. One false accept was worth
1.5873 percentage points, so the smallest non-zero FAR the corpus could resolve
was 1.59% -- an order of magnitude coarser than the 0.1% a production recognizer
needs to demonstrate. No amount of care in the recognizer fixes that; it is a
property of the measuring instrument.

This module builds a larger negative set. It does NOT touch the positive corpus,
and it does NOT replace the original 126 negatives -- those are carried forward
unchanged so Phase 1E remains comparable.

THE TRAP THIS MODULE IS BUILT AROUND
------------------------------------
Excerpt count is not sample size. Cutting 1,000 clips from 12 recordings gives
1,000 measurements of 12 draws, not 1,000 draws. Two clips from the same track
share mastering, instrumentation and noise floor, so they fail or succeed
together. Every count this module reports is therefore paired with the number of
SOURCE RECORDINGS behind it, and the split is made by source track rather than
by excerpt so a track can never appear on both sides.

The second trap is dilution. Silence and white noise are trivially rejected;
padding the set with them drives the aggregate FAR toward zero while telling you
nothing. Categories are kept separate in every report for that reason, and
out-of-catalog MUSIC is the number that matters.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

# Categories already supported by the evaluation framework. Kept identical to
# the Phase 0 condition names so old and new negatives aggregate cleanly.
CAT_MUSIC = "negative_out_of_catalog_music"
CAT_SPEECH = "negative_speech"
CAT_SILENCE = "negative_silence"
CAT_NEAR_SILENCE = "negative_near_silence"
CAT_PINK = "negative_noise_pink"
CAT_WHITE = "negative_noise_white"
SYNTHETIC_CATEGORIES = (
    CAT_SPEECH, CAT_SILENCE, CAT_NEAR_SILENCE, CAT_PINK, CAT_WHITE,
)
ALL_CATEGORIES = (CAT_MUSIC,) + SYNTHETIC_CATEGORIES

_SUFFIX = re.compile(r"_\d+$")
_NONWORD = re.compile(r"[^a-z0-9]+")


def base_track_id(track_id: str) -> str:
    """Strip a trailing `_1`-style suffix.

    The fixture corpus contains families like `ia_adr-002` / `ia_adr-002_1`:
    different recordings from one release, sharing a base identifier. Treating
    them as unrelated is how a near-duplicate ends up on both sides of a split.
    """
    return _SUFFIX.sub("", track_id)


def norm_text(s: str | None) -> str:
    """Casefold and strip punctuation, for comparing artist and title strings."""
    return _NONWORD.sub(" ", (s or "").lower()).strip()


@dataclass(frozen=True)
class NegativeSource:
    """A recording eligible to supply negative queries. Never an indexed track."""

    track_id: str
    path: str
    sha256: str
    duration_sec: float
    license: str
    license_url: str
    source: str
    source_url: str | None = None
    artist: str | None = None
    title: str | None = None
    # "heldout" = one of the manifest's held-out tracks, already used by Phase 1E.
    # "fetched" = newly acquired for this phase.
    origin: str = "fetched"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "NegativeSource":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass(frozen=True)
class NegativeExcerpt:
    """One planned negative query."""

    query_id: str
    category: str
    source_track: str | None  # None for synthetic categories
    start_sec: float
    duration: float
    split: str = "evaluation"
    rendered_path: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


# ------------------------------------------------------------- leakage ------
def screen_candidates(
    candidates: list[NegativeSource],
    catalog_ids: set[str],
    catalog_sha256: set[str],
    catalog_artist_title: set[tuple[str, str]],
    catalog_artists: set[str],
    corpus_ids: set[str],
) -> tuple[list[NegativeSource], list[dict]]:
    """Drop any candidate that could carry indexed audio. Returns (kept, rejected).

    Five independent gates, applied in order. They overlap deliberately -- a
    re-encode defeats the hash gate but not the artist/title gate, and a
    retitled upload defeats the title gate but not the hash gate.

    Same-artist-different-title is rejected too. That is stricter than strictly
    necessary, but netlabel releases by one artist routinely share stems,
    samples and mastering, and a negative that quietly contains catalog audio
    corrupts the exact number this whole phase exists to measure.
    """
    kept: list[NegativeSource] = []
    rejected: list[dict] = []
    seen_sha: set[str] = set()
    seen_base: set[str] = set()

    for c in candidates:
        reason = None
        if c.sha256 in catalog_sha256:
            reason = "sha256 matches a corpus track"
        elif c.track_id in corpus_ids:
            reason = "track_id already in the 44-track corpus"
        elif base_track_id(c.track_id) in {base_track_id(i) for i in corpus_ids}:
            reason = "near-duplicate id family of a corpus track"
        elif (norm_text(c.artist), norm_text(c.title)) in catalog_artist_title:
            reason = "artist+title matches a catalog track"
        elif norm_text(c.artist) and norm_text(c.artist) in catalog_artists:
            reason = "same artist as a catalog track"
        elif c.sha256 in seen_sha:
            reason = "duplicate of another candidate"
        elif base_track_id(c.track_id) in seen_base:
            reason = "near-duplicate of another candidate"
        if reason:
            rejected.append({"track_id": c.track_id, "reason": reason})
            continue
        seen_sha.add(c.sha256)
        seen_base.add(base_track_id(c.track_id))
        kept.append(c)
    return kept, rejected


# -------------------------------------------------------------- planning ----
def plan_disjoint_excerpts(
    source: NegativeSource,
    durations: tuple[float, ...] = (3.0, 5.0, 10.0),
    *,
    head_trim: float = 1.0,
    tail_trim: float = 1.0,
) -> list[NegativeExcerpt]:
    """Tile a source into NON-OVERLAPPING excerpts, cycling the durations.

    Every second of source audio is used at most once. Overlapping excerpts
    would be near-copies of each other, and near-copies inflate the negative
    count without adding evidence -- the thing this module exists to avoid.

    Durations cycle rather than each tiling the track separately, so a 3 s and a
    10 s excerpt never cover the same audio.
    """
    out: list[NegativeExcerpt] = []
    pos = head_trim
    limit = source.duration_sec - tail_trim
    i = 0
    while True:
        dur = durations[i % len(durations)]
        if pos + dur > limit:
            # Try the remaining durations before giving up on the tail.
            shorter = [d for d in durations if pos + d <= limit]
            if not shorter:
                break
            dur = min(shorter)
        out.append(
            NegativeExcerpt(
                query_id=f"neg__{source.track_id}__d{dur:g}s__t{pos:.1f}",
                category=CAT_MUSIC,
                source_track=source.track_id,
                start_sec=round(pos, 3),
                duration=dur,
            )
        )
        pos += dur
        i += 1
    return out


def assign_splits(
    excerpts: list[NegativeExcerpt],
    *,
    calibration_sources: set[str],
    synthetic_side: dict[str, str] | None = None,
) -> list[NegativeExcerpt]:
    """Assign each excerpt a split. Music splits by SOURCE TRACK, never by excerpt.

    Splitting by excerpt would put clips of one recording on both sides, letting
    the threshold be tuned on audio it is then judged against.
    """
    out = []
    for e in excerpts:
        if e.source_track is None:
            side = (synthetic_side or {}).get(e.query_id) or _synthetic_side(e.query_id)
        else:
            side = "calibration" if e.source_track in calibration_sources else "evaluation"
        out.append(
            NegativeExcerpt(**{**e.to_dict(), "split": side})
        )
    return out


def _synthetic_side(query_id: str) -> str:
    """Deterministic, id-derived side for synthetic negatives (Phase 1E rule)."""
    h = int.from_bytes(hashlib.sha256(query_id.encode()).digest()[:2], "big")
    return "calibration" if h % 2 == 0 else "evaluation"


def interleave_calibration(source_ids: list[str]) -> set[str]:
    """Every other source id, by sorted order -- the Phase 1E split policy."""
    return {t for i, t in enumerate(sorted(source_ids)) if i % 2 == 0}


# ------------------------------------------------------------------ set -----
@dataclass
class NegativeSet:
    """Sources plus the excerpts planned from them."""

    sources: list[NegativeSource] = field(default_factory=list)
    excerpts: list[NegativeExcerpt] = field(default_factory=list)
    version: int = 1

    def counts_by_category(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for e in self.excerpts:
            out[e.category] = out.get(e.category, 0) + 1
        return dict(sorted(out.items()))

    def counts_by_split(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for e in self.excerpts:
            out[e.split] = out.get(e.split, 0) + 1
        return dict(sorted(out.items()))

    def source_counts(self) -> dict[str, int]:
        """Distinct SOURCE RECORDINGS per split -- the real sample size."""
        out: dict[str, set[str]] = {}
        for e in self.excerpts:
            if e.source_track:
                out.setdefault(e.split, set()).add(e.source_track)
        return {k: len(v) for k, v in sorted(out.items())}

    def content_hash(self) -> str:
        """Identity of the negative set: which audio, cut where, on which side."""
        payload = sorted(
            (e.query_id, e.category, e.source_track or "", e.start_sec, e.duration, e.split)
            for e in self.excerpts
        )
        src = sorted((s.track_id, s.sha256) for s in self.sources)
        h = hashlib.sha256()
        h.update(json.dumps({"sources": src, "excerpts": payload}, sort_keys=True).encode())
        return h.hexdigest()

    def verify(self, catalog_ids: set[str]) -> list[str]:
        """Problems that would invalidate a FAR measurement. Empty means sound."""
        problems: list[str] = []
        src_ids = {s.track_id for s in self.sources}
        for s in self.sources:
            if s.track_id in catalog_ids:
                problems.append(f"source {s.track_id} is an INDEXED catalog track")
        seen_q: set[str] = set()
        for e in self.excerpts:
            if e.query_id in seen_q:
                problems.append(f"duplicate query_id: {e.query_id}")
            seen_q.add(e.query_id)
            if e.source_track and e.source_track not in src_ids:
                problems.append(f"excerpt {e.query_id} cites unknown source")
            if e.split not in ("calibration", "evaluation"):
                problems.append(f"excerpt {e.query_id} has bad split {e.split!r}")
        # A source track must never straddle the split.
        sides: dict[str, set[str]] = {}
        for e in self.excerpts:
            if e.source_track:
                sides.setdefault(e.source_track, set()).add(e.split)
        for tid, s in sides.items():
            if len(s) > 1:
                problems.append(f"source {tid} appears in both splits")
        # Overlapping excerpts from one source would be near-copies.
        by_src: dict[str, list[NegativeExcerpt]] = {}
        for e in self.excerpts:
            if e.source_track:
                by_src.setdefault(e.source_track, []).append(e)
        for tid, es in by_src.items():
            es = sorted(es, key=lambda x: x.start_sec)
            for a, b in zip(es, es[1:]):
                if a.start_sec + a.duration > b.start_sec + 1e-6:
                    problems.append(f"overlapping excerpts in {tid}")
                    break
        return problems

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": self.version,
            "content_hash": self.content_hash(),
            "source_count": len(self.sources),
            "excerpt_count": len(self.excerpts),
            "counts_by_category": self.counts_by_category(),
            "counts_by_split": self.counts_by_split(),
            "source_recordings_by_split": self.source_counts(),
            "sources": [s.to_dict() for s in self.sources],
            "excerpts": [e.to_dict() for e in self.excerpts],
        }
        p.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")

    @classmethod
    def load(cls, path: str | Path) -> "NegativeSet":
        d = json.loads(Path(path).read_text())
        return cls(
            sources=[NegativeSource.from_dict(s) for s in d["sources"]],
            excerpts=[NegativeExcerpt(**e) for e in d["excerpts"]],
            version=d.get("version", 1),
        )


__all__ = [
    "ALL_CATEGORIES", "CAT_MUSIC", "CAT_NEAR_SILENCE", "CAT_PINK", "CAT_SILENCE",
    "CAT_SPEECH", "CAT_WHITE", "SYNTHETIC_CATEGORIES", "NegativeExcerpt",
    "NegativeSet", "NegativeSource", "assign_splits", "base_track_id",
    "interleave_calibration", "norm_text", "plan_disjoint_excerpts",
    "screen_candidates",
]
