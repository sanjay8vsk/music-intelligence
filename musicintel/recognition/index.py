"""Persistent exact-match index over landmark fingerprints.

WHAT THIS IS FOR
----------------
Phase 1A turns audio into `(hash: uint32, anchor_frame: int32)` pairs. To
identify a recording you need the inverse mapping:

    hash -> [(track_id, anchor_frame), ...]

One hash legitimately occurs many times -- in different tracks, and repeatedly
within one track -- so this is a MULTIMAP, not a dictionary. Collisions are
expected and are not an error: a 28-bit key over ~150 landmarks per second per
track collides constantly. Resolving them is the matcher's job in Phase 1C,
which histograms `anchor_frame_reference - anchor_frame_query` per track and
looks for a spike. This index exists to hand that matcher every posting for a
hash, quickly and exactly.

WHY NOT FAISS
-------------
FAISS answers "which stored vector is nearest to this one". That is a
similarity question, and it is the question the Phase 0 baseline asked -- which
is why the baseline could not tell a matching recording from a merely
similar-sounding one (FAR 1.0, see eval/reports/baseline.md). Landmark lookup
is a different question: exact integer equality, then a vote on time alignment.
Approximate nearest-neighbour search is the wrong tool, so it is absent here.

REPRESENTATION
--------------
Three parallel arrays sorted by hash, plus a track table:

    hashes         uint32, ascending    -- the sort key
    track_ords     uint32               -- index into the track table
    anchor_frames  int32                -- anchor position, in frames

Lookup is a binary search for the equal-range of a hash, then a slice. That is
O(log N) for the search and O(k) for the k postings returned, with no hashing,
no buckets, and no per-key Python object.

The obvious alternative -- `dict[int, list[tuple[str, int]]]` -- reads more
naturally but costs roughly an order of magnitude more memory (a Python int
plus a list plus a tuple plus a str reference per posting, against 12 bytes
here) and has no canonical byte form to persist. Sorted arrays are both the
simpler artifact and the smaller one, so no trade is being made.

Track identity is the caller-supplied `track_id` string, never a filename. The
integer `track_ord` is a storage detail: a dense ordinal into the track table
so postings stay 4 bytes instead of holding a string.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

from musicintel.recognition.fingerprint import (
    FORMAT_VERSION as FINGERPRINT_FORMAT_VERSION,
)
from musicintel.recognition.fingerprint import (
    FingerprintConfig,
    FingerprintResult,
    fingerprint_file,
)

# Version of the on-disk INDEX layout (file names, meta.json schema, array
# dtypes). Independent of the fingerprint format: either can change alone.
INDEX_FORMAT_VERSION = 1

META_FILENAME = "meta.json"
HASHES_FILENAME = "hashes.npy"
TRACK_ORDS_FILENAME = "track_ords.npy"
ANCHOR_FRAMES_FILENAME = "anchor_frames.npy"
_ARRAY_FILENAMES = (HASHES_FILENAME, TRACK_ORDS_FILENAME, ANCHOR_FRAMES_FILENAME)

HASH_DTYPE = np.uint32
ORD_DTYPE = np.uint32
FRAME_DTYPE = np.int32


class IndexFormatError(RuntimeError):
    """A persisted index is corrupt, incomplete, or from an incompatible format."""


# ------------------------------------------------------------------ records --
@dataclass(frozen=True)
class Posting:
    """One occurrence of a hash: which track, and where in it."""

    track_id: str
    anchor_frame: int
    track_ord: int


@dataclass(frozen=True)
class TrackEntry:
    """A track in the index. Identity is `track_id` -- never a path."""

    track_id: str
    fingerprint_count: int
    duration_sec: float

    def to_dict(self) -> dict:
        return {
            "track_id": self.track_id,
            "fingerprint_count": self.fingerprint_count,
            "duration_sec": round(float(self.duration_sec), 6),
        }

    @classmethod
    def from_dict(cls, d: Mapping) -> "TrackEntry":
        try:
            return cls(
                track_id=str(d["track_id"]),
                fingerprint_count=int(d["fingerprint_count"]),
                duration_sec=float(d["duration_sec"]),
            )
        except (KeyError, TypeError, ValueError) as e:
            raise IndexFormatError(f"malformed track entry: {d!r}") from e


# -------------------------------------------------------------------- index --
@dataclass(frozen=True, eq=False)
class FingerprintIndex:
    """An immutable, hash-sorted posting list over a fingerprinted catalog."""

    hashes: np.ndarray  # uint32, ascending
    track_ords: np.ndarray  # uint32, index into `tracks`
    anchor_frames: np.ndarray  # int32
    tracks: tuple[TrackEntry, ...]
    config: FingerprintConfig
    # Recorded for humans, deliberately excluded from `content_hash` and
    # omitted from the artifact unless asked for: a timestamp that changed the
    # bytes would make two identical builds look different.
    built_utc: str | None = None

    # -- shape ----------------------------------------------------------
    def __len__(self) -> int:
        """Number of postings (fingerprints), not tracks."""
        return int(self.hashes.size)

    @property
    def track_ids(self) -> tuple[str, ...]:
        return tuple(t.track_id for t in self.tracks)

    @property
    def n_tracks(self) -> int:
        return len(self.tracks)

    @property
    def n_fingerprints(self) -> int:
        return int(self.hashes.size)

    @property
    def n_unique_hashes(self) -> int:
        return int(np.unique(self.hashes).size) if self.hashes.size else 0

    @property
    def nbytes(self) -> int:
        """Bytes held by the posting arrays (12 per posting)."""
        return int(
            self.hashes.nbytes + self.track_ords.nbytes + self.anchor_frames.nbytes
        )

    def track_entry(self, track_id: str) -> TrackEntry | None:
        for t in self.tracks:
            if t.track_id == track_id:
                return t
        return None

    # -- lookup ----------------------------------------------------------
    def _equal_range(self, key: int) -> tuple[int, int]:
        """Half-open [lo, hi) slice of postings whose hash == key."""
        if self.hashes.size == 0:
            return 0, 0
        k = np.uint32(int(key) & 0xFFFFFFFF)
        lo = int(np.searchsorted(self.hashes, k, side="left"))
        hi = int(np.searchsorted(self.hashes, k, side="right"))
        return lo, hi

    def lookup_raw(self, key: int) -> tuple[np.ndarray, np.ndarray]:
        """Postings as (track_ords, anchor_frames) array views.

        This is the form Phase 1C's offset histogram wants: integer arrays it
        can subtract and bincount without building Python objects per posting.
        """
        lo, hi = self._equal_range(key)
        return self.track_ords[lo:hi], self.anchor_frames[lo:hi]

    def lookup(self, key: int) -> list[Posting]:
        """All postings for a hash. An unknown hash returns [] -- never raises.

        A miss is the normal case, not an exception: most query hashes are
        absent from any given catalog.
        """
        ords, frames = self.lookup_raw(key)
        ids = self.track_ids
        return [
            Posting(track_id=ids[int(o)], anchor_frame=int(f), track_ord=int(o))
            for o, f in zip(ords, frames)
        ]

    def count(self, key: int) -> int:
        """Number of postings for a hash, without materializing them."""
        lo, hi = self._equal_range(key)
        return hi - lo

    def __contains__(self, key: int) -> bool:
        return self.count(key) > 0

    # -- identity ---------------------------------------------------------
    def content_hash(self) -> str:
        """SHA-256 over everything that defines the index's contents.

        Excludes `built_utc` by construction, so two builds of the same catalog
        with the same config produce the same identity even when built at
        different times.
        """
        h = hashlib.sha256()
        payload = {
            "index_format_version": INDEX_FORMAT_VERSION,
            "fingerprint_format_version": FINGERPRINT_FORMAT_VERSION,
            "config": dataclasses.asdict(self.config),
            "tracks": [t.to_dict() for t in self.tracks],
        }
        h.update(json.dumps(payload, sort_keys=True).encode())
        for arr in (self.hashes, self.track_ords, self.anchor_frames):
            h.update(np.ascontiguousarray(arr).tobytes())
        return h.hexdigest()

    def metadata(self) -> dict:
        """The meta.json payload."""
        return {
            "index_format_version": INDEX_FORMAT_VERSION,
            "fingerprint_format_version": FINGERPRINT_FORMAT_VERSION,
            "content_hash": self.content_hash(),
            "sample_rate": self.config.sample_rate,
            "fingerprint_config": dataclasses.asdict(self.config),
            "track_count": self.n_tracks,
            "fingerprint_count": self.n_fingerprints,
            "unique_hash_count": self.n_unique_hashes,
            "built_utc": self.built_utc,
            "tracks": [t.to_dict() for t in self.tracks],
        }

    # -- integrity ---------------------------------------------------------
    def validate(self) -> None:
        """Raise IndexFormatError on any internally inconsistent state."""
        n = self.hashes.size
        if self.track_ords.size != n or self.anchor_frames.size != n:
            raise IndexFormatError(
                f"array length mismatch: hashes={n} "
                f"track_ords={self.track_ords.size} "
                f"anchor_frames={self.anchor_frames.size}"
            )
        if self.hashes.dtype != HASH_DTYPE:
            raise IndexFormatError(f"hashes dtype {self.hashes.dtype}, want {HASH_DTYPE}")
        if self.track_ords.dtype != ORD_DTYPE:
            raise IndexFormatError(
                f"track_ords dtype {self.track_ords.dtype}, want {ORD_DTYPE}"
            )
        if self.anchor_frames.dtype != FRAME_DTYPE:
            raise IndexFormatError(
                f"anchor_frames dtype {self.anchor_frames.dtype}, want {FRAME_DTYPE}"
            )
        if n and not np.all(np.diff(self.hashes.astype(np.int64)) >= 0):
            # Binary-search lookup is only correct on sorted data; an unsorted
            # array would silently return wrong postings rather than fail.
            raise IndexFormatError("hashes are not sorted ascending")
        if n and self.n_tracks == 0:
            raise IndexFormatError("postings present but track table is empty")
        if n and int(self.track_ords.max()) >= self.n_tracks:
            raise IndexFormatError(
                f"track_ord {int(self.track_ords.max())} outside track table "
                f"of size {self.n_tracks}"
            )
        if n and int(self.anchor_frames.min()) < 0:
            raise IndexFormatError("negative anchor frame")
        ids = [t.track_id for t in self.tracks]
        if len(set(ids)) != len(ids):
            raise IndexFormatError("duplicate track_id in track table")

    # -- persistence -------------------------------------------------------
    def save(self, directory: str | Path, *, include_timestamp: bool = False) -> Path:
        """Write the index to `directory`, creating it if needed.

        `include_timestamp` defaults to False so that two builds of the same
        catalog produce byte-identical artifacts. Turn it on when provenance
        matters more than reproducible bytes; `content_hash` is unaffected
        either way.
        """
        self.validate()
        out = Path(directory)
        out.mkdir(parents=True, exist_ok=True)

        meta = self.metadata()
        if not include_timestamp:
            meta["built_utc"] = None
        (out / META_FILENAME).write_text(
            json.dumps(meta, indent=2, sort_keys=True) + "\n"
        )
        for name, arr in (
            (HASHES_FILENAME, self.hashes),
            (TRACK_ORDS_FILENAME, self.track_ords),
            (ANCHOR_FRAMES_FILENAME, self.anchor_frames),
        ):
            # .npy carries no timestamp and no compression, so identical arrays
            # give identical bytes. allow_pickle=False keeps loading safe.
            with open(out / name, "wb") as fh:
                np.save(fh, np.ascontiguousarray(arr), allow_pickle=False)
        return out

    @classmethod
    def load(cls, directory: str | Path) -> "FingerprintIndex":
        """Read an index from disk. No audio and no re-fingerprinting involved.

        Rejects anything it cannot vouch for: missing files, a format version
        it does not implement, a fingerprint format it cannot compare against,
        a config it cannot reconstruct, or a content hash that does not match.
        """
        d = Path(directory)
        meta_path = d / META_FILENAME
        if not meta_path.is_file():
            raise IndexFormatError(f"no {META_FILENAME} in {d}")
        for name in _ARRAY_FILENAMES:
            if not (d / name).is_file():
                raise IndexFormatError(f"missing array file {name} in {d}")

        try:
            meta = json.loads(meta_path.read_text())
        except json.JSONDecodeError as e:
            raise IndexFormatError(f"{META_FILENAME} is not valid JSON: {e}") from e
        if not isinstance(meta, dict):
            raise IndexFormatError(f"{META_FILENAME} must contain a JSON object")

        got = meta.get("index_format_version")
        if got != INDEX_FORMAT_VERSION:
            raise IndexFormatError(
                f"index format version {got!r}, this build reads "
                f"{INDEX_FORMAT_VERSION}"
            )
        fp_ver = meta.get("fingerprint_format_version")
        if fp_ver != FINGERPRINT_FORMAT_VERSION:
            raise IndexFormatError(
                f"index holds fingerprint format {fp_ver!r}, this build "
                f"produces {FINGERPRINT_FORMAT_VERSION}; the stored hashes "
                f"cannot be compared against freshly extracted ones"
            )

        config = _config_from_dict(meta.get("fingerprint_config"))
        raw_tracks = meta.get("tracks")
        if not isinstance(raw_tracks, list):
            raise IndexFormatError("meta.tracks must be a list")
        tracks = tuple(TrackEntry.from_dict(t) for t in raw_tracks)

        try:
            hashes = np.load(d / HASHES_FILENAME, allow_pickle=False)
            ords = np.load(d / TRACK_ORDS_FILENAME, allow_pickle=False)
            frames = np.load(d / ANCHOR_FRAMES_FILENAME, allow_pickle=False)
        except Exception as e:  # noqa: BLE001 -- any decode failure is corruption
            raise IndexFormatError(f"cannot read index arrays: {e}") from e

        idx = cls(
            hashes=np.ascontiguousarray(hashes),
            track_ords=np.ascontiguousarray(ords),
            anchor_frames=np.ascontiguousarray(frames),
            tracks=tracks,
            config=config,
            built_utc=meta.get("built_utc"),
        )
        idx.validate()

        declared = meta.get("content_hash")
        actual = idx.content_hash()
        if declared is not None and declared != actual:
            raise IndexFormatError(
                f"content hash mismatch: meta says {declared}, arrays hash to "
                f"{actual} -- the index is corrupt or was edited"
            )
        for field, value in (
            ("track_count", idx.n_tracks),
            ("fingerprint_count", idx.n_fingerprints),
        ):
            if meta.get(field) is not None and meta[field] != value:
                raise IndexFormatError(
                    f"meta.{field}={meta[field]} but index holds {value}"
                )
        return idx


def _config_from_dict(d) -> FingerprintConfig:
    """Rebuild a FingerprintConfig, refusing anything that is not an exact match.

    Silently dropping an unknown key or defaulting a missing one would produce
    an index whose stored hashes were made with settings nobody can recover.
    """
    if not isinstance(d, dict):
        raise IndexFormatError("meta.fingerprint_config must be an object")
    known = {f.name for f in dataclasses.fields(FingerprintConfig)}
    unknown = set(d) - known
    missing = known - set(d)
    if unknown:
        raise IndexFormatError(f"unknown fingerprint config keys: {sorted(unknown)}")
    if missing:
        raise IndexFormatError(f"missing fingerprint config keys: {sorted(missing)}")
    try:
        cfg = FingerprintConfig(**d)
        cfg.validate()
    except (TypeError, ValueError) as e:
        raise IndexFormatError(f"invalid fingerprint config: {e}") from e
    return cfg


# -------------------------------------------------------------------- build --
def build_index(
    items: Iterable[tuple[str, FingerprintResult]],
    *,
    config: FingerprintConfig | None = None,
    built_utc: str | None = None,
) -> FingerprintIndex:
    """Build an index from `(track_id, FingerprintResult)` pairs.

    Tracks keep the order they arrive in, so `track_ord` is stable for a stable
    input order. Postings are then sorted by (hash, track_ord, anchor_frame),
    which makes the arrays canonical: the same catalog always yields the same
    bytes regardless of how the sort was reached.
    """
    track_entries: list[TrackEntry] = []
    seen: set[str] = set()
    hash_chunks: list[np.ndarray] = []
    ord_chunks: list[np.ndarray] = []
    frame_chunks: list[np.ndarray] = []
    resolved = config

    for track_id, result in items:
        if not isinstance(track_id, str) or not track_id:
            raise ValueError(f"track_id must be a non-empty string, got {track_id!r}")
        if track_id in seen:
            # Two tracks under one id would make every posting for that id
            # ambiguous, and the ambiguity would only surface as wrong matches.
            raise ValueError(f"duplicate track_id: {track_id!r}")
        seen.add(track_id)

        if resolved is None:
            resolved = result.config
        elif result.config != resolved:
            raise ValueError(
                f"track {track_id!r} was fingerprinted with a different config; "
                f"hashes from different configs are not comparable"
            )

        ordinal = len(track_entries)
        track_entries.append(
            TrackEntry(
                track_id=track_id,
                fingerprint_count=len(result),
                duration_sec=float(result.duration_sec),
            )
        )
        if len(result):
            hash_chunks.append(result.hashes.astype(HASH_DTYPE, copy=False))
            ord_chunks.append(np.full(len(result), ordinal, dtype=ORD_DTYPE))
            frame_chunks.append(result.anchor_frames.astype(FRAME_DTYPE, copy=False))

    cfg = resolved if resolved is not None else FingerprintConfig()

    if hash_chunks:
        hashes = np.concatenate(hash_chunks)
        ords = np.concatenate(ord_chunks)
        frames = np.concatenate(frame_chunks)
        order = np.lexsort((frames, ords, hashes))  # primary key: hashes
        hashes, ords, frames = hashes[order], ords[order], frames[order]
    else:
        hashes = np.empty(0, dtype=HASH_DTYPE)
        ords = np.empty(0, dtype=ORD_DTYPE)
        frames = np.empty(0, dtype=FRAME_DTYPE)

    idx = FingerprintIndex(
        hashes=np.ascontiguousarray(hashes),
        track_ords=np.ascontiguousarray(ords),
        anchor_frames=np.ascontiguousarray(frames),
        tracks=tuple(track_entries),
        config=cfg,
        built_utc=built_utc,
    )
    idx.validate()
    return idx


def build_index_from_files(
    catalog: Sequence[tuple[str, str | Path]],
    *,
    config: FingerprintConfig | None = None,
    built_utc: str | None = None,
    verbose: bool = False,
) -> FingerprintIndex:
    """Fingerprint `(track_id, audio_path)` pairs and index them.

    The path is an input, never the identity: `track_id` is what the index
    stores, so moving or renaming the audio does not change what was indexed.
    """
    cfg = config or FingerprintConfig()

    def _pairs():
        for i, (track_id, path) in enumerate(catalog, start=1):
            if verbose:
                print(f"  [{i}/{len(catalog)}] {track_id}")
            yield track_id, fingerprint_file(path, cfg)

    return build_index(_pairs(), config=cfg, built_utc=built_utc)


__all__ = [
    "ANCHOR_FRAMES_FILENAME",
    "HASHES_FILENAME",
    "INDEX_FORMAT_VERSION",
    "META_FILENAME",
    "TRACK_ORDS_FILENAME",
    "FingerprintIndex",
    "IndexFormatError",
    "Posting",
    "TrackEntry",
    "build_index",
    "build_index_from_files",
]
