"""Catalog track identity and provenance.

WHAT A CATALOG IS FOR
---------------------
Phase 1 built a recognizer that answers "which track_id is this?". Nothing yet
decides what a track_id IS, or where the audio behind it came from. Every
benchmark so far borrowed the evaluation manifest for that, which is fine for a
fixture set and wrong for a catalog: the eval manifest encodes a benchmark
split, not an owned collection.

This module is the missing half. It records, for each track: a stable identity,
the content hash that pins the audio, and the provenance needed to explain where
it came from -- and nothing about how it was fingerprinted, which belongs to the
index.

IDENTITY IS NOT A FILENAME
--------------------------
The prototype this replaces derived identity with `file.split(".")[0]`. That is
not a stylistic complaint: `mix.1.wav` and `mix.2.wav` both collapse to `mix`,
so ingesting both silently destroys one, and `Song feat. Artist.mp3` becomes
`Song feat`. A catalog whose identities silently merge cannot be trusted to say
which recording matched.

Here identity is an explicit `track_id`, collisions are an error rather than an
overwrite, and the source path is retained as provenance only -- moving or
renaming a file does not change what was ingested.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

CATALOG_VERSION = 1
_HASH_CHUNK = 1 << 20


def sha256_file(path: str | Path) -> str:
    """SHA-256 of a file's bytes -- the content identity of one recording."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(_HASH_CHUNK):
            h.update(chunk)
    return h.hexdigest()


@dataclass(frozen=True)
class CatalogTrack:
    """One owned recording.

    `track_id` is the identity the recognizer returns. `source_path` is
    provenance: it records where the audio was read from and is deliberately
    NOT the identity, so a rename is not a new track.
    """

    track_id: str
    source_path: str
    sha256: str
    duration_sec: float
    bytes: int = 0
    fingerprint_count: int = 0
    title: str | None = None
    artist: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "CatalogTrack":
        known = {f for f in cls.__dataclass_fields__}
        missing = {"track_id", "source_path", "sha256", "duration_sec"} - set(d)
        if missing:
            raise ValueError(f"catalog track missing fields: {sorted(missing)}")
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class Catalog:
    """A set of owned recordings, with the integrity checks a catalog needs."""

    tracks: list[CatalogTrack] = field(default_factory=list)
    version: int = CATALOG_VERSION

    def __len__(self) -> int:
        return len(self.tracks)

    def __iter__(self):
        return iter(self.tracks)

    @property
    def track_ids(self) -> tuple[str, ...]:
        return tuple(t.track_id for t in self.tracks)

    @property
    def total_duration_sec(self) -> float:
        return float(sum(t.duration_sec for t in self.tracks))

    @property
    def total_fingerprints(self) -> int:
        return int(sum(t.fingerprint_count for t in self.tracks))

    def by_id(self, track_id: str) -> CatalogTrack | None:
        for t in self.tracks:
            if t.track_id == track_id:
                return t
        return None

    def content_hash(self) -> str:
        """Identity of WHICH audio the catalog holds.

        Depends only on (track_id, sha256) pairs -- not on ordering, not on
        paths, not on mutable metadata -- so moving the files or re-sorting the
        catalog does not change it.
        """
        payload = sorted((t.track_id, t.sha256) for t in self.tracks)
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    def duplicate_content(self) -> dict[str, list[str]]:
        """Track ids sharing one audio hash -- the same recording ingested twice.

        Not automatically an error: a catalog may legitimately hold one
        recording under two ids. It is always worth reporting, because at query
        time those ids are indistinguishable.
        """
        by_hash: dict[str, list[str]] = {}
        for t in self.tracks:
            by_hash.setdefault(t.sha256, []).append(t.track_id)
        return {h: sorted(ids) for h, ids in sorted(by_hash.items()) if len(ids) > 1}

    def verify(self, root: str | Path = ".", *, check_hashes: bool = False) -> list[str]:
        """Problems that would make the catalog untrustworthy. Empty means sound."""
        problems: list[str] = []
        seen: set[str] = set()
        for t in self.tracks:
            if not t.track_id:
                problems.append("empty track_id")
            if t.track_id in seen:
                problems.append(f"duplicate track_id: {t.track_id}")
            seen.add(t.track_id)
            if len(t.sha256) != 64:
                problems.append(f"malformed sha256 for {t.track_id}")
            if t.duration_sec <= 0:
                problems.append(f"non-positive duration for {t.track_id}")
            p = Path(root) / t.source_path
            if not p.is_file():
                problems.append(f"missing audio: {t.source_path}")
                continue
            if check_hashes and sha256_file(p) != t.sha256:
                problems.append(f"content changed since ingestion: {t.source_path}")
        return problems

    # -- persistence ------------------------------------------------------
    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": self.version,
            "content_hash": self.content_hash(),
            "track_count": len(self.tracks),
            "total_duration_sec": round(self.total_duration_sec, 3),
            "total_fingerprints": self.total_fingerprints,
            "extension_counts": dict(
                Counter(Path(t.source_path).suffix.lower() for t in self.tracks)
            ),
            "tracks": [t.to_dict() for t in sorted(self.tracks, key=lambda t: t.track_id)],
        }
        p.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")
        return p

    @classmethod
    def load(cls, path: str | Path) -> "Catalog":
        d = json.loads(Path(path).read_text())
        if not isinstance(d, dict) or "tracks" not in d:
            raise ValueError(f"{path} is not a catalog file")
        return cls(tracks=[CatalogTrack.from_dict(t) for t in d["tracks"]],
                   version=d.get("version", CATALOG_VERSION))


__all__ = ["CATALOG_VERSION", "Catalog", "CatalogTrack", "sha256_file"]
