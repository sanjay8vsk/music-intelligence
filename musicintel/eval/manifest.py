"""Fixture-corpus manifest: track identity, provenance, and licensing.

The manifest is the reproducibility anchor for the whole benchmark. Audio is
never committed; this file records exactly which audio a given result set was
produced from, so a report can be tied to a specific corpus by hash.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

MANIFEST_VERSION = 1
_HASH_CHUNK = 1 << 20


def sha256_file(path: str | Path) -> str:
    """SHA-256 of a file's bytes."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(_HASH_CHUNK):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@dataclass
class Track:
    """One reference recording in the evaluation corpus."""

    track_id: str
    path: str  # repo-relative
    sha256: str
    source: str
    license: str
    license_url: str
    duration_sec: float
    bytes: int = 0
    title: str | None = None
    artist: str | None = None
    genre: str | None = None
    source_url: str | None = None
    # Held-out tracks are deliberately NOT indexed; they become negative queries.
    held_out: bool = False

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Track":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class Manifest:
    """A set of reference tracks plus provenance metadata."""

    tracks: list[Track] = field(default_factory=list)
    version: int = MANIFEST_VERSION

    # -- access ---------------------------------------------------------
    def __len__(self) -> int:
        return len(self.tracks)

    def __iter__(self):
        return iter(self.tracks)

    @property
    def catalog(self) -> list[Track]:
        """Tracks that are indexed by the recognizer (positive ground truth)."""
        return [t for t in self.tracks if not t.held_out]

    @property
    def held_out(self) -> list[Track]:
        """Tracks deliberately excluded from the index (negative queries)."""
        return [t for t in self.tracks if t.held_out]

    def by_id(self, track_id: str) -> Track | None:
        for t in self.tracks:
            if t.track_id == track_id:
                return t
        return None

    def license_counts(self) -> dict[str, int]:
        return dict(Counter(t.license for t in self.tracks))

    # -- integrity ------------------------------------------------------
    def content_hash(self) -> str:
        """Stable identifier for WHICH audio is in the corpus.

        Depends only on track ids and audio hashes -- not on ordering, mutable
        metadata, or the catalog/held-out split -- so it stays constant from the
        moment the corpus is fetched and can be cited unambiguously in a report.
        """
        payload = sorted((t.track_id, t.sha256) for t in self.tracks)
        return sha256_bytes(json.dumps(payload, sort_keys=True).encode())

    def split_hash(self) -> str:
        """Identifier for the corpus AND its catalog/held-out split."""
        payload = sorted((t.track_id, t.sha256, t.held_out) for t in self.tracks)
        return sha256_bytes(json.dumps(payload, sort_keys=True).encode())

    def verify(self, repo_root: str | Path, *, check_hashes: bool = False) -> list[str]:
        """Return a list of problems; empty means the corpus is intact."""
        root = Path(repo_root)
        problems: list[str] = []
        seen: set[str] = set()
        for t in self.tracks:
            if t.track_id in seen:
                problems.append(f"duplicate track_id: {t.track_id}")
            seen.add(t.track_id)
            p = root / t.path
            if not p.exists():
                problems.append(f"missing audio: {t.path}")
                continue
            if check_hashes and sha256_file(p) != t.sha256:
                problems.append(f"hash mismatch: {t.path}")
        return problems

    # -- persistence ----------------------------------------------------
    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": self.version,
            "content_hash": self.content_hash(),
            "track_count": len(self.tracks),
            "license_counts": self.license_counts(),
            "tracks": [t.to_dict() for t in self.tracks],
        }
        p.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")

    @classmethod
    def load(cls, path: str | Path) -> "Manifest":
        d = json.loads(Path(path).read_text())
        return cls(
            tracks=[Track.from_dict(t) for t in d["tracks"]],
            version=d.get("version", MANIFEST_VERSION),
        )

    def assign_holdout(self, n: int) -> None:
        """Mark the last `n` tracks (by sorted track_id) as held out.

        Deterministic: depends only on track ids, so the catalog/negative split
        is stable across runs and machines.
        """
        for t in self.tracks:
            t.held_out = False
        for t in sorted(self.tracks, key=lambda t: t.track_id)[len(self.tracks) - n :]:
            t.held_out = True
