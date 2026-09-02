"""Catalog ingestion: walk -> decode -> fingerprint -> persist.

This is the shape the prototype's `src/audio_processing.py` had, rebuilt around
the Phase 1 recognizer. What carries over is the loop; what does not is how it
names things and what it produces.

THREE THINGS THE PROTOTYPE GOT WRONG, FIXED HERE
-------------------------------------------------
1. IDENTITY. `file.split(".")[0]` collapsed every dotted filename onto one name:
   `mix.1.wav` and `mix.2.wav` both became `mix`, so ingesting both silently
   destroyed one. Identity here comes from `Path.stem` (or the content hash),
   and a collision RAISES instead of overwriting.
2. OUTPUT. It wrote 26-d MFCC statistics, which Phase 0 measured at 29.46%
   Recall@1 with a 100% false-accept rate. This writes landmark fingerprints
   and a Phase 1B index.
3. IT WAS NOT RESUMABLE. Every run re-decoded everything. Fingerprints are
   cached by content hash here, so re-ingesting an unchanged catalog costs a
   hash read rather than a decode.

CACHE CORRECTNESS
-----------------
The cache key includes the fingerprint format version AND a digest of the
fingerprint config. Keying on audio content alone would silently serve
fingerprints made under different settings after a config change -- the exact
class of error the index's own format-version check exists to prevent.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import time
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from musicintel.catalog.models import Catalog, CatalogTrack, sha256_file
from musicintel.recognition.fingerprint import (
    FORMAT_VERSION,
    FingerprintConfig,
    FingerprintResult,
    fingerprint,
    load_audio,
)
from musicintel.recognition.index import FingerprintIndex, build_index

# Extensions the decoder is expected to handle. Kept explicit rather than
# "try everything": a catalog directory routinely contains artwork and text,
# and silently attempting to decode them produces noise, not tracks.
AUDIO_EXTENSIONS: tuple[str, ...] = (
    ".wav", ".mp3", ".flac", ".ogg", ".opus", ".m4a", ".aac", ".aiff", ".aif",
)


class IngestError(RuntimeError):
    """Ingestion could not produce a trustworthy catalog."""


def discover_audio(
    root: str | Path, *, extensions: Sequence[str] = AUDIO_EXTENSIONS
) -> list[Path]:
    """Every audio file under `root`, sorted, recursively.

    Sorted because ingestion order determines the index's track ordinals, and an
    unordered `os.listdir` would make two ingests of the same directory produce
    different artifacts.
    """
    root = Path(root)
    if not root.is_dir():
        raise IngestError(f"not a directory: {root}")
    exts = {e.lower() for e in extensions}
    out = [p for p in root.rglob("*")
           if p.is_file() and p.suffix.lower() in exts and not p.name.startswith(".")]
    return sorted(out)


def derive_track_id(path: str | Path, *, mode: str = "stem", sha256: str | None = None) -> str:
    """Identity for one file.

    `stem`    -- the filename without its final extension, via `Path.stem`.
                 Keeps interior dots, so `Song feat. Artist.mp3` stays intact.
    `content` -- the first 16 hex characters of the audio hash. Opaque, but the
                 same audio always gets the same id however the file is named.

    Never `split(".")[0]`.
    """
    p = Path(path)
    if mode == "stem":
        return p.stem
    if mode == "content":
        if not sha256:
            raise IngestError("content-mode ids need the file hash")
        return sha256[:16]
    raise IngestError(f"unknown id mode: {mode!r}")


def config_digest(cfg: FingerprintConfig) -> str:
    """Short digest of the fingerprint settings, for cache keying."""
    payload = json.dumps(dataclasses.asdict(cfg), sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


def _cache_path(cache_dir: Path, sha: str, cfg_digest: str) -> Path:
    return cache_dir / f"{sha}.fp{FORMAT_VERSION}.{cfg_digest}.npz"


def _cache_load(path: Path, cfg: FingerprintConfig) -> FingerprintResult | None:
    if not path.is_file():
        return None
    try:
        with np.load(path, allow_pickle=False) as z:
            return FingerprintResult(
                hashes=np.ascontiguousarray(z["hashes"]),
                anchor_frames=np.ascontiguousarray(z["anchor_frames"]),
                config=cfg,
                duration_sec=float(z["duration_sec"]),
                peak_count=int(z["peak_count"]),
            )
    except Exception:  # noqa: BLE001 -- a corrupt cache entry is a miss, not a crash
        return None


def _cache_store(path: Path, r: FingerprintResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as fh:
        np.savez(fh, hashes=r.hashes, anchor_frames=r.anchor_frames,
                 duration_sec=np.float64(r.duration_sec),
                 peak_count=np.int64(r.peak_count))
    tmp.replace(path)  # atomic, so an interrupted run cannot leave a torn entry


@dataclasses.dataclass
class IngestReport:
    """What one ingestion run did. Returned rather than printed, so callers can assert on it."""

    catalog: Catalog
    fingerprints: dict[str, FingerprintResult]
    scanned: int = 0
    ingested: int = 0
    cache_hits: int = 0
    # Tracks that received sidecar metadata. Zero when no sidecar was supplied.
    enriched: int = 0
    skipped: list[dict] = dataclasses.field(default_factory=list)
    seconds: float = 0.0

    @property
    def cache_hit_rate(self) -> float:
        return self.cache_hits / self.ingested if self.ingested else 0.0

    def summary(self) -> dict:
        return {"scanned": self.scanned, "ingested": self.ingested,
                "skipped": len(self.skipped), "cache_hits": self.cache_hits,
                "cache_hit_rate": round(self.cache_hit_rate, 4),
                "tracks": len(self.catalog),
                "fingerprints": self.catalog.total_fingerprints,
                "duration_sec": round(self.catalog.total_duration_sec, 2),
                "seconds": round(self.seconds, 2)}


@dataclasses.dataclass(frozen=True)
class SidecarMetadata:
    """Optional descriptive metadata supplied alongside the audio.

    Purely descriptive: `title` and `artist` only. Nothing here reaches
    `track_id`, `sha256`, `duration_sec` or any fingerprint, and neither
    `Catalog.content_hash()` nor `FingerprintIndex.content_hash()` reads these
    fields -- so attaching metadata cannot change catalog identity, index
    identity or recognition. A test asserts exactly that.

    Entries may be keyed by `sha256` or by `track_id`. `sha256` wins when both
    match, because it identifies the audio itself and survives a change of
    `id_mode`, whereas a track_id is a naming convention.

    This is metadata the CATALOG OWNER supplies. Metadata obtained from
    MusicBrainz lives in the `track_metadata` table and is never written here --
    keeping a customer's own claims separate from a third party's.
    """

    by_sha256: dict[str, dict] = dataclasses.field(default_factory=dict)
    by_track_id: dict[str, dict] = dataclasses.field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.by_sha256) + len(self.by_track_id)

    def lookup(self, track_id: str, sha256: str) -> dict | None:
        return self.by_sha256.get(sha256) or self.by_track_id.get(track_id)


def _clean(value) -> str | None:
    """Descriptive strings only; empty and the literal 'None' become absent."""
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() == "none":
        return None
    return text


def load_sidecar(path: str | Path) -> SidecarMetadata:
    """Read a metadata sidecar.

    Accepts either a JSON list of records or an object mapping key -> record.
    Only `title` and `artist` are read; every other field is ignored rather
    than stored, so a rich source file can be passed without smuggling
    unvalidated columns into the catalog.
    """
    raw = json.loads(Path(path).read_text())
    if isinstance(raw, dict) and "tracks" in raw:
        raw = raw["tracks"]

    by_sha: dict[str, dict] = {}
    by_id: dict[str, dict] = {}

    def record(entry: dict) -> dict:
        return {k: v for k, v in
                (("title", _clean(entry.get("title"))),
                 ("artist", _clean(entry.get("artist")))) if v is not None}

    if isinstance(raw, list):
        for entry in raw:
            if not isinstance(entry, dict):
                continue
            meta = record(entry)
            if not meta:
                continue
            sha = _clean(entry.get("sha256"))
            tid = _clean(entry.get("track_id"))
            if sha:
                by_sha[sha] = meta
            if tid:
                by_id[tid] = meta
    elif isinstance(raw, dict):
        for key, entry in raw.items():
            if not isinstance(entry, dict):
                continue
            meta = record(entry)
            if not meta:
                continue
            # A 64-character hex key is a content hash; anything else is an id.
            k = str(key)
            if len(k) == 64 and all(c in "0123456789abcdef" for c in k.lower()):
                by_sha[k.lower()] = meta
            else:
                by_id[k] = meta
    else:
        raise IngestError(f"sidecar {path} must be a JSON list or object")

    return SidecarMetadata(by_sha256=by_sha, by_track_id=by_id)


def ingest_paths(
    paths: Iterable[str | Path],
    *,
    root: str | Path = ".",
    config: FingerprintConfig | None = None,
    id_mode: str = "stem",
    cache_dir: str | Path | None = None,
    on_duplicate_id: str = "error",
    metadata: SidecarMetadata | None = None,
    verbose: bool = False,
) -> IngestReport:
    """Fingerprint each path and assemble a catalog.

    `on_duplicate_id="error"` is the default deliberately. The prototype's
    silent overwrite is how a two-track catalog becomes a one-track catalog
    without anyone noticing; refusing is the only safe default. `"skip"` keeps
    the first and records the rest in `report.skipped`.
    """
    cfg = config or FingerprintConfig()
    root = Path(root)
    cache = Path(cache_dir) if cache_dir else None
    digest = config_digest(cfg)
    t0 = time.perf_counter()

    tracks: list[CatalogTrack] = []
    fps: dict[str, FingerprintResult] = {}
    seen: dict[str, str] = {}  # track_id -> source path
    report = IngestReport(catalog=Catalog(), fingerprints={})

    for i, raw in enumerate(paths, start=1):
        p = Path(raw)
        report.scanned += 1
        try:
            sha = sha256_file(p)
            tid = derive_track_id(p, mode=id_mode, sha256=sha)
            if tid in seen:
                msg = f"duplicate track_id {tid!r} from {p} (already from {seen[tid]})"
                if on_duplicate_id == "error":
                    raise IngestError(msg)
                report.skipped.append({"path": str(p), "reason": msg})
                continue

            cpath = _cache_path(cache, sha, digest) if cache else None
            r = _cache_load(cpath, cfg) if cpath else None
            if r is not None:
                report.cache_hits += 1
            else:
                y, sr = load_audio(p, cfg)
                r = fingerprint(y, sr, cfg)
                if cpath:
                    _cache_store(cpath, r)

            try:
                rel = str(p.resolve().relative_to(root.resolve()))
            except ValueError:
                rel = str(p.resolve())

            seen[tid] = str(p)
            # Optional and additive: with no sidecar these stay None and the
            # catalog is byte-identical to one produced before sidecars existed.
            meta = metadata.lookup(tid, sha) if metadata is not None else None
            tracks.append(CatalogTrack(
                track_id=tid, source_path=rel, sha256=sha,
                duration_sec=round(float(r.duration_sec), 3),
                bytes=p.stat().st_size, fingerprint_count=len(r),
                title=(meta or {}).get("title"),
                artist=(meta or {}).get("artist")))
            if meta:
                report.enriched += 1
            fps[tid] = r
            report.ingested += 1
        except IngestError:
            raise
        except Exception as e:  # noqa: BLE001 -- one bad file must not sink the run
            report.skipped.append({"path": str(p), "reason": f"{type(e).__name__}: {e}"})
        if verbose and i % 10 == 0:
            print(f"    {i} scanned, {report.ingested} ingested, {report.cache_hits} cached")

    report.catalog = Catalog(tracks=tracks)
    report.fingerprints = fps
    report.seconds = time.perf_counter() - t0
    return report


def ingest_directory(directory: str | Path, **kw) -> IngestReport:
    """Discover audio under `directory` and ingest all of it."""
    kw.setdefault("root", directory)
    return ingest_paths(discover_audio(directory), **kw)


def build_catalog_index(
    catalog: Catalog,
    fingerprints: dict[str, FingerprintResult],
    *,
    config: FingerprintConfig | None = None,
) -> FingerprintIndex:
    """Index a catalog, in catalog order.

    Uses the Phase 1B builder unmodified. Track order follows sorted track_id so
    the index's ordinals -- and therefore its content hash -- are reproducible
    from the catalog alone.
    """
    missing = [t.track_id for t in catalog if t.track_id not in fingerprints]
    if missing:
        raise IngestError(f"no fingerprints for {len(missing)} tracks: {missing[:5]}")
    ordered = sorted(catalog.tracks, key=lambda t: t.track_id)
    return build_index(((t.track_id, fingerprints[t.track_id]) for t in ordered),
                       config=config)


__all__ = [
    "AUDIO_EXTENSIONS", "Catalog", "CatalogTrack", "IngestError", "IngestReport",
    "build_catalog_index", "config_digest", "derive_track_id", "discover_audio",
    "ingest_directory", "ingest_paths",
]
