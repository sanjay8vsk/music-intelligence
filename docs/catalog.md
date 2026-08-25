# Catalog ingestion (Stage 2)

Reference for [`musicintel/catalog/`](../musicintel/catalog/). Consumes the
Phase 1 recognizer ([`fingerprint-format.md`](fingerprint-format.md),
[`fingerprint-index.md`](fingerprint-index.md)) and changes none of it.

## What this stage adds

Phase 0–1H built a recognizer that answers *"which `track_id` is this?"*. Nothing
decided what a `track_id` **is**, or where the audio behind it came from — every
benchmark borrowed the evaluation manifest, which encodes a benchmark split
rather than an owned collection.

This is the missing half: identity, provenance, ingestion and index
construction for a catalog you own.

```
audio directory
  -> discover_audio()        recursive, sorted, extension-filtered
  -> sha256 + track_id       identity, never a filename
  -> fingerprint (cached)    the frozen Phase 1A extractor
  -> Catalog                 catalog.json — metadata only, no audio
  -> build_catalog_index()   the frozen Phase 1B builder
```

## Identity is not a filename

The prototype this replaces, `src/audio_processing.py`, derived identity with:

```python
feature_file = file.split(".")[0] + ".npy"
```

That is a silent data-loss bug, not a style problem:

| Filename | `split(".")[0]` | `Path.stem` |
|---|---|---|
| `track.mp3` | `track` | `track` |
| `Song feat. Artist.mp3` | **`Song feat`** | `Song feat. Artist` |
| `a.b.flac` | **`a`** | `a.b` |
| `mix.1.wav` | **`mix`** | `mix.1` |
| `mix.2.wav` | **`mix`** | `mix.2` |

`mix.1.wav` and `mix.2.wav` collapse onto one name, so ingesting both writes one
file twice and **one recording disappears without an error**. A catalog whose
identities silently merge cannot be trusted to say which recording matched.

Two id modes, neither of which can reproduce that:

- **`stem`** (default) — `Path.stem`, so interior dots survive.
- **`content`** — the first 16 hex characters of the audio SHA-256. Opaque, but
  the same audio gets the same id however the file is named.

**A duplicate `track_id` raises** rather than overwriting. `--on-duplicate-id skip`
keeps the first and records the rest in `report.skipped`. Silent overwrite is not
offered.

`source_path` is recorded as **provenance only**. Moving or renaming a file does
not change what was ingested, and the catalog's content hash is path-independent.

## The catalog file

`catalog.json` holds metadata and no audio:

| Field | Meaning |
|---|---|
| `track_id` | the identity the recognizer returns |
| `source_path` | where the audio was read from — provenance, not identity |
| `sha256` | content identity of the audio |
| `duration_sec`, `bytes`, `fingerprint_count` | shape |
| `title`, `artist` | optional metadata |

Plus `content_hash` — SHA-256 over sorted `(track_id, sha256)` pairs, so it
depends on *which audio is in the catalog* and not on ordering, paths or mutable
metadata.

`Catalog.verify()` reports empty or duplicate ids, malformed hashes,
non-positive durations and missing files; with `check_hashes=True` it also
detects audio that changed underneath a path. `duplicate_content()` reports one
recording held under two ids — legal, but always worth surfacing, because at
query time those ids are indistinguishable.

## Fingerprint cache

Ingestion caches fingerprints by content hash, so re-ingesting an unchanged
library costs a hash read instead of a decode — measured **4.12 s → 0.06 s** on a
four-track library (100% hit rate).

The cache key is `{sha256}.fp{FORMAT_VERSION}.{config_digest}.npz`. Keying on
audio content **alone would be wrong**: after a config change it would serve
fingerprints made under different settings, which is exactly the error the
index's own format-version check exists to prevent. A corrupt entry is treated
as a miss, and writes are atomic via a temp file, so an interrupted run cannot
leave a torn entry.

## Determinism

Two ingests of the same directory produce the same catalog hash and the same
index hash. Three things make that true:

- `discover_audio` sorts by **full path**, so track ordinals never depend on
  filesystem order (`os.listdir` in the prototype did).
- `build_catalog_index` orders tracks by sorted `track_id`, so the index is
  reproducible from the catalog alone.
- The Phase 1B index is already canonical and byte-stable.

## Usage

```
python scripts/ingest_catalog.py AUDIO_DIR --out catalog/main
python scripts/ingest_catalog.py AUDIO_DIR --out catalog/main --id-mode content
python scripts/ingest_catalog.py AUDIO_DIR --out catalog/main --verify
```

Writes `<out>/catalog.json` and `<out>/index/`. Audio is never copied or moved.
`--verify` re-hashes every file after ingestion.

Non-audio files are skipped by an explicit extension list rather than by trying
to decode everything — a catalog directory routinely contains artwork and text.
An undecodable file is skipped and recorded, never fatal.

## Limitations

- **The index is rebuilt, not extended.** Phase 1B's index is immutable by
  design; adding a track means rebuilding. The fingerprint cache makes that
  cheap in decode terms, but the whole index is still reconstructed.
- **The whole catalog is fingerprinted in memory** during ingestion. Fine at
  corpus scale; a large catalog wants streaming.
- **No metadata enrichment.** `title`/`artist` are accepted but nothing
  populates them; there is no tag reader and no MusicBrainz lookup.
- **No deletion or update path.** Re-ingest is the only way to change a catalog.
- Recognition quality is entirely the Phase 1 recognizer's; this stage adds no
  accuracy and changes no threshold.
