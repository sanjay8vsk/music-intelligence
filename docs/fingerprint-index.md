# Persistent fingerprint index (Phase 1B)

Reference for [`musicintel/recognition/index.py`](../musicintel/recognition/index.py).
The fingerprints it stores are described in
[`fingerprint-format.md`](fingerprint-format.md).

Phase 1B is **storage and lookup only**. There is no matcher, no offset
histogram, no score and no accept/reject decision, so nothing here makes an
accuracy claim. The Phase 0 baseline (Recall@1 29.46%, FAR 100%) stands
unchanged in [`eval/reports/baseline.md`](../eval/reports/baseline.md).

## The problem

Extraction gives `(hash, anchor_frame)` per track. Identification needs the
inverse:

```
hash -> [(track_id, anchor_frame), ...]
```

This is a **multimap**, not a dictionary. One hash legitimately occurs many
times — across different tracks, and repeatedly inside one track. That is not a
defect to be deduplicated: measured on 8 corpus tracks (27 minutes), **34.5% of
distinct hashes carry more than one posting**, and the busiest single hash has
**221 postings spread over 6 tracks**. An index that kept only the first
occurrence would discard most of the evidence a matcher needs.

Collisions are resolved later, not here. Phase 1C will histogram
`anchor_frame_reference − anchor_frame_query` per track: coincidental collisions
scatter across offsets, a true match piles onto one. This index's only job is to
return **every** posting for a hash, exactly and quickly.

## Representation

Three parallel arrays sorted by hash, plus a track table:

| Array | dtype | Meaning |
|---|---|---|
| `hashes` | `uint32` | the packed landmark key, ascending — the sort key |
| `track_ords` | `uint32` | dense ordinal into the track table |
| `anchor_frames` | `int32` | anchor position in STFT frames |

**12 bytes per posting**, confirmed by measurement. Rows are ordered by
`(hash, track_ord, anchor_frame)`, which makes the arrays canonical.

Lookup is `np.searchsorted` for the equal-range of a key, then a slice:
**O(log N) for the search plus O(k) for the k postings**. No hashing, no
buckets, no Python object per posting. Measured at **3.4 µs per lookup** over a
237,025-posting index, misses included.

### Why not a dict

`dict[int, list[tuple[str, int]]]` reads more naturally but costs roughly an
order of magnitude more memory — a boxed int, a list, a tuple and a str
reference per posting against 12 flat bytes — and has no canonical byte form to
persist. Sorted arrays are simultaneously the simpler artifact and the smaller
one, so this is not a performance/simplicity trade.

### Why not FAISS

FAISS answers *"which stored vector is nearest to this one"*. That is a
similarity question, and it is the question the Phase 0 baseline asked — which
is exactly why it could not distinguish a matching recording from a
similar-sounding one, and returned a catalog track for silence, speech and
noise alike (FAR 1.0). Landmark lookup is a different question: **exact integer
equality**, followed by a vote on time alignment. Approximate nearest-neighbour
search is the wrong structure and is deliberately absent.

## Track identity

Identity is the caller-supplied **`track_id` string**, never a filename. Paths
are an input to `build_index_from_files` and are not retained: moving or
renaming audio does not change what was indexed, and a test asserts no path or
audio extension appears anywhere in the artifact. `track_ord` is purely a
storage detail — a dense ordinal so a posting stays 4 bytes instead of holding a
string.

Duplicate `track_id` is rejected at build time with a `ValueError`. Two tracks
under one id would make every posting for that id ambiguous, and the ambiguity
would only ever surface as a wrong match.

## Persistence format

A directory of four files — inspectable with `cat` and `numpy.load`, no
bespoke binary container:

```
<index_dir>/
    meta.json          # metadata + content hash (human-readable)
    hashes.npy         # uint32, ascending
    track_ords.npy     # uint32
    anchor_frames.npy  # int32
```

`.npy` is used deliberately over `.npz`: `.npz` is a zip, and zip entries carry
a modification timestamp, which would make two identical builds produce
different bytes. Plain `.npy` has a fixed header and raw data, so identical
arrays give identical bytes. Arrays are read with `allow_pickle=False`, so a
malicious or corrupt file cannot execute code on load.

Loading reads only this directory. **No audio is opened and nothing is
re-fingerprinted** — measured at 0.046 s for a 237k-posting index.

## Metadata

`meta.json`, written with `sort_keys=True` so its bytes are canonical:

| Field | Purpose |
|---|---|
| `index_format_version` | on-disk layout version (currently 1) |
| `fingerprint_format_version` | the fingerprint format the hashes came from |
| `content_hash` | SHA-256 identity of the index contents |
| `sample_rate` | convenience copy of the decode rate |
| `fingerprint_config` | the **complete** `FingerprintConfig`, every field |
| `track_count`, `fingerprint_count`, `unique_hash_count` | shape |
| `built_utc` | build time, or `null` |
| `tracks` | `[{track_id, fingerprint_count, duration_sec}]` — ordinal is list position |

The two versions are independent: the disk layout can change without the
fingerprint format changing, and vice versa. Both are checked on load, and the
fingerprint mismatch message says why it matters — stored hashes made under a
different format cannot be compared against freshly extracted ones.

The full config is stored rather than a digest of it, so an index can always be
explained by reading it, and so Phase 1C can verify a query was fingerprinted
with the same settings.

**No source audio is stored**, per requirement.

## Determinism

Verified, not asserted: rebuilding the 8-track index from the same catalog
produced a **byte-identical artifact** — all four files, `cmp`-equal.

Two mechanisms:

- Postings are sorted by `(hash, track_ord, anchor_frame)` via `np.lexsort`, so
  the array contents are canonical regardless of input arrival order.
- **`built_utc` is omitted from the artifact by default** (`save()` takes
  `include_timestamp=False`). A timestamp in the bytes would make two identical
  builds look different. `content_hash` excludes it by construction either way,
  so provenance can be recorded with `include_timestamp=True` without changing
  the index's identity.

Track order is preserved from the input and *does* affect identity, since it
determines the ordinals — reordering the catalog is a different index, and the
content hash says so.

## Integrity and rejection

`load()` refuses anything it cannot vouch for, raising `IndexFormatError`:

- missing directory, `meta.json`, or any array file
- `meta.json` that is not valid JSON, or not an object
- an `index_format_version` this build does not implement
- a `fingerprint_format_version` this build cannot produce
- a config with unknown keys, missing keys, or invalid values — never silently
  defaulted, since that would leave stored hashes made with unrecoverable settings
- arrays that fail to decode
- `content_hash` disagreeing with the arrays (corruption or hand-editing)
- `track_count` / `fingerprint_count` disagreeing with the arrays

`validate()` additionally rejects internally inconsistent state: array length
mismatch, wrong dtypes, **unsorted hashes** (binary search on unsorted data
returns wrong postings instead of failing, so this one is load-bearing),
out-of-range `track_ord`, negative anchor frames, and duplicate track entries.

A hash that is simply absent is **not** an error — `lookup` returns `[]`. Most
query hashes miss any given catalog; a miss is the normal case.

## Measured behaviour

8 corpus tracks, 27.2 minutes of real audio:

| Quantity | Value |
|---|---|
| Postings | 237,025 |
| Distinct hashes | 106,353 (2.23 postings per hash) |
| In-memory | 2.84 MB (12.0 bytes/posting) |
| On disk | 2.85 MB — **0.104 MB per audio-minute** ≈ 6.2 MB/audio-hour |
| Build (sort + concat, excluding fingerprinting) | 0.047 s |
| Save | 0.026 s |
| Load | 0.046 s |
| Lookup | 3.4 µs/key over 20,000 keys |

## Compatibility with Phase 1C

`lookup_raw(hash)` returns `(track_ords, anchor_frames)` as integer array views
rather than Python objects. That is the shape an offset histogram wants: the
matcher subtracts the query anchor and `bincount`s per track without allocating
per posting. Anchor times are stored as integer **frames**, not seconds, so
offset differences bin exactly instead of needing a float tolerance.

## Limitations

- **Whole index is loaded into RAM.** Fine at corpus scale (2.85 MB for 27
  minutes); a million-track catalog needs memory-mapping or sharding, neither of
  which is Phase 1B.
- **Immutable.** No incremental add or delete — adding a track means rebuilding.
  Rebuilding 8 tracks costs 0.047 s once fingerprints exist, so this is not yet
  worth solving.
- **Linear `track_entry()` scan** over the track table. Irrelevant at tens of
  tracks; would want a dict at thousands.
- **No matcher.** Storage and retrieval prove nothing about recognition accuracy.
