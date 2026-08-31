# Stage 2 — 500-track scale validation

**Generated:** 2026-08-31T12:56:35+00:00  
**Repo commit:** `6ce12854adee`  
**Working tree:** DIRTY (39 paths)  
**Phase 1 fingerprint:** `991ca192fcd54555798eb82e159f376ba552c3a6912d175d0cef8f530847ac34`

> Measurement only. No threshold, recognition module or cascade was changed,
> and the frozen Phase 0/1 corpora and reports were read for comparison only.

## Stage 2 acceptance

> ingest a 500-track catalog end-to-end; duplicates detected by content hash; cross-tenant isolation proven by test; index artifact reproducible from the manifest

| Criterion | Result |
|---|---|
| Ingest a 500-track catalog end-to-end | **500 tracks**, 612 s, identify verified |
| Duplicates detected by content hash | **500 distinct hashes / 500 tracks**, 0 duplicate groups, 0 overlap with the frozen corpus |
| Cross-tenant isolation proven | own **MATCH**, other tenant **NO_MATCH** -> **isolated** |
| Index artifact reproducible from the manifest | catalog hash identical **True**, index hash identical **True** |

## Corpus

- **500** distinct recordings, **27.97 h** audio, 288 distinct artists
- Licences: {'CC-BY': 159, 'CC-BY-SA': 242, 'CC0-1.0': 99}
- Audio on disk: 3.02 GB (git-ignored)
- Manifest content hash: `35b1045b94fe4457b2243dfe900f2e902f67cb472d915e1ec568e413451e4e06`

## Ingestion and index

| Quantity | Value |
|---|---:|
| Cold ingestion | 612 s (0.8 tracks/s, 165x realtime) |
| Index build | 3.9 s |
| Artifact save | 0.7 s |
| Warm re-ingestion | 9.4 s (100% cache hits, 64.9x faster) |
| Postings | 14,574,966 (29,150/track, 144.8/audio-second) |
| Unique hashes | 1,250,872 |
| Index in memory | 174.9 MB (12 bytes/posting) |
| Artifact on disk | 175.1 MB |
| Peak RSS during ingestion | 630 MB |
| Peak Python allocation | 731 MB |

peak RSS covers the whole ingestion, which holds every track's fingerprints in memory at once -- the known limitation this run exists to quantify.

## Latency

200 in-catalog 5 s queries, top-1 correct 200/200.

| Stage | p50 ms | p95 ms |
|---|---:|---:|
| Index lookup | 1.45 | 3.51 |
| Fingerprint | 5.97 | 7.18 |
| Offset histogram | 24.23 | 61.84 |
| **End-to-end identify** | **36.93** | **76.11** |

**Against the roadmap's Stage 1 criterion** — *"p95 lookup < 50 ms at 1,000 tracks"*: measured here at 500 tracks as a scale-readiness signal only. This run does NOT claim the 1,000-track criterion is satisfied.


