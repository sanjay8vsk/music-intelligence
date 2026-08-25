# Multi-tenant catalogs and the recognition service (Stage 2)

Reference for [`musicintel/catalog/store.py`](../musicintel/catalog/store.py) and
[`musicintel/service/recognition.py`](../musicintel/service/recognition.py).
Builds on [`catalog.md`](catalog.md) and the frozen Phase 1 recognizer.

This is the production wiring for:

```
audio -> catalog store -> frozen recognizer -> gated speed cascade -> match / no-match
```

It adds no DSP, no hashing, no matching, no scoring and no thresholds of its
own. Every accuracy figure the system has was measured on the components
underneath and is unchanged by this layer.

## Tenant isolation is structural

The roadmap (`PRODUCT_ASSESSMENT.md` §18, Stage 2) asks for *"`catalog_id` on
every hash row with isolation tests"*. That shape tags each posting and filters
at query time — isolation then holds only as long as every lookup path remembers
the filter, and one forgotten filter leaks another tenant's catalog.

**Each catalog gets its own index artifact instead.** A query against catalog A
loads A's index; B's postings are not in the array being searched. There is no
filter to forget. It also leaves `musicintel/recognition/index.py` untouched,
which the Phase 1 freeze requires.

The cost is stated plainly: no cross-catalog query, and more resident memory
when many catalogs are open at once.

## Artifact layout

```
<store>/<catalog_id>/
    catalog.json     identity, provenance, licensing — no audio
    index/           the Phase 1B index artifact, format unchanged
    artifact.json    binds the two and versions the pair
```

`artifact.json` records the catalog's content hash, the index's content hash,
the track and fingerprint counts, and both format versions. That is what makes
the acceptance criterion *"index artifact reproducible from the manifest"*
checkable: rebuild from the same audio, compare hashes.

Loading refuses anything it cannot vouch for — unknown artifact version,
malformed JSON, a `catalog_id` that disagrees with its directory (a renamed or
copied catalog), a catalog or index whose content hash has drifted, a catalog
and index holding different track ids, or a fingerprint format this build cannot
produce. Saving refuses a catalog and index that disagree, because an index
built from different audio than the catalog describes would return tracks the
catalog cannot explain.

Catalog ids are directory names, so they are validated against
`^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$` — `../etc` and friends are rejected before
any path is constructed.

## The service

```python
store = CatalogStore("/var/lib/musicintel/catalogs")
svc = RecognitionService(store)
result = svc.identify_file("clip.wav", catalog_id="acme")
```

`Identification` carries the verdict, the **track** (not just its id), the stage
that produced it, the rate correction if the cascade escalated, the evidence
score and threshold, and `offset_seconds` — where in the reference recording the
query landed. That offset is free from the matcher's histogram and is the basis
of timestamped recognition.

The offset is measured against the reference index, so it is already in
reference time even when the query was rate-corrected.

`evidence_score` is a **rate, not a probability**, exactly as in the decision
layer. There is no confidence field, and a test asserts none appears.

Catalogs are cached between queries — building an index costs seconds, a query
costs milliseconds. `unload()` drops them.

## Thresholds

Reproduced from `eval/reports/phase1h_gated_benchmark.md`, the calibrated
operating point:

| Threshold | Value |
|---|---|
| stage 1 | 0.026316 |
| concentration gate | 0.032520 |
| stage 2 | 0.028571 |
| min aligned landmarks | 5 |
| rate grid | −4%, −2%, +2%, +4% |
| probe | 2.0 s |

Changing any of these invalidates every published number.

## Known limitation: the gate threshold is catalog-size sensitive

**Measured, and it matters.** The same +2% speed-changed query, same code, same
thresholds:

| Catalog | stage-1 concentration | Gate (0.032520) | Outcome |
|---|---:|---|---|
| 3 tracks | 0.020408 | blocked | NO_MATCH |
| 32 tracks (benchmark) | 0.034091 | passed | MATCH, stage 2, −2% |

The gate was calibrated on the 32-track benchmark catalog. On a much smaller
catalog the stage-1 concentration statistic shifts and the gate can block
queries the cascade would otherwise recover. Stage-1 matching is unaffected —
only escalation to the speed cascade is.

This is a property of the calibration, not of the wiring, and it is **not**
worked around here: silently loosening the gate would invalidate the benchmarked
FAR. A production deployment on a catalog materially different in size from 32
tracks should re-derive the gate on its own calibration split, using the
methodology in `eval/reports/phase1h_gated_benchmark.md`.

## CLI

```
python scripts/catalogctl.py add      STORE CATALOG_ID AUDIO_DIR
python scripts/catalogctl.py list     STORE
python scripts/catalogctl.py describe STORE CATALOG_ID
python scripts/catalogctl.py identify STORE CATALOG_ID AUDIO_FILE
```

`identify` exits 0 on MATCH and 1 on NO_MATCH. `scripts/ingest_catalog.py`
remains the single-catalog tool; `catalogctl` is the multi-tenant front end over
the same ingestion code.

## Other limitations

- **The index is rebuilt, not extended** — inherited from Phase 1B, whose index
  is immutable by design. Adding one track to a catalog rebuilds that catalog.
- **No cross-catalog query.** The direct consequence of structural isolation.
- **Whole catalog fingerprinted in memory** during ingestion.
- **No Postgres, no MusicBrainz/AcoustID enrichment, no object storage.** All
  three are named in the roadmap's Stage 2 and are deliberately out of scope
  here; catalogs are filesystem artifacts.
- **Not validated at 500 tracks.** The roadmap's Stage 2 acceptance asks for a
  500-track catalog end-to-end; only 63 distinct recordings exist locally.
- **No update or delete lifecycle** — re-ingest is still the way to change a
  catalog.
