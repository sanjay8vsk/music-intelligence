# MusicBrainz metadata enrichment (Stage 2)

Enrichment attaches third-party descriptive metadata — album, release date,
ISRC, MusicBrainz identifiers — to tracks that already exist in a catalog.

**It has nothing to do with recognition.** No fingerprint, threshold, index or
matching decision is read or written here, and nothing in this package is
imported by the API. A test asserts both: `musicintel/api/` and
`musicintel/recognition/` contain no reference to enrichment or MusicBrainz.

AcoustID and Chromaprint are **deliberately absent**. AcoustID would require a
second, incompatible fingerprint format (Chromaprint) and a native `fpcalc`
binary; it is deferred, not forgotten.

## Two kinds of metadata, kept apart

| | Owned identity | Corpus metadata | MusicBrainz metadata |
|---|---|---|---|
| Where | `tracks` | `tracks.title`, `tracks.artist` | `track_metadata` |
| Origin | ingestion | the catalog owner, via sidecar | the MusicBrainz web service |
| Authority | ours | the customer's claim | a third party's claim |
| Mutable by enrichment | never | never | yes, that is its purpose |

Keeping them in separate tables is not tidiness. Two concrete reasons:

1. `CatalogRepository.sync()` writes `title = EXCLUDED.title` from
   `catalog.json`. Anything enriched into `tracks.title` would be **erased by
   the next catalog re-sync**.
2. Enrichment written back into `catalog.json` changes its bytes without
   changing `catalog_content_hash` or `index_content_hash` — the same
   content-addressed artifact key with different content, which the storage
   layer correctly refuses.

## The metadata sidecar

`ingest_paths(..., metadata=load_sidecar(path))` attaches `title` and `artist`
at ingestion. It is optional; catalogs ingested without one are unchanged and
remain valid. Only those two fields are read — a richer file can be passed
without smuggling unvalidated columns into the catalog.

Entries may be keyed by `sha256` or `track_id`; `sha256` wins, because it
identifies the audio itself and survives a change of `id_mode`.

**This cannot change recognition identity.** Neither content hash reads `title`
or `artist`, and `scripts/backfill_catalog_metadata.py` verifies that
empirically: it computes both hashes before and after and **refuses to write if
either moves**.

## Running enrichment

Offline CLI, never the API:

```bash
MUSICINTEL_MUSICBRAINZ_CONTACT="you@example.com" \
python scripts/enrich_musicbrainz.py --dsn "$DSN" --catalog acme
```

It is restartable: already-enriched tracks are skipped unless `--force` or
`--max-age-days` is given, so an interrupted run resumes.

At one request per second, a 500-track catalog takes **at least 8 minutes**.
That is why this is a worker and not a request handler or a start-up step.

### What the run reports

`http_dispatches_per_sec` is **HTTP dispatches per second over this run's request
window**, computed as `(dispatches - 1) / window`.

- **Dispatches**, not responses: a request that was sent and then timed out
  reached the server and is counted. The client's own `requests_made` counts
  completed responses and cannot see it, so the two legitimately differ.
- **`(n - 1)`**, not `n`: *n* requests occupy *n-1* intervals, so dividing the
  count by the span it covers reports a rate higher than the one sustained.
- **This run only.** The client's lifetime counters are snapshotted at the start
  and reported as deltas, so a second `enrich()` on the same client does not
  inherit the first run's requests.
- **The window is measured at lookup boundaries** — first lookup start to last
  lookup end — because the client exposes no per-dispatch timestamps. That is
  marginally wider than the true first-to-last dispatch span, so the rate is
  slightly conservative and never overstated.
- **Fewer than two dispatches reports `0.0`.** One request spans no interval;
  a rate is not invented from a zero-length window.

`dispatch_window_seconds` and `http_dispatches` are reported alongside it so the
figure can be rechecked.

This is a measure of what **this process** did. It is **not** a statement about
compliance with MusicBrainz's policy and **not** a machine- or network-wide
rate: the limiter coordinates every process on the machine, so concurrent
workers can each be under this figure while the host is not.

### Outcomes

| Status | Meaning |
|---|---|
| `matched` | one clear best candidate |
| `ambiguous` | top candidates too close to choose between |
| `no_match` | nothing scored above the threshold |
| `error` | transport or payload failure, after bounded retries |
| `skipped` | the track has no title/artist to search with |

### The classification thresholds are uncalibrated

Two numbers decide those outcomes:

| Threshold | Value | Role |
|---|---:|---|
| `min_score` | **80.0** | the minimum score a top candidate must reach; below it the result is `no_match` |
| `ambiguity_margin` | **5.0** | if the top two candidates are within this many points, the result is `ambiguous` rather than `matched` |

**Both are implementation choices, not MusicBrainz policy.** MusicBrainz
publishes no recommended threshold for either, and neither number is derived
from anything they document.

**Neither has been calibrated against real data.** The Stage 2 live corpus could
not validate them: the observed responses were overwhelmingly empty — 18 of the
18 live `no_match` results returned *zero* candidates rather than low-scoring
ones — so the 80.0 floor was never exercised as a discriminator. The single
persisted match returned exactly **one** candidate, so the 5.0 margin was never
exercised at all.

They should therefore not be described as optimal, empirically validated, or
representative of any MusicBrainz recommendation. They are defaults that have
not yet been tested by the data they exist to judge, and both are constructor
arguments so a deployment can change them without touching this code.

Calibrating them needs multi-candidate responses with known-correct answers —
that is, a corpus MusicBrainz actually holds. The Stage 2 evaluation corpus is
obscure netlabel material and returned a match for 1 of 20 tracks, so it cannot
supply that evidence.

### A failed row explains itself

Failures used to collapse to `match_status='error'` plus the query: a `400` (the
query is malformed, so retrying is pointless) and a `503` retry-exhaustion
(transient, so re-run later) were byte-identical on disk. Two bounded columns
close that:

| Column | Contents |
|---|---|
| `error_detail` | the client's own bounded diagnostic string — `HTTP 503`, `wall-clock cap exceeded`, `connection error: …` |
| `attempts` | how many lookup attempts were made |

```
track  status     att  score   error_detail
t1     error        1  NULL    HTTP 400        <- not retried; the query is at fault
t2     error        3  NULL    HTTP 503        <- retries exhausted; transient
t3     matched      2  100.00  NULL            <- a retry was needed, and worked
t4     no_match     1  NULL    NULL
```

`attempts` is recorded for **every** outcome, not only failures. On a successful
row it is the only trace that a retry was needed — transient trouble that
resolved itself, which the request counter cannot show because it never counts a
timed-out request. A `skipped` row has no attempt count, because nothing was
looked up.

**`error_detail` is bounded to 512 characters**, chosen from measurement rather
than taste. Classifying diagnostics are short — `HTTP 503` is 8 characters, the
longest fixed message 33 — but a connection failure embeds the full request URL
and measured **326 characters** with a short title and **404** with the longest
real corpus title. Truncation is safe by construction: every diagnostic puts its
classification *first*, so cutting the tail costs URL detail and never the
failure class. The repository truncates; a CHECK constraint is the backstop.

**What `error_detail` is not.** It is the client's view of the failure, not a
server-side explanation. `HTTP 503` records that MusicBrainz refused the
request; it does not say why they refused it. **Error response bodies are
deliberately not persisted** — the client never reads a body on an error status,
and storing remote payloads to explain a failure is a cost with no diagnostic
return.

**Licensing:** `error_detail` and `attempts` are *operational* data this
application produced — a status code and our own exception text. They are a
third category, distinct from the CC0 core fields and from
`match_score`/`raw_response`, which may carry supplementary search-index data.

**`ambiguous` is never promoted to `matched`.** Recording a match when the
lookup could not choose would manufacture certainty the data does not support.
Likewise, tracks without title and artist are recorded as `skipped` rather than
searched by their filename-derived `track_id`, which would produce confident
wrong answers.

## Request behaviour

| Setting | Value | Source |
|---|---|---|
| Connect / read timeout | `(10, 20)` s | repository convention |
| Attempts | 3, bounded | repository convention |
| Per-track wall-clock cap | 60 s | repository convention |
| Retried | 429, 502, 503, 504, timeouts | — |
| **Not** retried | all other 4xx, malformed payloads | — |
| Minimum request interval | **2.0 s** | our operating default (see below) |
| User-Agent with real contact | **required** | **MusicBrainz policy** |

The User-Agent-with-contact row is MusicBrainz's own published requirement, not
a convention of this repository. It is honoured because calling a free service
without doing so gets clients blocked, and deserves to.

**The 2.0 s interval is ours, not theirs.** MusicBrainz publishes a limit of *on
average one request per second*. We ship 2.0 s because measurement showed what
1.0 s actually costs:

| Interval | Lookups | HTTP | 200 | 503 | Retries | Final errors |
|---|---:|---:|---:|---:|---:|---:|
| 3.0 s | 12 | 12 | 12 | 0 | 0 | 0 |
| **2.0 s** | 12 | 12 | **12** | **0** | **0** | **0** |
| 1.0 s (previously shipped) | 8 | 17 | 6 | **11** | 9 | 2 |

At 1.0 s the first six responses were 503 despite measured request spacing of
≥1.0 s, so the refusals were server behaviour rather than a limiter defect.

Three things to keep apart:

- **2.0 s is the smallest interval we *tested* that was 503-free.** It is not the
  threshold. 1.5 s was never tested, so the true server-side boundary lies
  somewhere in (1.0, 2.0] and is **unknown**.
- **2.0 s is not MusicBrainz's formal policy.** It is an application-side
  operating choice, configurable per deployment via `min_interval`.
- **Nothing here establishes that 2.0 s is faster overall.** It avoided the
  503/retry behaviour in one sample of 32 lookups in a single session; that a
  lower request count outweighs the longer wait is plausible but **not
  measured**.

### The throttle is machine-wide, not per process

MusicBrainz limits per source IP, so a per-process timer is not enough. The
first live experiment measured exactly that failure: because each CLI invocation
built a fresh client whose in-memory timestamp started at zero, run 2's first
request left **16 ms** after run 1's last, even though every gap *within* each
run was ≥ 1 s.

The limiter is **machine-wide**: it serialises every process of the same user on
the same machine, not merely requests within one client.

`RateLimiter` therefore keeps the last-request time in a small file under the
system temporary directory and holds an advisory `flock` on it for the whole of
`acquire()`, **including the sleep**. A second process entering `acquire()`
blocks until the first has waited and stamped, then computes its own wait from
that fresh timestamp -- so two invocations that start simultaneously serialise
instead of both concluding they may go first. Measured after the change, the
same boundary is **1.004 s**.

Details worth knowing:

- **Wall clock, not monotonic**, because `time.monotonic()` epochs are not
  comparable between processes. A clock moving backwards waits a full interval
  rather than trusting a negative elapsed time, and every wait is clamped to
  `[0, min_interval]`.
- **Crash-safe**: `flock` is released by the kernel when the descriptor closes,
  so an abnormal exit leaves no lock. A stale timestamp can only make the next
  caller wait longer, never less.
- **Degrades rather than fails**: an unwritable state directory falls back to
  in-process throttling, which is no weaker than the previous behaviour.
- **Scope**: it coordinates processes of one user on one machine -- the shape of
  this deployment. It does **not** coordinate across hosts sharing an outbound
  IP; that would need a shared store, and nothing in Stage 2 requires one.
- Spacing is measured between request *starts*. Server-observed arrival gaps
  vary by a few tens of milliseconds either side of the interval because
  dispatch latency varies; the enforced spacing at `acquire()` is never below
  the interval.

## Contact configuration is mandatory

`MUSICINTEL_MUSICBRAINZ_CONTACT` (or `--contact`) must be set. The client
**refuses to construct without it** rather than sending a placeholder.

No default is shipped. The repository's corpus fetcher uses
`"contact via repo"`, which is tolerable for a handful of archive.org calls and
is not appropriate at MusicBrainz's scale. Supply a real email address or
project URL at deployment.

## Attribution and licensing

**Verified from this repository:**

- The evaluation corpus carries a per-track `license` and `license_url` in
  `eval/fixtures/scale_corpus_manifest.json`: CC-BY (159), CC-BY-SA (242),
  CC0-1.0 (99). Those licences govern the **audio**, and several require
  attribution when the audio is redistributed.
- Corpus-provided `title`/`artist` come from that manifest and are recorded in
  `tracks`, distinct from anything MusicBrainz returns.
- Every MusicBrainz row records `source`, `query_used`, `fetched_at` and the
  raw response, so any field can be traced to the request that produced it and
  re-normalised without refetching.

**Verified against official MusicBrainz documentation** (`musicbrainz.org/doc/
About/Data_License` and `musicbrainz.org/doc/MusicBrainz_Database`), checked
2026-08-29:

- MusicBrainz **core data is CC0** — effectively public domain, with no
  restriction and no attribution obligation. The core-data listing names, for
  recordings, *"Title, artist credit, duration, ISRC, PUIDs, disambiguation
  comment, MBID"*; for releases, *"Title, artist credit, type, status, language,
  date, …, MBID"*; and for artists, *"Name, …, MBID"*.
- **All six fields this pipeline extracts fall in that core set**: `title`
  (recording title), `artist` (artist credit), `album` (release title),
  `release_date` (release date), `isrc`, and the three MBIDs. No
  non-commercial restriction and no attribution requirement attaches to them.
- **Supplementary data is CC BY-NC-SA 3.0** — non-commercial, attribution,
  share-alike. It covers *"user submitted annotations, tags (including genre
  associations) and ratings"*, *"derived statistics"*, **"search indexes"**,
  edit history and non-personal user data. This pipeline requests none of those
  fields (no `inc=tags`, no ratings, no annotations).
- **There is no single mandated attribution wording** for the CC0 core fields.
  MusicBrainz asks that credit be given for supplementary data, and offers a
  "powered by MusicBrainz" button as a recognised form, but prescribes no exact
  string.

**One nuance that is not CC0.** `match_score` is the Lucene relevance score from
the **search index**, and `raw_response` stores the whole search payload, which
carries that score. Search indexes are listed as *supplementary* data, so
neither field should be treated as CC0 for redistribution. Both exist for
internal provenance -- explaining why a row says what it says, and allowing
re-normalisation without refetching -- and internal use is not redistribution.
Do not export `raw_response` or `match_score` in a public or commercial data
feed without checking the CC BY-NC-SA terms first. The normalised core fields
beside them carry no such restriction.

**Still requiring a decision, not a lookup:** MetaBrainz asks commercial users
of its services to select a support tier (a $0 "Stealth Start-Up" tier exists
for early-stage non-public companies). That is a business decision about this
product, not a property of the data licence, and it is unresolved here.

`fetched_at` exists partly so that a licence change can be scoped to the rows
fetched under the previous terms.
