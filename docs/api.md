# HTTP API (Stage 3)

The recognition core is frozen. Everything in `musicintel/api/` is transport,
validation, isolation and safety around it — nothing in this layer changes how
audio is fingerprinted, matched or decided.

`/v1/analyze` is deliberately absent: its BPM/key/genre dependencies are Stage 4.

## Endpoints

| Method | Path | Auth | Purpose |
|---|---|---|---|
| GET | `/v1/health` | none | Liveness and readiness |
| GET | `/v1/version` | none | Build and format versions |
| GET | `/v1/metrics` | none | Prometheus exposition |
| GET | `/v1/catalogs` | key | Catalogs this key may access |
| GET | `/v1/catalogs/{catalog_id}` | key | One catalog, paginated tracks |
| POST | `/v1/identify` | key | Identify an audio excerpt |
| GET | `/v1/openapi.json` | none | OpenAPI 3.1 specification |

```bash
curl -X POST https://api.example.com/v1/identify \
  -H "Authorization: Bearer sk_live_..." \
  -F "file=@clip.wav" \
  -F "catalog_id=acme"
```

```json
{
  "decision": "match",
  "catalog_id": "acme",
  "match": {"track_id": "acme_2", "title": null, "artist": null,
            "offset_seconds": 6.002, "rate_percent": 0.0},
  "evidence_score": 0.451613, "threshold": 0.026316,
  "aligned_landmarks": 336, "query_landmarks": 744,
  "stage": 1, "escalated": false,
  "query_duration_seconds": 5.0, "latency_ms": 12.9
}
```

`evidence_score` is **a rate, not a probability** — aligned landmarks divided by
query landmarks, exactly as in the decision layer. It is not calibrated. There is
no `confidence` field anywhere in this API, and a test asserts none appears in
the response or the published schema.

`stage` is `null` on `no_match`, because no stage accepted the query.

## Errors are RFC 9457

Every failure is `application/problem+json`:

```json
{
  "type": "https://docs.musicintel.dev/problems/rate-limit-exceeded",
  "title": "Too Many Requests", "status": 429,
  "detail": "Request rate exceeded for this API key.",
  "instance": "/v1/identify", "retry_after": 10, "limit": 6
}
```

The published spec says so too. FastAPI documents validation failures as
`application/json` carrying `HTTPValidationError`, which this service never
returns; `musicintel/api/openapi.py` rewrites every 4xx/5xx response in the
generated document to the media type and schema actually sent. Without that the
spec would lie about the shape of every error, and a generated client would break
the first time anything went wrong — which is exactly what `schemathesis` caught.

Unknown query parameters are rejected rather than ignored, so `?limt=5` fails
loudly instead of silently returning page one.

## Authentication

`Authorization: Bearer <key>` or `X-API-Key: <key>`.

- **Hashes only.** Configuration carries SHA-256 digests; a presented key is
  hashed and looked up. The service never holds a usable key, so no code path
  can print one.
- **Prefixed** `sk_live_` / `sk_test_`, so secret scanners recognise a leak.
- **`api_key_id` in logs, never the key.**
- **Revocation** is `"active": false`, effective on the next request.
- Unknown, malformed and revoked keys return the *same* 401. Distinguishing them
  would confirm which keys exist.

Keys come from `MUSICINTEL_API_KEYS` (inline JSON) or `MUSICINTEL_API_KEYS_FILE`.
Stage 3 predates Postgres by design; the record shape is the one an `api_keys`
table will hold, so moving to a database later is a repository swap.

```json
[{"key_id": "k_acme", "tenant": "acme", "key_sha256": "<sha256 of the key>",
  "catalogs": ["acme"], "scopes": ["identify", "catalogs:read"],
  "rate_limit_per_minute": 120, "rate_limit_burst": 40,
  "audio_seconds_per_day": 36000, "active": true}]
```

An empty `catalogs` list means every catalog.

## Tenant isolation

Isolation is **structural first**: each catalog is its own index artifact, so a
query against catalog A cannot reach catalog B's postings — they are not in the
array being searched. Authorisation sits on top of that, not instead of it.

A catalog the caller may not access returns **404, not 403**, with a body
byte-identical to a genuinely missing catalog. 403 would confirm the catalog
exists, which is the fact a competitor should not be able to probe for.

## Limits

Two independent limits, because they price different things.

| Limit | Scope | Key | Breach |
|---|---|---|---|
| Request rate | per API key | token bucket | `429`, `Retry-After` |
| Audio seconds | per tenant per UTC day | counter | `429`, `Retry-After` to midnight |

A caller sending 30-second clips costs thirty times more to serve than one
sending 1-second clips at the same request rate, which is why the second limit
exists. Quota is checked *before* decode and charged *after*, against the
duration actually decoded.

Both are evaluated in Lua so check-and-update is atomic; two round trips would
let concurrent workers each read "under the limit" and each allow a request.

**On Redis failure the limiter fails closed — 503, not "allow".** An availability
incident on Redis must not silently become an unmetered-traffic incident. This is
the opposite of what a cache would do, because this is not a cache.

## The decode sandbox

Untrusted audio meets a C parser in exactly one place: a short-lived subprocess
spawned per request, fed over a pipe, dead before the response is written.

Resource limits are installed as the *first* statements in that process, before
`soundfile` — and therefore libsndfile — is imported:

| Limit | Value | Why |
|---|---|---|
| `RLIMIT_FSIZE` | 0 | No temp files, by construction rather than discipline (S4) |
| `RLIMIT_NPROC` | 0 | The decoder never forks |
| `RLIMIT_CORE` | 0 | A crash must not spill decoded audio to disk |
| `RLIMIT_CPU` | 10 s | Bounds a pathological decode |
| `RLIMIT_AS` | 512 MiB | Defence in depth; enforced inconsistently on Darwin |

The process runs in isolated mode (`python -I`) with a bare environment, and the
API package is never imported into it.

**Order of refusal.** Every cheap rejection happens before every expensive one:
auth → scope → catalog authorisation → request rate → quota → size → magic bytes
→ decode → quota charge → recognition. An unauthenticated or over-quota caller
never reaches the decoder.

**Bomb defence (S2)** is a cap on frames *read*, not on what the container
claims. A 82 KB FLAC declaring ten minutes allocates the same bounded array as a
30-second file — measured: 82 KB in, 1.3 MiB allocated, then rejected `413`. Over-
length audio is refused explicitly rather than silently truncated, because
identifying the first 30 seconds of a file the caller believes was processed
whole is a worse failure than refusing it.

**Format sniffing (S3)** is magic bytes against a whitelist. Filename and
`Content-Type` are never trusted; a successful decode is the real validation.

### Why not librosa in the worker

It is the obvious choice, since `load_audio` uses it and matching the reference
decode exactly is non-negotiable. Measured, a cold worker importing librosa costs
**2.3 s** — eight times the entire Stage 3 latency budget — because librosa's
lazy loader pulls in scipy and numba on first use. The worker uses `soundfile` +
`soxr` directly at **0.12 s**, reproducing `librosa.load(..., sr=11025,
mono=True)` exactly: mean across channels, then soxr HQ resampling.

Verified against 45 real corpus tracks at 8 kHz, 22.05 kHz and 44.1 kHz:
**45/45 produced byte-identical fingerprints**, maximum sample difference
2.4e-07 — float32 epsilon.

The cost is coverage. libsndfile opens **478 of 500 corpus tracks (95.6%)**; the
remaining 4.4% are MP3 variants only ffmpeg or audioread will take, and they get
a `415` rather than a second, larger parser on untrusted input. For untrusted
audio that is the right trade: one whitelisted parser, not two.

### A multipart gotcha

Starlette treats a multipart part as binary **only if it carries a `filename=`
attribute** — an empty `filename=""` is enough, and a `Content-Type` on the part
is *not*. A part without one is parsed as a text field and its bytes are UTF-8
mangled before any application code sees them. Those bytes are unrecoverable.

The upload field is therefore declared as `bytes`, not `UploadFile`, so such a
request fails with `415` ("not a recognised audio container"), which points at
the real problem, rather than `422` ("file: Field required") for a field the
client plainly sent. Every normal client — `curl -F`, browsers, `httpx`/`requests`
`files=` — sends a filename and is unaffected.

## Start-up and warm-up

Three first-use costs land on whichever request arrives first unless they are
paid at start-up: the matcher's numba JIT, and librosa's lazy loader, which
imports scipy on the first `stft` and again on the first `resample`.

Measured against a live server, an **unwarmed first request spent 1,735 ms in
recognition versus 13 ms steady state** — entirely lazy imports, not work.
`RecognitionService.__init__` now pays all three (~3 s, reported as
`warm_up_seconds`), so anything constructed through the service is warm before it
serves. Construct the service before accepting traffic; never scale to zero.

Numba's on-disk JIT cache is deliberately **not** used — it writes `.nbi`/`.nbc`
next to the module, which fails or silently degrades on a read-only install.

## Observability

`structlog` emits one JSON event per request: route, status, duration,
`api_key_id`, `tenant`, `request_id`. Never the key, never the uploaded bytes,
never a catalog filesystem path — a log line should be safe to ship to a
third-party aggregator.

Prometheus metrics live on a private registry. `route` labels are the *template*
(`/v1/catalogs/{catalog_id}`), never the resolved path, so a caller cannot mint
unbounded label values.

`musicintel_audio_seconds_total` is the billing quantity.

## Deployment

`Dockerfile` builds a two-stage image running as an unprivileged user; ffmpeg is
deliberately absent. `fly.toml` targets Fly.io; `deploy/docker-compose.yml` runs
the API plus Redis locally.

Operational notes that are not obvious from the files:

- **Concurrency comes from processes, not threads.** The matcher's compiled
  kernel holds the GIL, so a worker serves one identify at a time.
- **Memory is set by the index, not by traffic.** A 500-track catalog measured
  600 MB peak RSS per worker.
- **Never scale to zero** — see warm-up above.
- **Redis is required.** The limiter fails closed.

## Measured performance

500-track catalog (14,574,966 postings), 200 five-second WAV queries through the
real HTTP path — TCP, multipart framing, auth, Redis round trips, the sandboxed
decode subprocess and recognition. Nothing excluded, nothing projected.

| Run | p50 | **p95** | p99 |
|---|---:|---:|---:|
| 1 | 200.4 | **272.6** | 314.2 |
| 2 | 201.4 | **296.3** | 427.4 |
| 3 | 200.0 | **294.0** | 412.5 |
| 4 | 200.0 | **275.3** | 316.7 |
| 5 | 125.7 | **173.1** | 192.7 |

Top-1 correct 200/200 in every run. **The Stage 3 bar of p95 < 300 ms is met in
all five runs**, but the worst observed p95 is only 3.7 ms under it — the margin
is within run-to-run variance on a loaded development machine, not comfortable
headroom.

Of a ~200 ms median, recognition is ~61 ms; the rest is dominated by the decode
subprocess, which re-imports numpy/soundfile/soxr on every request (~120 ms). A
persistent pool of pre-warmed sandboxed decode workers would remove most of that
while keeping process isolation. It is not implemented: the bar is met, and the
change deserves its own measurement rather than being bundled here.
