# Offset-histogram matcher (Phase 1C)

Reference for [`musicintel/recognition/matcher.py`](../musicintel/recognition/matcher.py).
Prerequisites: [`fingerprint-format.md`](fingerprint-format.md) and
[`fingerprint-index.md`](fingerprint-index.md).

## Scope — read this first

Phase 1C **ranks candidates**. It does **not** decide whether the winner is real.

`MatchCandidate.score` is a **count of aligned landmarks**. It is **not a
probability, not a confidence, and not calibrated**. A query of pure noise will
still produce a ranked list with a non-zero top score, because some offset
always wins by chance.

**NO_MATCH is Phase 1D.** Thresholds, calibration and rejection are deliberately
absent. Anything consuming this module today must treat the top candidate as a
hypothesis, not an answer. An empty candidate list means *no query hash appeared
in the index at all* — it is not a rejection decision either.

The Phase 0 baseline (Recall@1 29.46%, FAR 100%) stands unchanged in
[`eval/reports/baseline.md`](../eval/reports/baseline.md). Nothing here has been
measured against it.

## How hash matches are retrieved

Query landmarks are `(hash, anchor_frame)` pairs. For each one the matcher takes
the equal-range of its hash in the index and retrieves **every posting** — not
the first, not the nearest. There is no distance metric and no vector search.

Because the index hashes are sorted, all query hashes are located with two
vectorized `np.searchsorted` calls, and the ragged `[lo, hi)` spans are expanded
into one flat array of row positions. One pass, no per-landmark Python loop.

## How temporal offsets are calculated

For every retrieved posting:

```
offset = db_anchor_frame - query_anchor_frame
```

in STFT frames (11.61 ms each at the default hop). Offsets are grouped **per
track** — the whole point is that different tracks are separate hypotheses.
Offsets may be negative; nothing assumes the query starts at or after the
reference.

Offsets are never averaged. An average over a true spike plus scattered noise is
neither the spike nor the noise; the cluster must be found, not smoothed.

## Why a true recording creates a spike

If the query is a recording of track T beginning `k` frames into it, then every
landmark the two genuinely share satisfies

```
db_anchor_frame - query_anchor_frame == k
```

for the **same** `k`. All true matches vote for one offset.

Coincidental collisions have no reason to agree. A 28-bit key over ~150
landmarks per second collides constantly — 34.5% of hashes in a small real index
already carry more than one posting — but those collisions land at whatever
offsets the two recordings happen to place them, spread across the track's whole
span.

So volume of shared hashes is weak evidence and *agreement about time* is strong
evidence. Measured on a 6-track index, a full-track self-query scored 44,070
aligned landmarks while the runner-up track scored 15 — from a comparable pool
of raw hits. That gap is the whole thesis.

## Offset binning and tolerance

Clusters are found with a **sliding window over sorted offsets**, not fixed bins.
Fixed bins put hard edges at arbitrary offsets, and a cluster straddling an edge
is split in half. A window has no edges to straddle.

`MatchConfig.offset_tolerance_frames` (default **2**) is the maximum spread
within one window, so a window covers offsets `[k, k+2]` — three frames, about
**34.8 ms** at the default hop of 128 samples @ 11025 Hz.

**Why not 0 (exact equality).** Query and reference are decoded separately and
their STFT frame grids need not line up. An excerpt starting mid-hop shifts every
peak's frame index, and a peak near a neighbourhood boundary can be picked one
frame either side. Both move the offset by a frame or two without the audio
differing at all.

**Why not large.** Widening the window scoops up unrelated collisions, and a
sufficiently wide window makes every track look aligned. Two frames is roughly
one hop of slack in each direction — enough for grid and boundary drift, not
enough to manufacture a spike.

This was chosen from the frame geometry, **not** tuned against the evaluation
corpus. Synthetic tests pin the intended behaviour: drift of 0–2 frames is
absorbed at the default, splits into three separate clusters at tolerance 0, and
merges fully at tolerance 8.

`best_offset` is reported as the **modal exact offset inside the winning
window** (ties toward the smaller value), so it stays meaningful even at
tolerance 0.

## Duplicate handling

Two different inflation risks, handled explicitly:

1. **One query landmark matching many postings.** A hash occurring 221 times in
   one track would contribute 221 raw hits from a single piece of query evidence.
2. **A repeated hash within the query.**

The ranking score therefore counts **distinct query landmarks** that align at the
winning offset, not raw hits. One query landmark is one unit of evidence however
many postings its hash has. The sliding window maintains a multiset of query
landmark indices and tracks the distinct count as it moves.

Raw hits remain visible as `best_offset_count`, which may legitimately exceed
`score` — a unit test asserts exactly that case (3 postings within tolerance of
one query landmark → `best_offset_count == 3`, `score == 1`).

## Ranking

Candidates are ordered by:

1. `score` — distinct query landmarks aligned at the best offset (descending)
2. `best_offset_count` — raw hits in that window (descending)
3. `matched_query_landmarks` (descending)
4. `track_id` (ascending) — a total order, so equal evidence still ranks
   deterministically and insertion order never leaks into the result

`MatchResult` exposes `best`, `top(k)`, `top_ids(k)` and `top_id`, so top-1 and
top-3 are directly available. `max_candidates` truncates the output only; ranking
always considers every track with a hit.

## What each field means

| Field | Meaning |
|---|---|
| `score` | distinct query landmarks aligned at `best_offset` — **the primary evidence, a count** |
| `best_offset` / `best_offset_seconds` | where the query sits inside the reference |
| `best_offset_count` | raw hits inside the winning window |
| `total_hits` | every posting retrieved for this track, aligned or not |
| `matched_query_landmarks` | distinct query landmarks touching this track at any offset |
| `second_best_offset` / `second_best_score` | best cluster **disjoint** from the winner, so it is a genuine alternative alignment rather than the same peak shifted by a frame |
| `concentration` | `score / matched_query_landmarks` — a dispersion measure, **not** a confidence |
| `margin` | `score - second_best_score` |

## Measured behaviour

6 corpus tracks (142–299 s), index of 204,441 postings. **Functional checks, not
accuracy claims** — the queries are drawn from the indexed audio itself.

Full-track self-match: **6/6 ranked themselves first**, all at offset 0 with
concentration 1.000. Runner-up scores were 9–1,069 against winning scores of
20,256–44,070.

Additional probe — a 10 s excerpt from the middle of each track:
**6/6 ranked correctly, and every recovered offset exactly equalled the expected
frame position** (e.g. 12,880 vs 12,880). Concentration 0.53–0.92, match time
10–64 ms.

Per-stage timing over six full-track queries:

| Stage | Time | Share |
|---|---:|---:|
| Fingerprint extraction | 3.083 s | 42.1% |
| Index lookup | 0.104 s | 1.4% |
| Histogram aggregation | 4.132 s | 56.5% |
| Ranking | <0.001 s | 0.0% |

## Limitations

- **No decision.** Ranking only; NO_MATCH, thresholds and calibration are 1D.
- **The score is a raw count.** It is not comparable across queries of different
  lengths — a longer query simply has more landmarks to align.
- **Histogram dominates full-track queries** (~706 ms each): the cluster search
  is a Python loop over that track's hits, and a full-track self-query produces
  tens of thousands. Realistic short queries cost 10–64 ms. Left unoptimized on
  purpose; correctness and inspectability come first.
- **No cap on postings per hash.** A pathologically hot hash in a large catalog
  would pull in a large posting list. Harmless at corpus scale, needs a guard
  before a big catalog.
- **Pitch and speed shifts** still break the underlying exact-integer keys; the
  matcher inherits that limitation from the fingerprint format.
- **Self-match proves plumbing, not accuracy.** Every measurement above queries
  audio that is in the index. Real accuracy requires the Phase 0 degradation
  benchmark, which has not been run.
