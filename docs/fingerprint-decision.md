# Match decision and rejection (Phase 1D)

Reference for [`musicintel/recognition/decision.py`](../musicintel/recognition/decision.py).
Prerequisites: [`fingerprint-format.md`](fingerprint-format.md),
[`fingerprint-index.md`](fingerprint-index.md),
[`fingerprint-matcher.md`](fingerprint-matcher.md).

Measurements: [`eval/reports/phase1d_baseline.md`](../eval/reports/phase1d_baseline.md),
produced by `python scripts/eval_phase1d.py`. The Phase 0 baseline
[`eval/reports/baseline.md`](../eval/reports/baseline.md) is **not** modified by
that run and remains the reference point.

## What this layer adds

Phase 1C always returned whatever ranked best. Phase 0 did too, and that is why
its False Accept Rate was **100%** — it named a catalog track for speech, for
silence and for pure noise. This is the first layer allowed to say *none of
these*.

## Three distinct things

Conflating these is how a system comes to report a confident-looking `0.97`
that means nothing:

```
raw evidence     aligned landmark count from the matcher
     |           an integer; scales with query length
     v
decision score   aligned_landmarks / query_landmarks
     |           a RATE in [0,1], comparable across query lengths
     v
threshold        a number chosen from measured data on held-out tracks
     |
     v
MATCH / NO_MATCH
```

## Evidence score

```
evidence_score = aligned_landmarks / query_landmarks
```

**Numerator.** `MatchCandidate.score` — distinct query landmarks aligned at the
winning offset. Distinct, so a hash with many postings contributes one unit of
evidence rather than many; aligned, because offset agreement is what separates a
recording from a coincidence.

**Denominator.** The query's own landmark count.

**Why only one term.** Concentration, margin and total hits are all strongly
correlated with this quantity, and each would add a parameter to justify and a
dimension to the sweep. A single interpretable rate, swept in one dimension,
is worth more than a blend nobody can reason about. The other quantities are
reported as evidence on `MatchDecision` rather than folded into the score.

## Why raw evidence is not a probability

The score is a **rate**, not a probability. It is bounded by [0,1] because it is
a fraction of landmarks — not because it has been calibrated against outcome
frequencies. **A score of 0.4 does not mean "40% likely correct."**

`MatchDecision` therefore exposes no `confidence`, `probability`, `certainty` or
`likelihood` field, and a unit test asserts none appears. Turning the rate into a
genuine posterior needs far more negative data than this corpus holds: 126
negatives cannot resolve a false-accept rate below ~0.8%.

## Query-length normalization

The matcher's raw score is a count, so it grows with query length. A 10 s query
has roughly three times the landmarks of a 3 s query and therefore roughly three
times the aligned count *for the same recording*. A single count threshold would
demand more evidence of a short query than a long one purely as an artifact of
duration.

Measured directly by unit test: the same 10% alignment fraction is **30**
landmarks at one query length and **120** at another — any fixed count threshold
splits them, while the rate decides both identically.

## Decision rule

```
MATCH  iff  evidence_score      >= threshold
       and  aligned_landmarks   >= min_aligned_landmarks
```

Nothing else. No tie-breaking against the runner-up, no per-condition cases. A
rule small enough to state in two lines is one whose failures can be diagnosed.

`min_aligned_landmarks = 5` is a fixed guard, not an operating point: two
landmarks that both align by chance is a rate of 1.0 on no evidence at all. It is
held constant during the sweep so the reported trade-off curve is one-dimensional
and means what it says.

On NO_MATCH the winning track is **withheld** (`track_id is None`) so a caller
cannot read a rejected hypothesis as an answer. The ranked candidates stay on
`.candidates` for inspection.

## How the threshold was selected

Catalog tracks (32) and held-out tracks (12) were each split in two by track id,
interleaved. **Every degradation of one recording lands on the same side**, so no
audio informs both the threshold and the number the threshold is judged by.

- Calibration: 927 queries (864 positive / 63 negative) — threshold chosen here
- Evaluation: 927 queries (864 positive / 63 negative) — reported here

Selected threshold: **0.019048** — the highest-recall calibration point with
zero observed false accepts.

## Measured results

Evaluation split; the threshold was **not** fitted on this data.

| Metric | Phase 0 | **Phase 1D (held out)** |
|---|---:|---:|
| Recall@1 | 29.46% | **79.40%** |
| FAR | 100% | **3.17%** (2/63) |
| Correct rejection | 0% | **96.83%** |
| Precision | — | **99.42%** |

Ranking-only Recall@1 (matcher top-1, before the decision layer) was 84.03%; the
gap to 79.40% is what rejection costs.

### The FAR target was not met

**FAR < 0.1% was not achieved on held-out data.** It was met on the calibration
split (0/63) and did not survive the move to evaluation data (2/63 = 3.17%).
That gap is precisely what the split exists to expose.

Two further honesty constraints:

- With 63 evaluation negatives the smallest resolvable non-zero FAR is
  **1.59%**. A 0.1% target is **not measurable** on this corpus at all;
  demonstrating it would need roughly 3,000 negatives.
- Both false accepts were marginal 3 s out-of-catalog **music** queries
  (9–10 aligned landmarks of ~450, concentration below 0.05). Silence, near-
  silence, white noise, pink noise and speech were rejected **18/18** — the
  categories Phase 0 failed on completely.

A threshold of 0.0245 would give 0/63 false accepts at Recall@1 79.05%, but that
number is read off the evaluation split, so quoting it as a result would be the
leakage the split exists to prevent. It appears in the report labelled as
diagnosis only.

### Per-family results (evaluation split)

| Family | Phase 0 R@1 | Phase 1D R@1 |
|---|---:|---:|
| clean | 31.25% | **87.50%** |
| noise | 10.35% | **87.50%** |
| codec | 42.08% | **100.00%** |
| filter (telephone + low-pass) | 27.60% | **100.00%** |
| speed | 43.75% | **0.00%** |
| pitch | 42.97% | **0.00%** |

**Speed and pitch are a regression, not an improvement.** Resampling and pitch
shifting move every frequency bin, and the fingerprint key is exact integer bin
indices, so not one landmark survives. Phase 0's MFCC mean/std was a coarse
timbre average and largely indifferent to a ±2–5% speed change; this
representation is not. The limitation was predicted in
[`fingerprint-format.md`](fingerprint-format.md) and is now measured: **0 of 128
speed/pitch queries** recognized.

Excluding speed and pitch, Recall@1 on the evaluation split is **93.21%**
(736 queries). This also explains the duration pattern — 3 s 91.07%, 5 s 64.42%,
10 s 95.54% — since every speed and pitch query is 5 s: excluding them, the 5 s
bucket is **93.05%**.

### Performance (all 1,854 queries)

| Stage | p50 ms | p95 ms |
|---|---:|---:|
| Fingerprint extraction | 12.53 | 25.70 |
| Index lookup | 0.81 | 1.90 |
| Histogram matching | 16.42 | 68.20 |
| Decision scoring | <0.01 | <0.01 |
| **Total** | **30.54** | **97.04** |

Index: 32 tracks, 854,018 postings, 10.2 MB, built in 23.7 s.

## Limitations

- **FAR < 0.1% is unproven and unmeasurable here.** Held-out FAR is 3.17%
  (2/63). Expanding the negative set is the single highest-value next step.
- **Speed and pitch fail completely** — a real regression against Phase 0.
- **63 negatives per split is very few.** Every FAR figure has wide error bars;
  a single false accept moves it by 1.59 points.
- **One corpus, 32 tracks.** Recognition difficulty grows with catalog size, so
  these numbers are an optimistic upper bound.
- **The score is not calibrated.** It is a rate and is documented as one.
- **Threshold is corpus-specific.** 0.019048 was selected on this corpus at this
  catalog size and should be re-derived for any other.
- **Not production accuracy.** These are measurements on a small research corpus
  with synthetic degradations, not a claim about real-world performance.
