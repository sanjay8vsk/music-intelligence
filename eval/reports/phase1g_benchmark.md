# Phase 1G — Higher-Resolution False-Accept Benchmark

**Recognizer:** `landmark@c420d00` — unchanged from Phase 1E  
**Generated:** 2026-08-25T07:13:54+00:00  
**Repo commit:** `c420d0054dc2`  
**Working tree:** clean  
**Phase 1 fingerprint:** `50f815b788d325ad3fa9f58c278f670d139c5add14363c491d650194c6f58305`

> **What changed: the negative sample size only.** The recognizer, the 1,728
> positive queries and the original 126 negatives are all reused unchanged.
> `eval/reports/baseline.md`, `phase1d_baseline.md` and `phase1e_benchmark.md`
> are untouched.

## Negative set

- **1361 negatives** from **31 source recordings**
- Evaluation split: **626 negatives** from **15 recordings**
- **One false accept = 0.1597 percentage points** (Phase 1E: 1.5873 pp)
- Smallest resolvable non-zero FAR: **0.1597%** (Phase 1E: 1.5873%)

| Category | Count |
|---|---:|
| `near_silence` | 18 |
| `noise_pink` | 18 |
| `noise_white` | 18 |
| `out_of_catalog_music` | 1151 |
| `silence` | 18 |
| `speech` | 12 |

**Excerpt count is not sample size.** Clips from one recording share mastering and
instrumentation and fail together, so the effective diversity is the source-recording
count, quoted above alongside every FAR figure.

## Results (evaluation split — threshold fitted only on calibration)

| Metric | Calibration | **Evaluation** | All queries |
|---|---:|---:|---:|
| Recall@1 | 81.13% | **78.47%** | 79.80% |
| FAR | 0.0000% | **0.1597%** | 0.0735% |
| Correct rejection | 100.00% | **99.84%** | 99.93% |
| Precision | 99.86% | **99.85%** | 99.86% |
| TP | 701 | **678** | 1379 |
| FP | 0 | **1** | 1 |
| TN | 735 | **625** | 1360 |
| FN | 162 | **186** | 348 |

### FAR by negative category (evaluation split)

| Category | Negatives | False accepts | FAR |
|---|---:|---:|---:|
| `near_silence` | 14 | 0 | 0.0000% |
| `noise_pink` | 10 | 0 | 0.0000% |
| `noise_white` | 13 | 0 | 0.0000% |
| `out_of_catalog_music` | 569 | 1 | 0.1757% |
| `silence` | 8 | 0 | 0.0000% |
| `speech` | 12 | 0 | 0.0000% |

**Aggregate FAR depends on category mix.** Silence and noise are trivially rejected, so including them lowers the aggregate without improving anything. Out-of-catalog music is the number that matters.

### FAR by cohort (evaluation split)

| Cohort | Negatives | False accepts | FAR |
|---|---:|---:|---:|
| phase1e | 63 | 0 | 0.0000% |
| phase1g | 563 | 1 | 0.1776% |

## Comparison with the frozen Phase 1E benchmark

| | Phase 1E (frozen) | Phase 1G |
|---|---:|---:|
| Evaluation negatives | 63 | **626** |
| Source recordings behind them | 6 | **15** |
| 1 false accept = | 1.5873 pp | **0.1597 pp** |
| FAR | 3.1746% | **0.1597%** |
| Recall@1 | 79.63% | 78.47% |

> Phase 1E and Phase 1G measure DIFFERENT negative populations. Phase 1E's 63 evaluation negatives came from 6 held-out recordings; Phase 1G adds newly fetched sources and disjoint excerpting. The FAR figures are therefore NOT directly equivalent — what improved is resolution, not the system.

## Latency

| Stage | p50 ms | p95 ms | mean ms |
|---|---:|---:|---:|
| fingerprint | 17.64 | 31.67 | 20.34 |
| lookup | 0.51 | 1.22 | 0.61 |
| histogram | 8.88 | 33.99 | 12.58 |
| rank | 0.01 | 0.01 | 0.01 |
| total | 27.53 | 67.28 | 33.53 |

## Every false accept, itemized

| Query | Category | Cohort | Matched | Evidence | Aligned |
|---|---|---|---|---:|---:|
| `neg__ia_alg056__d3s__t163.0` | out_of_catalog_music | phase1g | `ia_32EP02Descent_1` | 0.0299 | 13 |

## Limitations

- Excerpt count is not statistical sample size. Clips from one recording share mastering and instrumentation and fail together; the effective diversity is the source-recording count, reported alongside every FAR figure.
- Aggregate FAR depends on category mix. Silence and noise are trivially rejected, so adding them lowers aggregate FAR without improving the system. Out-of-catalog MUSIC is the number that matters and is reported separately.
- The catalog is still 32 tracks; difficulty grows with catalog size.
- Negative sources are CC-licensed netlabel music, not a genre-balanced sample of commercial music.

