# Phase 1H — Speed-Tolerant Recognition Cascade

**Recognizer:** `landmark-cascade@d0d5bc4`  
**Generated:** 2026-08-25T08:03:53+00:00  
**Repo commit:** `d0d5bc46becc`  
**Working tree:** clean  
**Phase 1 fingerprint:** `4104900b318d5febc9f10eaa6a153579e27604f2da4cc8ba458e363ecbe7b53c`

> An orchestration layer only. `fingerprint.py`, `index.py`, `matcher.py` and
> `decision.py` are byte-identical to Phase 1E/1G, and the Phase 0, 1D, 1E and
> 1G reports are untouched.

## Verdict

**FALSIFICATION CRITERIA NOT ALL MET**

| # | Criterion | Target | Measured | Result |
|---|---|---:|---:|---|
| 1 | Speed recall (4 conditions) | ≥ 60% | 100.00% | **PASS** |
| 2 | Clean/noise/codec/filter vs stage 1 | ≥ −1 pp | worst +0.00 pp | **PASS** |
| 3 | Held-out FAR | ≤ 5% | 0.3195% | **PASS** |
| 4 | p50 latency | ≤ 40 ms | 144.46 ms | **FAIL** |

## Cascade configuration

```
rate grid   : [-5.0, -4.0, -3.0, -2.0, -1.0, 1.0, 2.0, 3.0, 4.0, 5.0]   (stage 1 supplies rate 0)
convention  : apply_rate(+p%) plays faster and higher; a recording captured at +p% is corrected by about -p%
stage 1     : threshold 0.026316  (frozen Phase 1G operating point, not re-fitted)
stage 2     : threshold 0.034783  (calibration split only, judged on END-TO-END cascade FAR)
min aligned : 5
```

## Headline (evaluation split)

| Metric | Stage 1 only | **Cascade** | Δ |
|---|---:|---:|---:|
| Recall@1 | 78.4700% | **85.8800%** | +7.41 pp |
| FAR | 0.1597% | **0.3195%** | +0.16 pp |
| Precision | 99.8500% | **99.4600%** | -0.39 pp |
| Correct rejection | 99.8400% | **99.6800%** | -0.16 pp |
| TP | 678 | **742** | +64 |
| FP | 1 | **2** | +1 |
| TN | 625 | **624** | -1 |
| FN | 186 | **120** | -66 |

## Cascade behaviour

- Stage-1 match rate: **44.67%** of all queries
- Stage-2 escalation rate: **55.33%**
- Stage-2 acceptance rate among escalated: **7.72%** (132 acceptances)

**Which rate hypothesis won:**

| Correction | Stage-2 acceptances |
|---|---:|
| -5% | 32 |
| -3% | 1 |
| -2% | 31 |
| +2% | 33 |
| +3% | 1 |
| +5% | 34 |

## Per-family: stage 1 vs cascade

| Family | Queries | Stage 1 | Cascade | Δ |
|---|---:|---:|---:|---:|
| clean | 288 | 88.89% | **88.89%** | +0.00 pp |
| codec | 480 | 100.00% | **100.00%** | +0.00 pp |
| filter | 192 | 100.00% | **100.00%** | +0.00 pp |
| noise | 512 | 87.50% | **87.50%** | +0.00 pp |
| pitch | 128 | 0.78% | **2.34%** | +1.56 pp |
| speed | 128 | 0.78% | **100.00%** | +99.22 pp |

## Per-condition: stage 1 vs cascade

| Condition | n | Stage 1 | Cascade | Δ |
|---|---:|---:|---:|---:|
| `clean_duration_10s` | 96 | 95.83% | 95.83% | +0.00 pp |
| `clean_duration_3s` | 96 | 80.21% | 80.21% | +0.00 pp |
| `clean_duration_5s` | 96 | 90.62% | 90.62% | +0.00 pp |
| `codec_mp3_128k_duration_10s` | 32 | 100.00% | 100.00% | +0.00 pp |
| `codec_mp3_128k_duration_3s` | 32 | 100.00% | 100.00% | +0.00 pp |
| `codec_mp3_128k_duration_5s` | 32 | 100.00% | 100.00% | +0.00 pp |
| `codec_mp3_32k_duration_10s` | 32 | 100.00% | 100.00% | +0.00 pp |
| `codec_mp3_32k_duration_3s` | 32 | 100.00% | 100.00% | +0.00 pp |
| `codec_mp3_32k_duration_5s` | 32 | 100.00% | 100.00% | +0.00 pp |
| `codec_mp3_64k_duration_10s` | 32 | 100.00% | 100.00% | +0.00 pp |
| `codec_mp3_64k_duration_3s` | 32 | 100.00% | 100.00% | +0.00 pp |
| `codec_mp3_64k_duration_5s` | 32 | 100.00% | 100.00% | +0.00 pp |
| `codec_opus_32k_duration_10s` | 32 | 100.00% | 100.00% | +0.00 pp |
| `codec_opus_32k_duration_3s` | 32 | 100.00% | 100.00% | +0.00 pp |
| `codec_opus_32k_duration_5s` | 32 | 100.00% | 100.00% | +0.00 pp |
| `codec_opus_64k_duration_10s` | 32 | 100.00% | 100.00% | +0.00 pp |
| `codec_opus_64k_duration_3s` | 32 | 100.00% | 100.00% | +0.00 pp |
| `codec_opus_64k_duration_5s` | 32 | 100.00% | 100.00% | +0.00 pp |
| `filter_lowpass8k_duration_10s` | 32 | 100.00% | 100.00% | +0.00 pp |
| `filter_lowpass8k_duration_3s` | 32 | 100.00% | 100.00% | +0.00 pp |
| `filter_lowpass8k_duration_5s` | 32 | 100.00% | 100.00% | +0.00 pp |
| `filter_telephone_duration_10s` | 32 | 100.00% | 100.00% | +0.00 pp |
| `filter_telephone_duration_3s` | 32 | 100.00% | 100.00% | +0.00 pp |
| `filter_telephone_duration_5s` | 32 | 100.00% | 100.00% | +0.00 pp |
| `noise_pink_snr0db_duration_10s` | 32 | 71.88% | 71.88% | +0.00 pp |
| `noise_pink_snr0db_duration_3s` | 32 | 75.00% | 75.00% | +0.00 pp |
| `noise_pink_snr0db_duration_5s` | 32 | 71.88% | 71.88% | +0.00 pp |
| `noise_pink_snr10db_duration_10s` | 32 | 93.75% | 93.75% | +0.00 pp |
| `noise_pink_snr10db_duration_3s` | 32 | 90.62% | 90.62% | +0.00 pp |
| `noise_pink_snr10db_duration_5s` | 32 | 93.75% | 93.75% | +0.00 pp |
| `noise_pink_snr20db_duration_10s` | 32 | 93.75% | 93.75% | +0.00 pp |
| `noise_pink_snr20db_duration_3s` | 32 | 93.75% | 93.75% | +0.00 pp |
| `noise_pink_snr20db_duration_5s` | 32 | 93.75% | 93.75% | +0.00 pp |
| `noise_pink_snr5db_duration_10s` | 32 | 90.62% | 90.62% | +0.00 pp |
| `noise_pink_snr5db_duration_3s` | 32 | 84.38% | 84.38% | +0.00 pp |
| `noise_pink_snr5db_duration_5s` | 32 | 87.50% | 87.50% | +0.00 pp |
| `noise_white_snr0db_duration_5s` | 32 | 78.12% | 78.12% | +0.00 pp |
| `noise_white_snr10db_duration_5s` | 32 | 93.75% | 93.75% | +0.00 pp |
| `noise_white_snr20db_duration_5s` | 32 | 96.88% | 96.88% | +0.00 pp |
| `noise_white_snr5db_duration_5s` | 32 | 90.62% | 90.62% | +0.00 pp |
| `pitch_+1st_duration_5s` | 32 | 0.00% | 0.00% | +0.00 pp |
| `pitch_+2st_duration_5s` | 32 | 0.00% | 0.00% | +0.00 pp |
| `pitch_-1st_duration_5s` | 32 | 0.00% | 6.25% | +6.25 pp |
| `pitch_-2st_duration_5s` | 32 | 3.12% | 3.12% | +0.00 pp |
| `speed_+2pct_duration_5s` | 32 | 3.12% | 100.00% | +96.88 pp |
| `speed_+5pct_duration_5s` | 32 | 0.00% | 100.00% | +100.00 pp |
| `speed_-2pct_duration_5s` | 32 | 0.00% | 100.00% | +100.00 pp |
| `speed_-5pct_duration_5s` | 32 | 0.00% | 100.00% | +100.00 pp |

## False accepts and rejection

Evaluation negatives: **626** excerpts from **15** source recordings. One false accept ≈ **0.1597 pp**. Excerpt count is not statistical sample size — clips from one recording fail together, so the effective diversity is the recording count.

| Category | Negatives | False accepts | FAR |
|---|---:|---:|---:|
| `near_silence` | 14 | 0 | 0.0000% |
| `noise_pink` | 10 | 0 | 0.0000% |
| `noise_white` | 13 | 0 | 0.0000% |
| `out_of_catalog_music` | 569 | 2 | 0.3515% |
| `silence` | 8 | 0 | 0.0000% |
| `speech` | 12 | 0 | 0.0000% |

| Query | Category | Stage | Correction | Matched |
|---|---|---:|---:|---|
| `ia_alg-045D-InPassing__neg_music__d3s_` | out_of_catalog_music | 2 | +2% | `ia_30-hcir-berthaJamesSp` |
| `neg__ia_alg056__d3s__t163.0` | out_of_catalog_music | 1 | +0% | `ia_32EP02Descent_1` |

## Latency

| Stage | p50 ms | p95 ms | mean ms |
|---|---:|---:|---:|
| stage 1 (every query) | 26.03 | 78.41 | 33.89 |
| stage 2 (when escalated) | 246.26 | 669.77 | 297.86 |
| **total cascade** | **144.46** | **622.29** | 198.69 |

p99 total: 844.69 ms.

**Post-hoc diagnostic — not a re-scored criterion.** Criterion 4 is judged on
the whole-corpus p50 above and stands as measured. This split is reported because
the benchmark corpus is 44% negatives and every negative escalates by design, so
the corpus mix drives the median as much as the cascade does.

| Query type | Queries | Escalation rate | p50 ms | p95 ms |
|---|---:|---:|---:|---:|
| positives | 1728 | 20.20% | 39.39 | 344.99 |
| negatives | 1361 | 99.93% | 270.93 | 734.65 |

## Is the full grid necessary? (exploratory, calibration only)

> EXPLORATORY. The evaluation used the full +/-5% @1% grid, fixed in advance. These alternatives are scored on the CALIBRATION split only and were not used to choose anything.

| Grid | Hypotheses | Calibration Recall@1 |
|---|---:|---:|
| full ±5% @1% | 10 | 88.54% |
| ±5% @2% (-4,-2,2,4) | 4 | 87.15% |
| ±5% coarse (-5,-2,2,5) | 4 | 88.54% |
| ±2% only (-2,-1,1,2) | 4 | 84.61% |

**Toward a continuous rate estimator.** Of 883 escalated calibration queries with any signal, **0.68%** had a single-peaked evidence-vs-rate profile. A single-peaked evidence-vs-rate profile is what a coarse-to-fine or continuous rate estimator would need. Measured here only; nothing of the sort is implemented.

## Limitations

- FAR rests on 626 evaluation negatives from 15 source recordings. Excerpt count is not sample size: clips from one recording fail together.
- Pitch is unaddressed by design; a rate sweep cannot invert a duration-preserving pitch shift, and pitch is not an acceptance criterion.
- Stage 2 multiplies false-accept exposure by the grid size, which is why it carries its own stricter threshold.
- The catalog is 32 tracks; difficulty grows with catalog size.

