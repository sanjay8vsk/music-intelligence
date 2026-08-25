# Phase 1H (gated) — Latency-Reduced Speed Cascade

**Recognizer:** `landmark-gated@49f0cf2`  
**Generated:** 2026-08-25T08:32:51+00:00  
**Repo commit:** `49f0cf2f36d6`  
**Working tree:** clean  
**Phase 1 fingerprint:** `7215a6e6b8e0006271685854ac726694aeedc66efcba4b00ad6cdeab8511a873`

> Orchestration only. `fingerprint.py`, `index.py`, `matcher.py`, `decision.py` and
> `cascade.py` are all byte-identical; Phase 0/1D/1E/1G/1H reports are untouched.

## Verdict

**ALL FOUR CRITERIA PASS — candidate accepted**

| # | Criterion | Target | Measured | Result |
|---|---|---:|---:|---|
| 1 | Speed recall | ≥ 60% | 67.19% | **PASS** |
| 2 | Clean/noise/codec/filter vs stage 1 | ≥ −1 pp | worst +0.00 pp | **PASS** |
| 3 | Held-out FAR | ≤ 5% | 0.4792% | **PASS** |
| 4 | Whole-corpus p50 latency | ≤ 40 ms | 26.67 ms | **PASS** |

## Threshold derivation (calibration split only)

- **Stage 1: 0.026316** — frozen Phase 1G/1H operating point, not re-fitted
- **Stage 2: 0.028571** — highest calibration Recall@1 with end-to-end cascade FAR <= 0.001, gate open, calibration split only
- **Gate: 0.032520** — most aggressive gate whose calibration Recall@1 stays within 0.01 of the ungated cascade; among those, lowest calibration escalation. Calibration split only.
  - at that gate, calibration Recall@1 86.46%, escalation 35.65%

The evaluation split was not consulted for either threshold.

## Headline (evaluation split)

| Metric | Stage 1 only | **Gated cascade** | Δ |
|---|---:|---:|---:|
| Recall@1 | 78.4700% | **82.8700%** | +4.40 pp |
| FAR | 0.1597% | **0.4792%** | +0.32 pp |
| Precision | 99.8500% | **99.4400%** | -0.41 pp |
| Correct rejection | 99.8400% | **99.5200%** | -0.32 pp |
| TP | 678 | **716** | +38 |
| FP | 1 | **3** | +2 |
| TN | 625 | **623** | -2 |
| FN | 186 | **147** | -39 |

## Against ungated Phase 1H

| | Phase 1H (ungated) | **Phase 1H (gated)** |
|---|---:|---:|
| Escalation rate | 55.33% | **36.52%** |
| p50 latency | 144.5 ms | **26.7 ms** |
| Recall@1 | 85.88% | **82.87%** |
| FAR | 0.3195% | **0.4792%** |

## Cascade behaviour

- Stage-1 match rate: **44.67%**
- Gate pass / escalation rate: **36.52%**
- Negatives skipped by the gate: **35.71%**
- Probe pass rate: **23.70%**; stage-2 acceptances: **88**

| Winning correction | Acceptances |
|---|---:|
| -4% | 22 |
| -2% | 24 |
| +2% | 29 |
| +4% | 13 |

## Per-family: stage 1 vs gated cascade

| Family | Queries | Stage 1 | Cascade | Δ |
|---|---:|---:|---:|---:|
| clean | 288 | 88.89% | **88.89%** | +0.00 pp |
| codec | 480 | 100.00% | **100.00%** | +0.00 pp |
| filter | 192 | 100.00% | **100.00%** | +0.00 pp |
| noise | 512 | 87.50% | **87.50%** | +0.00 pp |
| pitch | 128 | 0.78% | **0.78%** | +0.00 pp |
| speed | 128 | 0.78% | **67.19%** | +66.41 pp |

## Per-condition

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
| `pitch_-1st_duration_5s` | 32 | 0.00% | 0.00% | +0.00 pp |
| `pitch_-2st_duration_5s` | 32 | 3.12% | 3.12% | +0.00 pp |
| `speed_+2pct_duration_5s` | 32 | 3.12% | 78.12% | +75.00 pp |
| `speed_+5pct_duration_5s` | 32 | 0.00% | 65.62% | +65.62 pp |
| `speed_-2pct_duration_5s` | 32 | 0.00% | 87.50% | +87.50 pp |
| `speed_-5pct_duration_5s` | 32 | 0.00% | 37.50% | +37.50 pp |

## Rejection

Evaluation negatives: **626** excerpts from **15** source recordings; one false accept ≈ 0.1597 pp. Excerpt count is not statistical sample size.

| Category | Negatives | False accepts | FAR |
|---|---:|---:|---:|
| `near_silence` | 14 | 0 | 0.0000% |
| `noise_pink` | 10 | 0 | 0.0000% |
| `noise_white` | 13 | 0 | 0.0000% |
| `out_of_catalog_music` | 569 | 3 | 0.5272% |
| `silence` | 8 | 0 | 0.0000% |
| `speech` | 12 | 0 | 0.0000% |

| Query | Category | Stage | Correction | Matched |
|---|---|---:|---:|---|
| `ia_alg-045D-InPassing__neg_music__d3s_` | out_of_catalog_music | 2 | +2% | `ia_30-hcir-berthaJamesSp` |
| `neg__ia_2018.01.24__d3s__t91.0` | out_of_catalog_music | 2 | -4% | `ia_32EP02Descent_1` |
| `neg__ia_alg056__d3s__t163.0` | out_of_catalog_music | 1 | +0% | `ia_32EP02Descent_1` |

## Latency (wall-clock, real gated run)

| Stage | p50 ms | p95 ms |
|---|---:|---:|
| stage 1 (every query) | 14.93 | 43.53 |
| probe (gate passed) | 22.52 | 35.77 |
| confirm (probe passed) | 13.19 | 33.98 |
| **total** | **26.67** | **70.78** |

p99 105.43 ms, mean 31.59 ms.

## Limitations

- FAR rests on 626 evaluation negatives from 15 source recordings; excerpt count is not statistical sample size.
- The gate's discriminative power is weak (AUC 0.639 measured in the Phase 1H investigation). It works because only ~20% of negatives need skipping, which makes the design sensitive to the corpus's negative share.
- Pitch is unaddressed by design and is not an acceptance criterion.
- The catalog is 32 tracks; difficulty grows with catalog size.

