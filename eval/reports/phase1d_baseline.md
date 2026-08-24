# Phase 1D — Match Decision Baseline

**Recognizer:** `landmark_fingerprint` (fingerprint format v1)  
**Generated:** 2026-08-24T09:17:34+00:00  
**Repo commit:** `89552a4f6b5e`  
**Working tree: DIRTY** — the commit above does not contain the exact code that ran.
  
**Harness fingerprint:** `495c1c6bfa8e0b85…`

> This report does **not** replace `eval/reports/baseline.md`. The Phase 0
> baseline (Recall@1 29.46%, FAR 100%) stands unchanged as the reference.

## Decision rule

```
evidence_score = aligned_landmarks / query_landmarks
MATCH  iff  evidence_score >= 0.019048
       and  aligned_landmarks >= 5
```

The score is a **rate, not a probability**. It is bounded by [0,1] because it is a fraction of landmarks, not because it is calibrated against outcome frequencies.

## Dataset and splits

- Manifest hash `4006aacb0abc1e7f…`, split hash `b560583c9edab461…`
- Catalog **32** tracks, held out **12**
- Queries **1854** — reused verbatim from the Phase 0 run
- Index: 32 tracks, 854,018 postings, 10.2 MB, built in 23.76s
- Split policy: by track id, interleaved; no track appears in both
- Calibration: 927 queries (864 pos / 63 neg)
- Evaluation: 927 queries (864 pos / 63 neg)

## Threshold selection

- FAR target: **0.001** (0.1%)
  - on the calibration split: **met**
  - on the held-out evaluation split: **NOT MET** — this is the number that counts
- Selected on the calibration split only: **0.019048**
- Smallest FAR the calibration negatives can resolve: **1.59%** (63 negatives)
- Smallest FAR the evaluation negatives can resolve: **1.59%**

## Headline results (evaluation split — threshold NOT fitted on this data)

| Metric | Calibration | **Evaluation** | All queries | Phase 0 |
|---|---:|---:|---:|---:|
| Recall@1 | 82.18% | **79.40%** | 80.79% | 29.46% |
| FAR | 0.00% | **3.17%** | 1.59% | 100.00% |
| Correct rejection | 100.00% | **96.83%** | 98.41% | 0.00% |
| Precision | 99.86% | **99.42%** | 99.64% | — |
| True positives | 710 | **686** | 1396 | — |
| Wrong accepts | 1 | **2** | 3 | — |
| False negatives | 153 | **176** | 329 | — |
| False accepts | 0 | **2** | 2 | — |

Ranking-only Recall@1 (matcher top-1 before the decision layer): **84.03%** on the evaluation split. The gap to the post-decision figure is what rejection costs.

### Every false accept, itemized

| Query | Category | Split | Matched | Evidence | Aligned | Concentration |
|---|---|---|---|---:|---:|---:|
| `ia_Andrejigon-zeleniOblaki_375__neg_music__d` | out_of_catalog_music | evaluation | `ia_04MonoSyntaxEchoes` | 0.0224 | 10 | 0.047 |
| `ia_amroque-antalio-times-synthonatic-therapy` | out_of_catalog_music | evaluation | `ia_04MonoSyntaxEchoes` | 0.0200 | 9 | 0.050 |

> **Diagnosis only, not an operating point.** A threshold of 0.0245 would give 0/63 false accepts at Recall@1 79.05% — but that threshold is read off the evaluation split itself, so quoting it as a result would be exactly the leakage the calibration split exists to prevent.

## Threshold trade-off (calibration split)

| Threshold | Recall@1 | FAR | Precision | Correct rejection | TP | FP |
|---:|---:|---:|---:|---:|---:|---:|
| 0.0000 | 85.30% | 68.25% | 88.16% | 31.75% | 737 | 43 |
| 0.0055 | 85.07% | 49.21% | 89.52% | 50.79% | 735 | 31 |
| 0.0091 | 83.68% | 22.22% | 96.40% | 77.78% | 723 | 14 |
| 0.0190 | 82.18% | 0.00% | 99.86% | 100.00% | 710 | 0 |
| 0.0779 | 77.31% | 0.00% | 100.00% | 100.00% | 668 | 0 |
| 0.1406 | 72.34% | 0.00% | 100.00% | 100.00% | 625 | 0 |
| 0.2098 | 67.25% | 0.00% | 100.00% | 100.00% | 581 | 0 |
| 0.2443 | 62.27% | 0.00% | 100.00% | 100.00% | 538 | 0 |
| 0.2750 | 56.83% | 0.00% | 100.00% | 100.00% | 491 | 0 |
| 0.2993 | 51.74% | 0.00% | 100.00% | 100.00% | 447 | 0 |
| 0.3260 | 46.41% | 0.00% | 100.00% | 100.00% | 401 | 0 |
| 0.3618 | 41.32% | 0.00% | 100.00% | 100.00% | 357 | 0 |
| 0.3816 | 34.49% | 0.00% | 100.00% | 100.00% | 298 | 0 |
| 0.4144 | 28.24% | 0.00% | 100.00% | 100.00% | 244 | 0 |
| 0.4549 | 22.80% | 0.00% | 100.00% | 100.00% | 197 | 0 |
| 0.5117 | 17.59% | 0.00% | 100.00% | 100.00% | 152 | 0 |
| 0.5814 | 12.04% | 0.00% | 100.00% | 100.00% | 104 | 0 |
| 0.7097 | 6.83% | 0.00% | 100.00% | 100.00% | 59 | 0 |
| 0.9860 | 1.50% | 0.00% | 100.00% | 100.00% | 13 | 0 |

## Positive results by condition (evaluation split)

| Condition | Queries | Recall@1 | Recall@3 | No-match | p50 ms |
|---|---:|---:|---:|---:|---:|
| `clean_duration_3s` | 48 | 77.08% | 77.08% | 18.75% | 16.1 |
| `clean_duration_5s` | 48 | 89.58% | 89.58% | 10.42% | 25.9 |
| `clean_duration_10s` | 48 | 95.83% | 95.83% | 4.17% | 50.3 |
| `codec_mp3_128k_duration_3s` | 16 | 100.00% | 100.00% | 0.00% | 16.9 |
| `codec_mp3_128k_duration_5s` | 16 | 100.00% | 100.00% | 0.00% | 28.7 |
| `codec_mp3_128k_duration_10s` | 16 | 100.00% | 100.00% | 0.00% | 57.6 |
| `codec_mp3_32k_duration_3s` | 16 | 100.00% | 100.00% | 0.00% | 19.0 |
| `codec_mp3_32k_duration_5s` | 16 | 100.00% | 100.00% | 0.00% | 29.9 |
| `codec_mp3_32k_duration_10s` | 16 | 100.00% | 100.00% | 0.00% | 56.5 |
| `codec_mp3_64k_duration_3s` | 16 | 100.00% | 100.00% | 0.00% | 18.1 |
| `codec_mp3_64k_duration_5s` | 16 | 100.00% | 100.00% | 0.00% | 28.8 |
| `codec_mp3_64k_duration_10s` | 16 | 100.00% | 100.00% | 0.00% | 56.6 |
| `codec_opus_32k_duration_3s` | 16 | 100.00% | 100.00% | 0.00% | 19.4 |
| `codec_opus_32k_duration_5s` | 16 | 100.00% | 100.00% | 0.00% | 31.6 |
| `codec_opus_32k_duration_10s` | 16 | 100.00% | 100.00% | 0.00% | 61.8 |
| `codec_opus_64k_duration_3s` | 16 | 100.00% | 100.00% | 0.00% | 18.3 |
| `codec_opus_64k_duration_5s` | 16 | 100.00% | 100.00% | 0.00% | 30.5 |
| `codec_opus_64k_duration_10s` | 16 | 100.00% | 100.00% | 0.00% | 61.5 |
| `filter_lowpass8k_duration_3s` | 16 | 100.00% | 100.00% | 0.00% | 17.0 |
| `filter_lowpass8k_duration_5s` | 16 | 100.00% | 100.00% | 0.00% | 29.1 |
| `filter_lowpass8k_duration_10s` | 16 | 100.00% | 100.00% | 0.00% | 59.3 |
| `filter_telephone_duration_3s` | 16 | 100.00% | 100.00% | 0.00% | 14.4 |
| `filter_telephone_duration_5s` | 16 | 100.00% | 100.00% | 0.00% | 22.2 |
| `filter_telephone_duration_10s` | 16 | 100.00% | 100.00% | 0.00% | 45.8 |
| `noise_pink_snr0db_duration_3s` | 16 | 81.25% | 81.25% | 18.75% | 21.4 |
| `noise_pink_snr0db_duration_5s` | 16 | 75.00% | 75.00% | 25.00% | 33.7 |
| `noise_pink_snr0db_duration_10s` | 16 | 75.00% | 75.00% | 25.00% | 65.2 |
| `noise_pink_snr10db_duration_3s` | 16 | 87.50% | 87.50% | 12.50% | 19.5 |
| `noise_pink_snr10db_duration_5s` | 16 | 93.75% | 93.75% | 6.25% | 31.2 |
| `noise_pink_snr10db_duration_10s` | 16 | 93.75% | 93.75% | 6.25% | 58.7 |
| `noise_pink_snr20db_duration_3s` | 16 | 93.75% | 93.75% | 6.25% | 18.1 |
| `noise_pink_snr20db_duration_5s` | 16 | 93.75% | 93.75% | 6.25% | 29.5 |
| `noise_pink_snr20db_duration_10s` | 16 | 93.75% | 93.75% | 6.25% | 60.0 |
| `noise_pink_snr5db_duration_3s` | 16 | 81.25% | 81.25% | 18.75% | 20.2 |
| `noise_pink_snr5db_duration_5s` | 16 | 87.50% | 87.50% | 12.50% | 32.3 |
| `noise_pink_snr5db_duration_10s` | 16 | 87.50% | 87.50% | 12.50% | 59.7 |
| `noise_white_snr0db_duration_5s` | 16 | 81.25% | 81.25% | 18.75% | 25.9 |
| `noise_white_snr10db_duration_5s` | 16 | 93.75% | 93.75% | 6.25% | 28.6 |
| `noise_white_snr20db_duration_5s` | 16 | 93.75% | 93.75% | 6.25% | 30.5 |
| `noise_white_snr5db_duration_5s` | 16 | 87.50% | 87.50% | 12.50% | 27.8 |
| `pitch_+1st_duration_5s` | 16 | 0.00% | 0.00% | 100.00% | 23.6 |
| `pitch_+2st_duration_5s` | 16 | 0.00% | 0.00% | 100.00% | 24.9 |
| `pitch_-1st_duration_5s` | 16 | 0.00% | 0.00% | 100.00% | 23.5 |
| `pitch_-2st_duration_5s` | 16 | 0.00% | 0.00% | 100.00% | 25.7 |
| `speed_+2pct_duration_5s` | 16 | 0.00% | 0.00% | 100.00% | 25.9 |
| `speed_+5pct_duration_5s` | 16 | 0.00% | 0.00% | 100.00% | 25.9 |
| `speed_-2pct_duration_5s` | 16 | 0.00% | 0.00% | 100.00% | 25.6 |
| `speed_-5pct_duration_5s` | 16 | 0.00% | 0.00% | 100.00% | 26.7 |

### By family and duration (evaluation split)

| Slice | Queries | Recall@1 | Recall@3 |
|---|---:|---:|---:|
| family = clean | 144 | 87.50% | 87.50% |
| family = codec | 240 | 100.00% | 100.00% |
| family = filter | 96 | 100.00% | 100.00% |
| family = noise | 256 | 87.50% | 87.50% |
| family = pitch | 64 | 0.00% | 0.00% |
| family = speed | 64 | 0.00% | 0.00% |
| duration = 3.0 s | 224 | 91.07% | 91.07% |
| duration = 5.0 s | 416 | 64.42% | 64.42% |
| duration = 10.0 s | 224 | 95.54% | 95.54% |

## Negative results (all negatives, by category)

| Category | Queries | False accepts | FAR | Correct rejection |
|---|---:|---:|---:|---:|
| `negative_near_silence` | 3 | 0 | 0.00% | 100.00% |
| `negative_noise_pink` | 3 | 0 | 0.00% | 100.00% |
| `negative_noise_white` | 3 | 0 | 0.00% | 100.00% |
| `negative_out_of_catalog_music` | 108 | 2 | 1.85% | 98.15% |
| `negative_silence` | 3 | 0 | 0.00% | 100.00% |
| `negative_speech` | 6 | 0 | 0.00% | 100.00% |

## Performance (all queries)

| Stage | p50 ms | p95 ms | mean ms |
|---|---:|---:|---:|
| fingerprint | 11.96 | 23.76 | 13.83 |
| lookup | 0.68 | 1.43 | 0.76 |
| histogram | 15.89 | 63.90 | 22.84 |
| rank | 0.01 | 0.01 | 0.01 |
| total | 28.90 | 86.97 | 37.43 |

