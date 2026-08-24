# Phase 1E — Landmark Recognizer Benchmark

**Recognizer:** `landmark@5a6ab3f+dirty` (fingerprint format v1, index format v1)  
**Generated:** 2026-08-24T09:45:29+00:00  
**Repo commit:** `5a6ab3f590bc`  
**Working tree:** **DIRTY** — 2 uncommitted paths; the commit above does not contain the exact code that ran  
**Phase 1 pipeline fingerprint:** `c31b8e453653558a773014c49faf14cc8c55d0dd5247bed459b4995d163e3511`

> The Phase 0 baseline (`eval/reports/baseline.md`, Recall@1 29.46%, FAR 100%)
> is **not modified** by this run and remains the reference point.

## Provenance

| Field | Value |
|---|---|
| Git commit | `5a6ab3f590bc808f20148ecc154091ebd4b9446f` |
| Git dirty | `True` (2 paths) |
| **Phase 1 source fingerprint** | `c31b8e453653558a773014c49faf14cc8c55d0dd5247bed459b4995d163e3511` |
| Harness fingerprint | `803a68f3d1df509f4a4c7b724e9537d4e92057375642a329405befa4ca62a721` |
| Phase 0 algorithm fingerprint | `d53b191a2a98f3ed57ee42dc932c5506732ab8ba20a4ca1f1ea2b6c0db3f98fc` |
| Fingerprint format version | 1 |
| Index format version | 1 |
| Manifest hash | `4006aacb0abc1e7f2e12eee8a9f205a6f6b8cc563fef7a9396872cf797767a0e` |
| Split hash | `b560583c9edab46151106ac84ef2bacc786b6d870870a905bb814152581a24d0` |
| Catalog / held out | 32 / 12 |
| Queries evaluated | 1854 |
| Python | 3.11.1 on macOS-26.6.1-arm64-arm-64bit |

Files covered by the Phase 1 fingerprint:

- `musicintel/recognition/fingerprint.py`
- `musicintel/recognition/index.py`
- `musicintel/recognition/matcher.py`
- `musicintel/recognition/decision.py`
- `scripts/eval_phase1e.py`

## Configuration

```
fingerprint : 11025 Hz, n_fft 1024, hop 128, band 200-3000 Hz,
              30.0 peaks/s, fan_out 5, dt 1-128 frames
matcher     : offset tolerance 2 frames
decision    : aligned_landmarks / query_landmarks
              MATCH iff score >= 0.018088 and aligned >= 5
index       : 32 tracks, 854,018 postings, 10.2 MB, built in 26.26s
```

The decision score is a **rate, not a probability**.

## Splits and threshold

- Policy: by track id, interleaved; no track appears on both sides
- Calibration: 927 queries (864 pos / 63 neg) — threshold selected here
- Evaluation: 927 queries (864 pos / 63 neg) — headline numbers reported here
- FAR target 0.001: met on calibration, **NOT met** on held-out data

## Phase 0 vs Phase 1 — headline

| Metric | Phase 0 (MFCC/FAISS) | **Phase 1 held-out** | Phase 1 all queries |
|---|---:|---:|---:|
| Recall@1 | 29.46% | **79.63%** | 80.96% |
| FAR | 100.00% | **3.17%** | 1.59% |
| Correct rejection | 0.00% | **96.83%** | 98.41% |
| Precision | — | **99.28%** | 99.57% |
| p50 latency | 19.48 ms | **30.43 ms** | |
| p95 latency | 58.23 ms | **97.29 ms** | |

Ranking-only Recall@1 (matcher top-1, before the decision layer): **85.01%** over all queries. The gap to the accepted figure is the cost of rejection.

### By family — every family, improvement or regression

| Family | Phase 0 R@1 | Phase 1 R@1 | Δ | Verdict |
|---|---:|---:|---:|---|
| clean | 31.25% | 90.63% | +59.38 pp | improves |
| codec | 42.08% | 100.00% | +57.92 pp | improves |
| filter | 27.60% | 100.00% | +72.40 pp | improves |
| noise | 10.35% | 89.84% | +79.49 pp | improves |
| pitch | 42.97% | 0.78% | -42.19 pp | **regresses** |
| speed | 43.75% | 3.91% | -39.84 pp | **regresses** |

## Per-condition results (all queries; Phase 0 denominators)

| Condition | n | P0 R@1 | P1 R@1 | Δ | P1 R@3 | Verdict |
|---|---:|---:|---:|---:|---:|---|
| `clean_duration_10s` | 96 | 38.54% | 97.92% | +59.38 pp | 97.92% | improves |
| `clean_duration_3s` | 96 | 25.00% | 82.29% | +57.29 pp | 82.29% | improves |
| `clean_duration_5s` | 96 | 30.21% | 91.67% | +61.46 pp | 91.67% | improves |
| `codec_mp3_128k_duration_10s` | 32 | 43.75% | 100.00% | +56.25 pp | 100.00% | improves |
| `codec_mp3_128k_duration_3s` | 32 | 40.62% | 100.00% | +59.38 pp | 100.00% | improves |
| `codec_mp3_128k_duration_5s` | 32 | 46.88% | 100.00% | +53.12 pp | 100.00% | improves |
| `codec_mp3_32k_duration_10s` | 32 | 34.38% | 100.00% | +65.62 pp | 100.00% | improves |
| `codec_mp3_32k_duration_3s` | 32 | 37.50% | 100.00% | +62.50 pp | 100.00% | improves |
| `codec_mp3_32k_duration_5s` | 32 | 37.50% | 100.00% | +62.50 pp | 100.00% | improves |
| `codec_mp3_64k_duration_10s` | 32 | 43.75% | 100.00% | +56.25 pp | 100.00% | improves |
| `codec_mp3_64k_duration_3s` | 32 | 40.62% | 100.00% | +59.38 pp | 100.00% | improves |
| `codec_mp3_64k_duration_5s` | 32 | 43.75% | 100.00% | +56.25 pp | 100.00% | improves |
| `codec_opus_32k_duration_10s` | 32 | 43.75% | 100.00% | +56.25 pp | 100.00% | improves |
| `codec_opus_32k_duration_3s` | 32 | 43.75% | 100.00% | +56.25 pp | 100.00% | improves |
| `codec_opus_32k_duration_5s` | 32 | 43.75% | 100.00% | +56.25 pp | 100.00% | improves |
| `codec_opus_64k_duration_10s` | 32 | 43.75% | 100.00% | +56.25 pp | 100.00% | improves |
| `codec_opus_64k_duration_3s` | 32 | 40.62% | 100.00% | +59.38 pp | 100.00% | improves |
| `codec_opus_64k_duration_5s` | 32 | 46.88% | 100.00% | +53.12 pp | 100.00% | improves |
| `filter_lowpass8k_duration_10s` | 32 | 37.50% | 100.00% | +62.50 pp | 100.00% | improves |
| `filter_lowpass8k_duration_3s` | 32 | 34.38% | 100.00% | +65.62 pp | 100.00% | improves |
| `filter_lowpass8k_duration_5s` | 32 | 37.50% | 100.00% | +62.50 pp | 100.00% | improves |
| `filter_telephone_duration_10s` | 32 | 18.75% | 100.00% | +81.25 pp | 100.00% | improves |
| `filter_telephone_duration_3s` | 32 | 18.75% | 100.00% | +81.25 pp | 100.00% | improves |
| `filter_telephone_duration_5s` | 32 | 18.75% | 100.00% | +81.25 pp | 100.00% | improves |
| `noise_pink_snr0db_duration_10s` | 32 | 6.25% | 81.25% | +75.00 pp | 81.25% | improves |
| `noise_pink_snr0db_duration_3s` | 32 | 6.25% | 81.25% | +75.00 pp | 81.25% | improves |
| `noise_pink_snr0db_duration_5s` | 32 | 6.25% | 81.25% | +75.00 pp | 81.25% | improves |
| `noise_pink_snr10db_duration_10s` | 32 | 9.38% | 93.75% | +84.37 pp | 93.75% | improves |
| `noise_pink_snr10db_duration_3s` | 32 | 9.38% | 90.62% | +81.24 pp | 90.62% | improves |
| `noise_pink_snr10db_duration_5s` | 32 | 9.38% | 93.75% | +84.37 pp | 93.75% | improves |
| `noise_pink_snr20db_duration_10s` | 32 | 21.88% | 96.88% | +75.00 pp | 96.88% | improves |
| `noise_pink_snr20db_duration_3s` | 32 | 21.88% | 93.75% | +71.87 pp | 93.75% | improves |
| `noise_pink_snr20db_duration_5s` | 32 | 21.88% | 93.75% | +71.87 pp | 93.75% | improves |
| `noise_pink_snr5db_duration_10s` | 32 | 9.38% | 90.62% | +81.24 pp | 90.62% | improves |
| `noise_pink_snr5db_duration_3s` | 32 | 6.25% | 87.50% | +81.25 pp | 87.50% | improves |
| `noise_pink_snr5db_duration_5s` | 32 | 6.25% | 90.62% | +84.37 pp | 90.62% | improves |
| `noise_white_snr0db_duration_5s` | 32 | 3.12% | 81.25% | +78.13 pp | 81.25% | improves |
| `noise_white_snr10db_duration_5s` | 32 | 9.38% | 93.75% | +84.37 pp | 93.75% | improves |
| `noise_white_snr20db_duration_5s` | 32 | 12.50% | 96.88% | +84.38 pp | 96.88% | improves |
| `noise_white_snr5db_duration_5s` | 32 | 6.25% | 90.62% | +84.37 pp | 90.62% | improves |
| `pitch_+1st_duration_5s` | 32 | 46.88% | 0.00% | -46.88 pp | 0.00% | regresses |
| `pitch_+2st_duration_5s` | 32 | 43.75% | 0.00% | -43.75 pp | 0.00% | regresses |
| `pitch_-1st_duration_5s` | 32 | 46.88% | 0.00% | -46.88 pp | 0.00% | regresses |
| `pitch_-2st_duration_5s` | 32 | 34.38% | 3.12% | -31.26 pp | 3.12% | regresses |
| `speed_+2pct_duration_5s` | 32 | 40.62% | 6.25% | -34.37 pp | 6.25% | regresses |
| `speed_+5pct_duration_5s` | 32 | 40.62% | 0.00% | -40.62 pp | 0.00% | regresses |
| `speed_-2pct_duration_5s` | 32 | 46.88% | 9.38% | -37.50 pp | 9.38% | regresses |
| `speed_-5pct_duration_5s` | 32 | 46.88% | 0.00% | -46.88 pp | 0.00% | regresses |

### By position and duration (all queries)

| Slice | Queries | Recall@1 | Recall@3 | No-match |
|---|---:|---:|---:|---:|
| position = beginning | 96 | 95.83% | 95.83% | 3.12% |
| position = middle | 1536 | 80.34% | 80.34% | 19.60% |
| position = end | 96 | 76.04% | 76.04% | 21.88% |
| duration = 3 s | 448 | 92.86% | 92.86% | 6.25% |
| duration = 5 s | 832 | 65.99% | 65.99% | 34.01% |
| duration = 10 s | 448 | 96.88% | 96.88% | 3.12% |

## Negatives and rejection

| Category | Queries | False accepts | FAR | Correct rejection |
|---|---:|---:|---:|---:|
| `near_silence` | 3 | 0 | 0.00% | 100.00% |
| `noise_pink` | 3 | 0 | 0.00% | 100.00% |
| `noise_white` | 3 | 0 | 0.00% | 100.00% |
| `out_of_catalog_music` | 108 | 2 | 1.85% | 98.15% |
| `silence` | 3 | 0 | 0.00% | 100.00% |
| `speech` | 6 | 0 | 0.00% | 100.00% |

**Resolution limit.** The held-out split has **63 negatives**, so a single false accept is worth **1.5873 percentage points** and the smallest resolvable non-zero FAR is **1.5873%**. Observing a 0.1% FAR at all needs on the order of **1,000 negatives**; the corpus has 126 in total. **No figure here should be read as a production FAR.** Expanding the negative fixture set is a future requirement, deliberately not done inside this run because it would change the corpus methodology mid-comparison.

Every false accept, itemized:

| Query | Category | Split | Matched | Evidence | Aligned / landmarks |
|---|---|---|---|---:|---:|
| `ia_Andrejigon-zeleniOblaki_375__neg_musi` | out_of_catalog_music | evaluation | `ia_04MonoSyntaxEchoes` | 0.0224 | 10 / 447 |
| `ia_amroque-antalio-times-synthonatic-the` | out_of_catalog_music | evaluation | `ia_04MonoSyntaxEchoes` | 0.0200 | 9 / 450 |

## Latency

| Stage | p50 ms | p95 ms | p99 ms | mean ms |
|---|---:|---:|---:|---:|
| fingerprint | 12.29 | 25.84 | 47.91 | 15.12 |
| lookup | 0.82 | 1.98 | 2.73 | 1.00 |
| histogram | 16.53 | 67.01 | 136.71 | 24.24 |
| rank | 0.01 | 0.02 | 0.03 | 0.01 |
| decision | 0.00 | 0.00 | 0.00 | 0.00 |
| total | 30.43 | 97.29 | 165.02 | 40.38 |

Phase 0 measured p50 19.48 ms / p95 58.23 ms, but its latency excluded feature extraction on the reference side and was taken on a differently loaded machine; treat the comparison as indicative only.

## Threshold sweep (calibration split)

Stored columnar in the JSON (120 operating points, index-aligned arrays). Selected rows:

| Threshold | Recall@1 | FAR | Precision | TP | FP |
|---:|---:|---:|---:|---:|---:|
| 0.0000 | 85.30% | 68.25% | 88.16% | 737 | 43 |
| 0.0059 | 85.07% | 46.03% | 90.29% | 735 | 29 |
| 0.0109 | 83.45% | 20.63% | 97.56% | 721 | 13 |
| 0.0514 | 79.05% | 0.00% | 100.00% | 683 | 0 |
| 0.1368 | 72.92% | 0.00% | 100.00% | 630 | 0 |
| 0.2143 | 66.78% | 0.00% | 100.00% | 577 | 0 |
| 0.2511 | 60.53% | 0.00% | 100.00% | 523 | 0 |
| 0.2919 | 54.17% | 0.00% | 100.00% | 468 | 0 |
| 0.3201 | 47.69% | 0.00% | 100.00% | 412 | 0 |
| 0.3614 | 41.44% | 0.00% | 100.00% | 358 | 0 |
| 0.3908 | 32.99% | 0.00% | 100.00% | 285 | 0 |
| 0.4313 | 26.16% | 0.00% | 100.00% | 226 | 0 |
| 0.4966 | 19.44% | 0.00% | 100.00% | 168 | 0 |
| 0.5594 | 12.96% | 0.00% | 100.00% | 112 | 0 |
| 0.7405 | 6.37% | 0.00% | 100.00% | 55 | 0 |

## Limitations

- FAR is measured against 63 held-out negatives, so one false accept moves it by 1.5873 percentage points and the smallest resolvable non-zero FAR is 1.5873%. A 0.1% target needs roughly 1,000 negatives and cannot be demonstrated here.
- Speed and pitch conditions fail completely: the fingerprint key packs exact integer frequency-bin indices, and resampling or pitch shifting moves every bin. This is a representation limitation, confirmed at the hash level, not a matcher or threshold problem.
- The catalog is 32 tracks. Recognition difficulty grows sharply with catalog size, so these numbers are an OPTIMISTIC upper bound.
- The all-queries column shares half its positives with the calibration split. It exists for like-for-like comparison with Phase 0 denominators; the evaluation-split column is the unbiased result.
- Degradations are synthetic; no real acoustic captures are included.
- The decision score is a rate, not a calibrated probability.

