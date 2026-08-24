# Recognition Baseline

**Recognizer:** `mfcc_faiss_baseline` (version `prototype@3c3981c`)  
**Generated:** 2026-08-24T08:10:24+00:00  
**Repo commit:** `3c3981c3ccfb`  
**Working tree:** clean

## Environment

- Python **3.11.1** on macOS-26.6.1-arm64-arm-64bit (arm64)
- librosa: `0.11.0`
- numpy: `2.4.6`
- scipy: `1.17.1`
- soundfile: `0.14.0`
- faiss: `1.15.0`
- sklearn: `1.9.0`
- ffmpeg available: **False**
- Harness source fingerprint: `495c1c6bfa8e0b85ee238927bce1cde0f7cfc556a880dac9107d52f6b4e78701`
- Algorithm source fingerprint: `d53b191a2a98f3ed57ee42dc932c5506732ab8ba20a4ca1f1ea2b6c0db3f98fc`
- Tracks actually indexed: **32**

## Dataset

- Corpus: **archive.org netlabels (CC-BY / CC-BY-SA only)**
- Tracks total: **44** (indexed catalog: **32**, held out: **12**)
- Licenses: {'CC-BY': 18, 'CC-BY-SA': 21, 'CC0-1.0': 5}
- Manifest content hash: `4006aacb0abc1e7f2e12eee8a9f205a6f6b8cc563fef7a9396872cf797767a0e`
- Total reference audio: 139.9 minutes

Audio is **not** committed. Reproduce with `python scripts/fetch_fixture_corpus.py --tracks 50`.

## Methodology

- **1728** positive queries, **126** negative queries (**1854** total).
- Excerpts taken at three positions (beginning / middle / end) and three durations (3 s / 5 s / 10 s). A query is never longer than its source.
- Clean conditions are crossed with all durations x all positions. Noise, codec and filtering are crossed with all durations at the middle position. Speed and pitch are evaluated at 5 s, middle.
- All randomness is seeded from a SHA-256 of the query id, so the query set is byte-reproducible.
- Latency times **only** the `recognize()` call, using `time.perf_counter()`. Index construction is excluded and reported separately.
- Index build (catalog of 32): **6.4 s** total, of which **0.693 s** was FAISS index construction.

## Clean Results

| Condition | Queries | Recall@1 | Recall@3 | No-match | FAR | p50 ms | p95 ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| `clean_duration_3s` | 96 | 25.0% | 45.8% | 0.0% | — | 12.5 | 58.2 |
| `clean_duration_5s` | 96 | 30.2% | 49.0% | 0.0% | — | 16.4 | 55.9 |
| `clean_duration_10s` | 96 | 38.5% | 56.2% | 0.0% | — | 25.6 | 63.7 |

### Positive queries by excerpt position and duration

| Slice | Queries | Recall@1 | Recall@3 |
|---|---:|---:|---:|
| position = beginning | 96 | 40.6% | 64.6% |
| position = end | 96 | 11.5% | 26.0% |
| position = middle | 1536 | 29.9% | 44.8% |
| duration = 10.0 | 448 | 30.6% | 44.9% |
| duration = 3.0 | 448 | 26.8% | 43.8% |
| duration = 5.0 | 832 | 30.3% | 45.4% |

## Degradation Results

### Noise

| Condition | Queries | Recall@1 | Recall@3 | No-match | FAR | p50 ms | p95 ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| `noise_pink_snr0db_duration_3s` | 32 | 6.2% | 15.6% | 0.0% | — | 17.5 | 29.3 |
| `noise_pink_snr0db_duration_5s` | 32 | 6.2% | 15.6% | 0.0% | — | 18.7 | 43.4 |
| `noise_pink_snr0db_duration_10s` | 32 | 6.2% | 15.6% | 0.0% | — | 24.6 | 52.1 |
| `noise_pink_snr10db_duration_3s` | 32 | 9.4% | 18.8% | 0.0% | — | 16.9 | 57.6 |
| `noise_pink_snr10db_duration_5s` | 32 | 9.4% | 18.8% | 0.0% | — | 18.4 | 49.6 |
| `noise_pink_snr10db_duration_10s` | 32 | 9.4% | 18.8% | 0.0% | — | 23.9 | 68.9 |
| `noise_pink_snr20db_duration_3s` | 32 | 21.9% | 37.5% | 0.0% | — | 15.1 | 33.0 |
| `noise_pink_snr20db_duration_5s` | 32 | 21.9% | 43.8% | 0.0% | — | 19.1 | 73.7 |
| `noise_pink_snr20db_duration_10s` | 32 | 21.9% | 43.8% | 0.0% | — | 26.9 | 72.9 |
| `noise_pink_snr5db_duration_3s` | 32 | 6.2% | 18.8% | 0.0% | — | 16.4 | 45.9 |
| `noise_pink_snr5db_duration_5s` | 32 | 6.2% | 18.8% | 0.0% | — | 19.6 | 31.8 |
| `noise_pink_snr5db_duration_10s` | 32 | 9.4% | 18.8% | 0.0% | — | 24.8 | 81.0 |
| `noise_white_snr0db_duration_5s` | 32 | 3.1% | 12.5% | 0.0% | — | 15.9 | 37.4 |
| `noise_white_snr10db_duration_5s` | 32 | 9.4% | 15.6% | 0.0% | — | 16.3 | 54.0 |
| `noise_white_snr20db_duration_5s` | 32 | 12.5% | 34.4% | 0.0% | — | 16.8 | 57.2 |
| `noise_white_snr5db_duration_5s` | 32 | 6.2% | 15.6% | 0.0% | — | 15.3 | 49.7 |

### Codec

| Condition | Queries | Recall@1 | Recall@3 | No-match | FAR | p50 ms | p95 ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| `codec_mp3_128k_duration_3s` | 32 | 40.6% | 59.4% | 0.0% | — | 15.9 | 29.8 |
| `codec_mp3_128k_duration_5s` | 32 | 46.9% | 59.4% | 0.0% | — | 16.0 | 33.7 |
| `codec_mp3_128k_duration_10s` | 32 | 43.8% | 56.2% | 0.0% | — | 22.7 | 50.9 |
| `codec_mp3_32k_duration_3s` | 32 | 37.5% | 59.4% | 0.0% | — | 14.9 | 45.8 |
| `codec_mp3_32k_duration_5s` | 32 | 37.5% | 59.4% | 0.0% | — | 17.2 | 46.3 |
| `codec_mp3_32k_duration_10s` | 32 | 34.4% | 53.1% | 0.0% | — | 23.2 | 43.7 |
| `codec_mp3_64k_duration_3s` | 32 | 40.6% | 59.4% | 0.0% | — | 15.6 | 38.3 |
| `codec_mp3_64k_duration_5s` | 32 | 43.8% | 59.4% | 0.0% | — | 16.2 | 56.5 |
| `codec_mp3_64k_duration_10s` | 32 | 43.8% | 53.1% | 0.0% | — | 26.2 | 45.3 |
| `codec_opus_32k_duration_3s` | 32 | 43.8% | 62.5% | 0.0% | — | 18.1 | 38.0 |
| `codec_opus_32k_duration_5s` | 32 | 43.8% | 59.4% | 0.0% | — | 19.5 | 56.4 |
| `codec_opus_32k_duration_10s` | 32 | 43.8% | 59.4% | 0.0% | — | 28.4 | 81.1 |
| `codec_opus_64k_duration_3s` | 32 | 40.6% | 59.4% | 0.0% | — | 18.7 | 37.8 |
| `codec_opus_64k_duration_5s` | 32 | 46.9% | 59.4% | 0.0% | — | 21.2 | 50.2 |
| `codec_opus_64k_duration_10s` | 32 | 43.8% | 59.4% | 0.0% | — | 30.9 | 108.8 |

### Filter

| Condition | Queries | Recall@1 | Recall@3 | No-match | FAR | p50 ms | p95 ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| `filter_lowpass8k_duration_3s` | 32 | 34.4% | 56.2% | 0.0% | — | 12.5 | 29.3 |
| `filter_lowpass8k_duration_5s` | 32 | 37.5% | 56.2% | 0.0% | — | 16.8 | 38.8 |
| `filter_lowpass8k_duration_10s` | 32 | 37.5% | 53.1% | 0.0% | — | 24.2 | 84.5 |
| `filter_telephone_duration_3s` | 32 | 18.8% | 28.1% | 0.0% | — | 17.3 | 66.1 |
| `filter_telephone_duration_5s` | 32 | 18.8% | 28.1% | 0.0% | — | 17.3 | 45.9 |
| `filter_telephone_duration_10s` | 32 | 18.8% | 28.1% | 0.0% | — | 23.3 | 77.0 |

### Speed

| Condition | Queries | Recall@1 | Recall@3 | No-match | FAR | p50 ms | p95 ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| `speed_+2pct_duration_5s` | 32 | 40.6% | 59.4% | 0.0% | — | 15.9 | 44.5 |
| `speed_+5pct_duration_5s` | 32 | 40.6% | 59.4% | 0.0% | — | 16.5 | 48.7 |
| `speed_-2pct_duration_5s` | 32 | 46.9% | 62.5% | 0.0% | — | 15.0 | 53.4 |
| `speed_-5pct_duration_5s` | 32 | 46.9% | 62.5% | 0.0% | — | 17.3 | 42.3 |

### Pitch

| Condition | Queries | Recall@1 | Recall@3 | No-match | FAR | p50 ms | p95 ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| `pitch_+1st_duration_5s` | 32 | 46.9% | 56.2% | 0.0% | — | 15.6 | 63.6 |
| `pitch_+2st_duration_5s` | 32 | 43.8% | 56.2% | 0.0% | — | 16.5 | 55.3 |
| `pitch_-1st_duration_5s` | 32 | 46.9% | 62.5% | 0.0% | — | 16.1 | 53.9 |
| `pitch_-2st_duration_5s` | 32 | 34.4% | 59.4% | 0.0% | — | 15.5 | 53.2 |

## Negative Results

**False Accept Rate: 100.0%** — 126 of 126 negative queries returned a catalog track.
**Correct rejection rate: 0.0%**

| Condition | Queries | Recall@1 | Recall@3 | No-match | FAR | p50 ms | p95 ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| `negative_near_silence_duration_3s` | 1 | — | — | — | 100.0% | 13.4 | 13.4 |
| `negative_near_silence_duration_5s` | 1 | — | — | — | 100.0% | 24.5 | 24.5 |
| `negative_near_silence_duration_10s` | 1 | — | — | — | 100.0% | 21.5 | 21.5 |
| `negative_noise_pink_duration_3s` | 1 | — | — | — | 100.0% | 14.7 | 14.7 |
| `negative_noise_pink_duration_5s` | 1 | — | — | — | 100.0% | 16.6 | 16.6 |
| `negative_noise_pink_duration_10s` | 1 | — | — | — | 100.0% | 25.2 | 25.2 |
| `negative_noise_white_duration_3s` | 1 | — | — | — | 100.0% | 14.9 | 14.9 |
| `negative_noise_white_duration_5s` | 1 | — | — | — | 100.0% | 17.3 | 17.3 |
| `negative_noise_white_duration_10s` | 1 | — | — | — | 100.0% | 27.0 | 27.0 |
| `negative_out_of_catalog_music_duration_3s` | 36 | — | — | — | 100.0% | 18.5 | 41.7 |
| `negative_out_of_catalog_music_duration_5s` | 36 | — | — | — | 100.0% | 23.6 | 66.1 |
| `negative_out_of_catalog_music_duration_10s` | 36 | — | — | — | 100.0% | 32.5 | 68.8 |
| `negative_silence_duration_3s` | 1 | — | — | — | 100.0% | 13.4 | 13.4 |
| `negative_silence_duration_5s` | 1 | — | — | — | 100.0% | 19.5 | 19.5 |
| `negative_silence_duration_10s` | 1 | — | — | — | 100.0% | 24.9 | 24.9 |
| `negative_speech_duration_3s` | 6 | — | — | — | 100.0% | 16.5 | 20.3 |

### Would a score threshold have helped?

The prototype has no rejection stage, so its FAR is 1.0 by construction. This sweep asks a different question: if a distance threshold *were* added, what could it achieve?

| Max FAR allowed | Best Recall@1 | at distance |
|---|---:|---:|
| ≤ 0.001 | 10.4% | 1794.779 |
| ≤ 0.01 | 11.7% | 2081.601 |
| ≤ 0.05 | 13.8% | 2480.300 |
| ≤ 0.10 | 15.7% | 2912.853 |

L2 distance distributions (overlap = inseparability):

| Set | n | min | p05 | median | p95 | max |
|---|---:|---:|---:|---:|---:|---:|
| Correct matches | 509 | 29.351 | 469.2157 | 2616.7007 | 13745.7111 | 36904.8398 |
| Negatives (no true match) | 126 | 1824.0072 | 2523.5459 | 7252.6294 | 95827.3223 | 427287.625 |

## Worst Conditions (measured)

| Condition | Queries | Recall@1 | Recall@3 | No-match | FAR | p50 ms | p95 ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| `noise_white_snr0db_duration_5s` | 32 | 3.1% | 12.5% | 0.0% | — | 15.9 | 37.4 |
| `noise_pink_snr0db_duration_3s` | 32 | 6.2% | 15.6% | 0.0% | — | 17.5 | 29.3 |
| `noise_pink_snr0db_duration_5s` | 32 | 6.2% | 15.6% | 0.0% | — | 18.7 | 43.4 |
| `noise_pink_snr0db_duration_10s` | 32 | 6.2% | 15.6% | 0.0% | — | 24.6 | 52.1 |
| `noise_white_snr5db_duration_5s` | 32 | 6.2% | 15.6% | 0.0% | — | 15.3 | 49.7 |
| `noise_pink_snr5db_duration_3s` | 32 | 6.2% | 18.8% | 0.0% | — | 16.4 | 45.9 |
| `noise_pink_snr5db_duration_5s` | 32 | 6.2% | 18.8% | 0.0% | — | 19.6 | 31.8 |
| `noise_white_snr10db_duration_5s` | 32 | 9.4% | 15.6% | 0.0% | — | 16.3 | 54.0 |

## Best Conditions (measured)

| Condition | Queries | Recall@1 | Recall@3 | No-match | FAR | p50 ms | p95 ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| `speed_-5pct_duration_5s` | 32 | 46.9% | 62.5% | 0.0% | — | 17.3 | 42.3 |
| `speed_-2pct_duration_5s` | 32 | 46.9% | 62.5% | 0.0% | — | 15.0 | 53.4 |
| `pitch_-1st_duration_5s` | 32 | 46.9% | 62.5% | 0.0% | — | 16.1 | 53.9 |
| `codec_opus_64k_duration_5s` | 32 | 46.9% | 59.4% | 0.0% | — | 21.2 | 50.2 |
| `codec_mp3_128k_duration_5s` | 32 | 46.9% | 59.4% | 0.0% | — | 16.0 | 33.7 |
| `pitch_+1st_duration_5s` | 32 | 46.9% | 56.2% | 0.0% | — | 15.6 | 63.6 |
| `codec_opus_32k_duration_3s` | 32 | 43.8% | 62.5% | 0.0% | — | 18.1 | 38.0 |
| `codec_opus_64k_duration_10s` | 32 | 43.8% | 59.4% | 0.0% | — | 30.9 | 108.8 |

## Current Recognizer Assessment

On **clean 5-second excerpts** — the easiest condition in the whole matrix — the prototype reaches **Recall@1 = 30.2%** against a catalog of only **32** tracks.

Its **False Accept Rate is 100.0%**: it returns catalog tracks for speech, silence and pure noise, because `src/music_recognition.py` has no rejection stage at all — it returns `k=3` unconditionally. That alone makes it unusable as a product.

The threshold sweep shows this is **not merely a missing threshold**. Even with an oracle distance cut-off tuned on the test data itself, the best achievable Recall@1 at FAR ≤ 1% is **11.7%**. The 26-dimensional MFCC mean/std representation cannot separate a matching recording from a non-matching one.

**The position breakdown exposes what the recognizer is actually doing.** Recall@1 is 40.6% for excerpts from the *beginning* of a track, 29.9% from the *middle*, and 11.5% from the *end*. The reference side indexes only the first 30 seconds of each track (`librosa.load(..., duration=30)` in `src/audio_processing.py`), and the median corpus track is 152 s long — so middle and end excerpts share **no audio content at all** with what was indexed. Any correct answer there cannot come from recognising the content; it comes from the two excerpts happening to have a similar overall spectral character. That is the empirical confirmation that 26-d MFCC mean/std is a **timbre-similarity descriptor, not an identity fingerprint**.

**Verdict: the current MFCC/FAISS approach is not a viable foundation for Phase 1.** It should be replaced by spectral-peak landmark hashing with time-offset consistency scoring, not tuned. The measured numbers above are the baseline every future engine must beat.

## Limitations

- Catalog is tiny (32 tracks). Recognition difficulty grows sharply with catalog size, so these numbers are an OPTIMISTIC upper bound; a 10,000-track catalog would be materially harder.
- Corpus is CC-BY/CC-BY-SA netlabel electronic-leaning music from archive.org, not a genre-balanced sample of mainstream commercial music.
- Degradations are synthetic. Real phone captures add room acoustics, handset response, AGC and codec chains simultaneously; no real-world recordings are included in this run.
- ffmpeg is not installed, so codec tests use libsndfile's MP3 and Opus encoders. Bitrates are VBR/CBR-quality targeted and the achieved bitrate is measured per file rather than exactly dialled in.
- Speed and pitch conditions are evaluated only at 5 s / middle position to keep the matrix affordable.
- Latency is measured on one machine with a warm filesystem cache and excludes index construction; it is not a server-side SLA figure.
- The threshold sweep is fitted on the evaluation set itself, so it is an OPTIMISTIC upper bound on what any real threshold could achieve.
- Negative speech is synthetic (macOS `say`), not natural human speech.
- Family-level aggregates are NOT directly comparable to each other: `clean` spans all three positions while codec/speed/pitch are middle-position only, and position strongly affects the score. Compare conditions at the condition level, not the family level.
- The reference side indexes only the first 30 s of each track, an existing property of the prototype that was preserved. This is a limitation OF THE RECOGNIZER being measured, not of the benchmark.
