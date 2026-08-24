# Acoustic fingerprint format (Phase 1A)

Reference for the landmark fingerprint produced by
[`musicintel/recognition/fingerprint.py`](../musicintel/recognition/fingerprint.py).

Phase 1A is **extraction only**. There is no index, no matcher and no
accept/reject decision yet, so this document makes no accuracy claim. The
Phase 0 baseline (Recall@1 29.46%, FAR 100%) stands unchanged in
[`eval/reports/baseline.md`](../eval/reports/baseline.md) until a matcher exists
to measure against it.

## Pipeline

```
audio file
  -> decode: mono, 11025 Hz, float32          (load_audio)
  -> STFT: n_fft 1024, hop 128, Hann, centered
  -> log-power spectrogram, dB, absolute ref  (spectrogram)
  -> local spectral peaks, density-controlled (find_peaks)
  -> anchor/target pairing in a bounded window(pair_landmarks)
  -> packed uint32 hash + int32 anchor frame  (FingerprintResult)
```

## Audio representation

| Parameter | Value | Why |
|---|---|---|
| Sample rate | 11025 Hz | Nyquist 5512 Hz. Every landmark lives well below it, and a quarter of the samples is a quarter of the STFT cost. |
| Channels | mono | Stereo differences are a property of the delivery chain, not of the recording's identity. |
| `n_fft` | 1024 | 10.77 Hz per bin, 92.9 ms window — fine enough to separate partials, short enough that a peak is a local event. |
| `hop_length` | 128 | 11.61 ms per frame, **86.13 frames/s**. The matcher will vote on time offsets, and offset precision is bounded by the frame rate, so the hop is deliberately small. |
| Scale | dB, `ref=1.0` | An absolute dB scale turns a gain change into a constant offset on every value, which the relative amplitude gate then cancels exactly. |

Decode goes through one function for both reference and query. A mismatch there
shifts every frequency bin and silently destroys landmark agreement.

## Peak detection

A peak is a point that dominates its local neighbourhood in both time and
frequency. Three filters run in order:

1. **Local dominance** — the point must equal the maximum of a
   9 bin × 9 frame neighbourhood (~97 Hz × ~104 ms), via
   `scipy.ndimage.maximum_filter`. This is inherently level-invariant: scaling
   the signal scales the neighbourhood with it, so the same points win.
2. **Amplitude gate** — strictly above the 75th percentile of the in-band
   spectrogram. A percentile rather than an absolute dB threshold, so volume
   and microphone gain cancel. The strictness matters: on digital silence every
   value equals the percentile, so `>` admits nothing and silence yields **zero**
   peaks instead of a peak at every bin.
3. **Density caps** — at most 5 peaks per frame, and at most
   `target_peak_density` (default 30) peaks per one-second bucket, strongest
   first. Bucketing per second rather than globally keeps quiet passages
   represented instead of letting loud sections consume the whole budget.

**Band: 200–3000 Hz** (bins 18–278). Below ~200 Hz is bass and room rumble,
which handset response mangles. Above ~3000 Hz is the first thing low-bitrate
codecs and telephone-band filtering discard. These bounds come from the
properties of the signal chain, **not** from tuning against the 32-track Phase 0
catalog — numbers fitted to that catalog would not survive a real one.

Ties are broken by (magnitude, frame, bin) so the ordering is total, which is
what makes selection reproducible rather than dependent on sort implementation.

## Landmark pairing

Each anchor peak is paired with up to `fan_out` (default 5) later peaks whose
frame distance lies in `[min_delta_frames, max_delta_frames]` = `[1, 128]`,
i.e. 11.6 ms to **1.486 s**. `dt = 0` is excluded because it carries no temporal
information.

Bounding the window is what makes a landmark *local*: it describes a
relationship inside about a second and a half of audio, so a query only has to
overlap that much of the reference to regenerate the same key.

## Hash encoding

Each landmark packs into one unsigned 32-bit integer:

```
 bit  31 ....... 28 27 ........... 18 17 ............ 8 7 ......... 0
      (unused, 0)   anchor freq bin    target freq bin   delta frames
                    10 bits (0-1023)   10 bits (0-1023)  8 bits (0-255)
```

```python
key = (f_anchor << 18) | (f_target << 8) | dt
```

Only integer arithmetic on integer bin indices — **no floating point enters the
key**, so it is exactly reproducible across platforms. `pack_hash` /
`unpack_hash` are inverses, and out-of-range components raise rather than
silently wrapping. `FingerprintConfig.validate()` rejects any configuration the
layout cannot represent (`n_fft` above 2046, `max_delta_frames` above 255).

## Output representation

`FingerprintResult` holds two parallel arrays rather than a list of objects —
**8 bytes per landmark** instead of roughly 60:

| Field | dtype | Meaning |
|---|---|---|
| `hashes` | `uint32` | packed landmark key |
| `anchor_frames` | `int32` | frame index of the anchor peak |

Rows are sorted by `(anchor_frame, hash)`, so two runs over the same audio
produce byte-identical arrays.

`to_index_rows(track_id)` yields `(hash, track_id, anchor_frame)` — directly
persistable as the `hash -> (track_id, anchor_time)` multimap Phase 1B needs.
Anchor time is carried as an integer **frame**, not seconds: the matcher will
histogram differences of these values, and integers bin exactly where floats
would need a tolerance.

FAISS is deliberately not used. FAISS answers "which vector is nearest", a
similarity question. Landmark matching is exact equality over integer keys
followed by a time-offset vote, so the right structure is a hash multimap.

## Expected density

Measured on five corpus tracks (192–299 s, ~21 minutes of real audio):

| Quantity | Observed |
|---|---|
| Peaks per second | 27.6 – 30.0 (target 30) |
| Fingerprints per second | 138 – 150 |
| Storage | 1.15 KB per audio-second ≈ 4.2 MB per audio-hour |
| Extraction throughput | ~480× realtime, excluding decode |

Fingerprint density is approximately `peak_density × fan_out`; anchors near the
end of a track have fewer eligible targets, so it lands slightly below.

## Why this preserves temporal identity

The Phase 0 representation reduced a whole track to a 26-dimensional MFCC
mean/std vector — one point per track, time averaged away. Its own report
records the consequence: Recall@1 was 40.6% for excerpts from the beginning of a
track but 11.5% for excerpts from the end, because the reference side only
indexed the first 30 seconds and the descriptor had no way to represent *where*
in the track anything happened. Any correct answer from the middle or end came
from overall spectral character, not from recognising content.

A landmark asserts something different: *a peak at frequency A was followed,
`dt` frames later, by a peak at frequency B.* That is a statement about the
internal temporal structure of one specific recording. Three properties follow:

- **Locality** — each key depends on ~1.5 s of audio, so a short query
  regenerates the same keys as the full reference over the region they share.
- **Sparsity** — ~30 events per second survive; the rest of the spectrogram,
  including most of what noise and codecs alter, is discarded before hashing.
- **Alignment** — anchor frames are retained, so matching keys can be checked
  for a *consistent* time offset. Coincidental key collisions scatter across
  offsets; a true match piles them onto one. That check is what a mean/std
  vector can never provide, and it is the basis of the rejection stage the
  Phase 0 recognizer entirely lacked (FAR 100%).

None of this is demonstrated yet. It is the reason the representation was
chosen; Phase 1B's matcher and a benchmark run are what would establish it.

## Known limitations

- **No matcher yet.** Extraction alone proves nothing about recognition accuracy.
- **Pitch and speed shifts move every bin**, so the exact-integer key breaks
  under them. Phase 0 measured those as separate conditions; handling them needs
  either multi-rate indexing or a shift-tolerant key, and neither is in Phase 1A.
- **Memory scales with track length** — the full spectrogram is held in RAM
  (~172 MB peak for a 299 s track). Chunked extraction is the fix when it matters.
- **Defaults are reasoned, not optimised.** They were chosen from signal-chain
  properties and deliberately not swept against the evaluation corpus.
