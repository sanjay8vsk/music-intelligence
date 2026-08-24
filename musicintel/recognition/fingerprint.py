"""Sparse spectral-landmark acoustic fingerprinting.

WHY THIS EXISTS
---------------
The Phase 0 baseline collapsed each track to a 26-d MFCC mean/std vector. That
representation measures average timbre, so it cannot tell "this recording" from
"a recording that sounds broadly similar", and it discards time entirely. The
measured consequence is in eval/reports/baseline.md: Recall@1 29.46% against a
32-track catalog, and a false accept rate of 1.0.

This module takes the opposite approach. It keeps a sparse set of LOCAL events
-- individual spectral peaks -- and encodes the TIME RELATIONSHIP between pairs
of them. A landmark says "a peak at frequency A was followed, dt frames later,
by a peak at frequency B". That is a statement about the internal temporal
structure of one specific recording, not about its average colour, which is why
this class of representation can support exact identification.

WHAT IT DOES NOT DO
-------------------
Phase 1A is extraction only. There is no index, no matcher, no scoring and no
accept/reject decision here, so this module makes no accuracy claim of any kind.

PIPELINE
--------
    audio -> mono 11025 Hz -> STFT -> log-power spectrogram
          -> local spectral peaks (density-controlled)
          -> anchor/target pairing inside a bounded time window
          -> packed 32-bit integer hash + anchor frame

The output is designed to be persisted directly as `hash -> (track_id,
anchor_time)`; see `FingerprintResult.to_index_rows`.

FAISS is deliberately not involved. FAISS answers "which vector is nearest",
which is a similarity question. Landmark matching is an exact-equality question
over integer keys followed by a time-offset vote, so the right structure is a
hash multimap, not a vector index.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Version of the FINGERPRINT FORMAT -- the hash layout plus the pipeline that
# fills it. Bump it whenever a change would make previously stored fingerprints
# uncomparable with freshly extracted ones (different bit layout, different
# peak-selection rule, different default band). It is NOT a version of this
# file: refactoring that leaves the produced keys identical must not bump it.
# A persisted index records this value and refuses to load against a mismatch.
FORMAT_VERSION = 1

# -- hash packing layout ------------------------------------------------------
# A landmark is packed into one unsigned 32-bit integer:
#
#     bits 27..18 : anchor frequency bin   (10 bits, 0..1023)
#     bits 17..8  : target frequency bin   (10 bits, 0..1023)
#     bits  7..0  : delta frames           ( 8 bits, 0..255)
#
# 10 bits of frequency covers every bin of an n_fft up to 2046, and 8 bits of
# delta covers ~3.0 s at the default hop. The layout is fixed and documented so
# a stored fingerprint stays readable independently of this source file; see
# docs/fingerprint-format.md.
FREQ_BITS = 10
DELTA_BITS = 8
MAX_FREQ_BIN = (1 << FREQ_BITS) - 1
MAX_DELTA_FRAMES = (1 << DELTA_BITS) - 1
HASH_DTYPE = np.uint32
FRAME_DTYPE = np.int32


@dataclass(frozen=True)
class FingerprintConfig:
    """Every tunable in one place, with the reasoning for each default.

    These defaults were chosen from properties of the signal chain -- codec
    bandwidth, handset response, STFT resolution -- and NOT fitted against the
    32-track Phase 0 catalog. Tuning them on that catalog would produce numbers
    that do not survive contact with a real one.
    """

    # -- decode ---------------------------------------------------------
    # 11025 Hz puts Nyquist at 5512 Hz. Everything the landmarks use lives well
    # below that, and a quarter of the samples is a quarter of the STFT cost.
    sample_rate: int = 11025

    # -- STFT -----------------------------------------------------------
    # 1024 @ 11025 Hz -> 10.77 Hz per bin, 92.9 ms window.
    n_fft: int = 1024
    # 128 -> 11.61 ms per frame, 86.1 frames/s. Fine time resolution matters:
    # the eventual matcher votes on time offsets, and offset precision is
    # bounded by the frame rate.
    hop_length: int = 128

    # -- peak detection --------------------------------------------------
    # Below ~200 Hz is bass and room rumble, which handset response mangles.
    # Above ~3000 Hz is the first thing low-bitrate codecs and telephone-band
    # filtering throw away. The band between survives the degradations that
    # matter without being dominated by either.
    freq_min_hz: float = 200.0
    freq_max_hz: float = 3000.0
    # A peak must dominate its neighbourhood in both axes. Sizes are full
    # widths in bins/frames; 9 bins ~ 97 Hz, 9 frames ~ 104 ms.
    peak_neighborhood_freq_bins: int = 9
    peak_neighborhood_time_frames: int = 9
    # Amplitude gate as a PERCENTILE of the in-band spectrogram, not an
    # absolute dB value. A gain change shifts every dB value equally, so it
    # shifts the percentile equally too -- the gate is therefore invariant to
    # volume and microphone gain, which an absolute threshold would not be.
    threshold_percentile: float = 75.0
    # Caps that bound density from above. Without them a loud transient can
    # monopolise a whole second and a noise burst can explode the peak count.
    max_peaks_per_frame: int = 5
    target_peak_density: float = 30.0  # peaks per second, per one-second bucket

    # -- landmark pairing -------------------------------------------------
    fan_out: int = 5  # target peaks paired per anchor
    min_delta_frames: int = 1  # dt == 0 carries no temporal information
    max_delta_frames: int = 128  # ~1.49 s at the default hop

    # -- derived ----------------------------------------------------------
    @property
    def frame_rate(self) -> float:
        """Frames per second."""
        return self.sample_rate / self.hop_length

    @property
    def bin_hz(self) -> float:
        """Hertz per frequency bin."""
        return self.sample_rate / self.n_fft

    @property
    def n_bins(self) -> int:
        return self.n_fft // 2 + 1

    def frames_to_seconds(self, frames) -> np.ndarray | float:
        return np.asarray(frames) * self.hop_length / self.sample_rate

    def validate(self) -> None:
        """Fail loudly on a configuration the hash layout cannot represent."""
        if self.n_bins - 1 > MAX_FREQ_BIN:
            raise ValueError(
                f"n_fft={self.n_fft} yields {self.n_bins} bins, which does not "
                f"fit in {FREQ_BITS} bits (max bin {MAX_FREQ_BIN})"
            )
        if self.max_delta_frames > MAX_DELTA_FRAMES:
            raise ValueError(
                f"max_delta_frames={self.max_delta_frames} does not fit in "
                f"{DELTA_BITS} bits (max {MAX_DELTA_FRAMES})"
            )
        if self.min_delta_frames < 1:
            raise ValueError("min_delta_frames must be >= 1; dt=0 has no time content")
        if self.min_delta_frames > self.max_delta_frames:
            raise ValueError("min_delta_frames exceeds max_delta_frames")
        if not 0.0 <= self.threshold_percentile <= 100.0:
            raise ValueError("threshold_percentile must be a percentile in [0, 100]")
        if self.freq_min_hz >= self.freq_max_hz:
            raise ValueError("freq_min_hz must be below freq_max_hz")
        if self.fan_out < 1:
            raise ValueError("fan_out must be >= 1")
        if self.max_peaks_per_frame < 1:
            raise ValueError("max_peaks_per_frame must be >= 1")
        if self.target_peak_density <= 0:
            raise ValueError("target_peak_density must be positive")


DEFAULT_CONFIG = FingerprintConfig()


# ------------------------------------------------------------------ hashing --
def pack_hash(freq_anchor: int, freq_target: int, delta_frames: int) -> int:
    """Pack (f_anchor, f_target, dt) into one unsigned 32-bit key.

    Pure integer arithmetic on integer inputs, so it is exactly reproducible on
    any platform -- no floating point enters the key.
    """
    if not 0 <= freq_anchor <= MAX_FREQ_BIN:
        raise ValueError(f"freq_anchor {freq_anchor} out of range")
    if not 0 <= freq_target <= MAX_FREQ_BIN:
        raise ValueError(f"freq_target {freq_target} out of range")
    if not 0 <= delta_frames <= MAX_DELTA_FRAMES:
        raise ValueError(f"delta_frames {delta_frames} out of range")
    return (
        (freq_anchor << (FREQ_BITS + DELTA_BITS))
        | (freq_target << DELTA_BITS)
        | delta_frames
    )


def unpack_hash(key: int) -> tuple[int, int, int]:
    """Inverse of `pack_hash`. Returns (f_anchor, f_target, dt)."""
    key = int(key)
    delta = key & MAX_DELTA_FRAMES
    target = (key >> DELTA_BITS) & MAX_FREQ_BIN
    anchor = (key >> (FREQ_BITS + DELTA_BITS)) & MAX_FREQ_BIN
    return anchor, target, delta


# ------------------------------------------------------------------ results --
@dataclass(frozen=True)
class Landmark:
    """One fingerprint, in readable form.

    The compact arrays on `FingerprintResult` are the storage representation;
    this is the per-item view used by tests and by anyone reading output.
    """

    hash: int
    anchor_frame: int
    anchor_time: float

    @property
    def freq_anchor_bin(self) -> int:
        return unpack_hash(self.hash)[0]

    @property
    def freq_target_bin(self) -> int:
        return unpack_hash(self.hash)[1]

    @property
    def delta_frames(self) -> int:
        return unpack_hash(self.hash)[2]


@dataclass(frozen=True, eq=False)
class FingerprintResult:
    """Fingerprints for one piece of audio.

    Stored as two parallel numpy arrays rather than a list of objects: 8 bytes
    per landmark instead of ~60, which is the difference between a catalog that
    fits in memory and one that does not.

    Rows are ordered by (anchor_frame, hash) so the output is canonical -- two
    runs over the same audio produce byte-identical arrays.
    """

    hashes: np.ndarray  # uint32, shape (n,)
    anchor_frames: np.ndarray  # int32,  shape (n,)
    config: FingerprintConfig
    duration_sec: float
    peak_count: int

    def __len__(self) -> int:
        return int(self.hashes.size)

    @property
    def anchor_times(self) -> np.ndarray:
        """Anchor times in seconds."""
        return self.anchor_frames * (self.config.hop_length / self.config.sample_rate)

    @property
    def density(self) -> float:
        """Fingerprints per second of audio."""
        return len(self) / self.duration_sec if self.duration_sec > 0 else 0.0

    @property
    def peak_density(self) -> float:
        """Retained spectral peaks per second of audio."""
        return self.peak_count / self.duration_sec if self.duration_sec > 0 else 0.0

    @property
    def nbytes(self) -> int:
        """Bytes held by the fingerprint arrays themselves."""
        return int(self.hashes.nbytes + self.anchor_frames.nbytes)

    def landmarks(self) -> list[Landmark]:
        """Per-item view. Materializes objects; prefer the arrays at scale."""
        scale = self.config.hop_length / self.config.sample_rate
        return [
            Landmark(hash=int(h), anchor_frame=int(f), anchor_time=int(f) * scale)
            for h, f in zip(self.hashes, self.anchor_frames)
        ]

    def to_index_rows(self, track_id: str) -> list[tuple[int, str, int]]:
        """Rows for a future `hash -> (track_id, anchor_frame)` index.

        Anchor time is carried as an integer FRAME, not seconds: the matcher
        histograms differences of these values, and integers bin exactly where
        floats would need a tolerance.
        """
        return [
            (int(h), track_id, int(f))
            for h, f in zip(self.hashes, self.anchor_frames)
        ]


def _empty_result(config: FingerprintConfig, duration_sec: float) -> FingerprintResult:
    return FingerprintResult(
        hashes=np.empty(0, dtype=HASH_DTYPE),
        anchor_frames=np.empty(0, dtype=FRAME_DTYPE),
        config=config,
        duration_sec=duration_sec,
        peak_count=0,
    )


# ------------------------------------------------------------------- stages --
def load_audio(
    path: str | Path, config: FingerprintConfig | None = None
) -> tuple[np.ndarray, int]:
    """Decode a file to mono float32 at the configured rate.

    Every query and every reference goes through this one function, so the
    decode is identical on both sides. A mismatch there shifts the spectrum and
    silently destroys landmark agreement.
    """
    import librosa

    cfg = config or DEFAULT_CONFIG
    y, sr = librosa.load(path, sr=cfg.sample_rate, mono=True)
    return np.asarray(y, dtype=np.float32), int(sr)


def spectrogram(y: np.ndarray, config: FingerprintConfig | None = None) -> np.ndarray:
    """Log-power spectrogram in dB, shape (n_bins, n_frames).

    `ref=1.0` keeps the dB scale absolute. That matters for the percentile gate
    downstream: with an absolute scale a gain change is a constant offset on
    every value, which the percentile then cancels exactly.
    """
    import librosa

    cfg = config or DEFAULT_CONFIG
    y = np.asarray(y, dtype=np.float32)
    if y.size == 0:
        return np.empty((cfg.n_bins, 0), dtype=np.float32)
    power = (
        np.abs(
            librosa.stft(
                y,
                n_fft=cfg.n_fft,
                hop_length=cfg.hop_length,
                window="hann",
                center=True,
            )
        )
        ** 2
    )
    return librosa.power_to_db(power, ref=1.0, amin=1e-10, top_db=None)


def find_peaks(
    spec_db: np.ndarray, config: FingerprintConfig | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Sparse local spectral peaks, density-controlled.

    Returns (freq_bins, frames), both int arrays sorted by (frame, freq_bin).

    Three filters run in order:

      1. LOCAL DOMINANCE -- a point must equal the maximum of its
         neighbourhood. This is inherently level-invariant: scaling the signal
         scales the neighbourhood too, so the same points win.
      2. AMPLITUDE GATE -- strictly above a percentile of the in-band values.
         The strictness is load-bearing: on digital silence every value is
         equal to the percentile, so `>` admits nothing and silence yields zero
         peaks rather than a peak at every bin.
      3. DENSITY CAPS -- at most `max_peaks_per_frame` per frame and at most
         `target_peak_density` per one-second bucket, strongest first. Bucketing
         by second rather than globally keeps quiet passages represented instead
         of letting loud sections consume the whole budget.
    """
    from scipy.ndimage import maximum_filter

    cfg = config or DEFAULT_CONFIG
    cfg.validate()

    empty = (np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32))
    if spec_db.size == 0 or spec_db.shape[1] == 0:
        return empty

    min_bin = int(np.ceil(cfg.freq_min_hz / cfg.bin_hz))
    max_bin = int(np.floor(cfg.freq_max_hz / cfg.bin_hz))
    min_bin = max(min_bin, 0)
    max_bin = min(max_bin, spec_db.shape[0] - 1)
    if max_bin < min_bin:
        return empty

    band = spec_db[min_bin : max_bin + 1, :]

    footprint = (
        max(1, cfg.peak_neighborhood_freq_bins),
        max(1, cfg.peak_neighborhood_time_frames),
    )
    local_max = maximum_filter(band, size=footprint, mode="constant", cval=-np.inf)
    gate = float(np.percentile(band, cfg.threshold_percentile))
    mask = (band >= local_max) & (band > gate)

    f_local, frames = np.nonzero(mask)
    if f_local.size == 0:
        return empty
    mags = band[f_local, frames]
    freq_bins = (f_local + min_bin).astype(np.int32)
    frames = frames.astype(np.int32)

    # Strongest first; ties broken by frame then bin so the order is total and
    # therefore reproducible, which a bare magnitude sort would not be.
    order = np.lexsort((freq_bins, frames, -mags))

    n_frames = band.shape[1]
    frames_per_bucket = max(1, int(round(cfg.frame_rate)))
    bucket_cap = max(
        1,
        int(round(cfg.target_peak_density * frames_per_bucket / cfg.frame_rate)),
    )
    n_buckets = int(n_frames // frames_per_bucket) + 1

    per_frame = np.zeros(n_frames, dtype=np.int32)
    per_bucket = np.zeros(n_buckets, dtype=np.int32)
    keep = np.zeros(f_local.size, dtype=bool)
    for i in order:
        t = int(frames[i])
        b = t // frames_per_bucket
        if per_frame[t] >= cfg.max_peaks_per_frame:
            continue
        if per_bucket[b] >= bucket_cap:
            continue
        per_frame[t] += 1
        per_bucket[b] += 1
        keep[i] = True

    freq_bins = freq_bins[keep]
    frames = frames[keep]
    final = np.lexsort((freq_bins, frames))  # canonical: by frame, then bin
    return freq_bins[final], frames[final]


def pair_landmarks(
    freq_bins: np.ndarray,
    frames: np.ndarray,
    config: FingerprintConfig | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Pair each anchor with up to `fan_out` later peaks; return (hashes, anchors).

    The target zone is bounded in time by [min_delta_frames, max_delta_frames].
    Bounding it is what makes the fingerprint local: a landmark describes a
    relationship inside roughly a second and a half of audio, so a query only
    has to overlap that much of the reference to reproduce it.
    """
    cfg = config or DEFAULT_CONFIG
    cfg.validate()

    n = int(freq_bins.size)
    if n < 2:
        return np.empty(0, dtype=HASH_DTYPE), np.empty(0, dtype=FRAME_DTYPE)

    shift = FREQ_BITS + DELTA_BITS
    hashes: list[int] = []
    anchors: list[int] = []

    for i in range(n):
        t1 = int(frames[i])
        f1 = int(freq_bins[i])
        if f1 > MAX_FREQ_BIN:
            continue
        paired = 0
        for j in range(i + 1, n):
            dt = int(frames[j]) - t1
            if dt < cfg.min_delta_frames:
                continue
            if dt > cfg.max_delta_frames:
                break  # peaks are time-sorted, so nothing later can qualify
            f2 = int(freq_bins[j])
            if f2 > MAX_FREQ_BIN:
                continue
            hashes.append((f1 << shift) | (f2 << DELTA_BITS) | dt)
            anchors.append(t1)
            paired += 1
            if paired >= cfg.fan_out:
                break

    if not hashes:
        return np.empty(0, dtype=HASH_DTYPE), np.empty(0, dtype=FRAME_DTYPE)

    h = np.asarray(hashes, dtype=HASH_DTYPE)
    a = np.asarray(anchors, dtype=FRAME_DTYPE)
    order = np.lexsort((h, a))  # canonical: by anchor frame, then hash
    return h[order], a[order]


# --------------------------------------------------------------- public API --
def fingerprint(
    y: np.ndarray,
    sr: int,
    config: FingerprintConfig | None = None,
) -> FingerprintResult:
    """Fingerprint already-decoded audio.

    Audio at a different rate is resampled to `config.sample_rate` first --
    silently accepting a mismatched rate would shift every frequency bin and
    produce fingerprints that cannot match anything.
    """
    cfg = config or DEFAULT_CONFIG
    cfg.validate()

    y = np.asarray(y, dtype=np.float32)
    if y.ndim > 1:  # mixdown to mono
        y = y.mean(axis=tuple(range(y.ndim - 1)))
    if sr != cfg.sample_rate and y.size:
        import librosa

        y = librosa.resample(y, orig_sr=sr, target_sr=cfg.sample_rate)
        sr = cfg.sample_rate

    duration = float(y.size) / cfg.sample_rate
    if y.size == 0:
        return _empty_result(cfg, 0.0)

    spec = spectrogram(y, cfg)
    freq_bins, frames = find_peaks(spec, cfg)
    hashes, anchors = pair_landmarks(freq_bins, frames, cfg)
    return FingerprintResult(
        hashes=hashes,
        anchor_frames=anchors,
        config=cfg,
        duration_sec=duration,
        peak_count=int(freq_bins.size),
    )


def fingerprint_file(
    path: str | Path, config: FingerprintConfig | None = None
) -> FingerprintResult:
    """Decode a file and fingerprint it."""
    cfg = config or DEFAULT_CONFIG
    y, sr = load_audio(path, cfg)
    return fingerprint(y, sr, cfg)


__all__ = [
    "DEFAULT_CONFIG",
    "DELTA_BITS",
    "FORMAT_VERSION",
    "FREQ_BITS",
    "FingerprintConfig",
    "FingerprintResult",
    "Landmark",
    "MAX_DELTA_FRAMES",
    "MAX_FREQ_BIN",
    "fingerprint",
    "fingerprint_file",
    "find_peaks",
    "load_audio",
    "pack_hash",
    "pair_landmarks",
    "spectrogram",
    "unpack_hash",
]
