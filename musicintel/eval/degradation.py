"""Deterministic query generation and the audio degradation matrix.

Two responsibilities:

  1. Excerpting -- cut a query of a given duration from a given position of a
     reference track. Never produces a query longer than its source.
  2. Degrading -- apply one reproducible transform (noise, codec, filtering,
     speed, pitch) to that excerpt.

Everything is deterministic. Randomness is drawn from a seed derived from the
query id, so the same manifest always yields byte-identical queries.

Reference audio is never modified in place; all transforms return new arrays.
"""

from __future__ import annotations

import hashlib
import io
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal as sps

QUERY_SAMPLE_RATE = 22050
POSITIONS = ("beginning", "middle", "end")
DURATIONS = (3.0, 5.0, 10.0)


# ---------------------------------------------------------------- specs ----
@dataclass(frozen=True)
class QuerySpec:
    """A fully-specified, reproducible query."""

    query_id: str
    track_id: str | None  # None for synthetic negatives
    duration: float
    position: str
    condition: str  # canonical label, e.g. "noise_pink_snr10db"
    family: str  # clean | noise | codec | filter | speed | pitch | negative
    params: dict = field(default_factory=dict)
    seed: int = 0
    source_hash: str = ""
    is_negative: bool = False
    # Filled in after rendering:
    rendered_path: str | None = None
    measured: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def derive_seed(query_id: str) -> int:
    """Stable 32-bit seed from a query id (independent of PYTHONHASHSEED)."""
    return int.from_bytes(hashlib.sha256(query_id.encode()).digest()[:4], "big")


def make_query_id(track_id: str, duration: float, position: str, condition: str) -> str:
    return f"{track_id}__{condition}__d{duration:g}s__{position}"


# ------------------------------------------------------------ excerpting ----
def excerpt_offset(total_sec: float, duration: float, position: str) -> float:
    """Start offset for an excerpt, clamped so it never runs past the source."""
    if duration >= total_sec:
        return 0.0
    if position == "beginning":
        return 0.0
    if position == "middle":
        return (total_sec - duration) / 2.0
    if position == "end":
        return total_sec - duration
    raise ValueError(f"unknown position: {position}")


def load_excerpt(
    path: str | Path, duration: float, position: str, sr: int = QUERY_SAMPLE_RATE
) -> tuple[np.ndarray, int]:
    """Load one excerpt, mono, at `sr`. Raises if the source is too short.

    Convenience wrapper. When cutting many excerpts from one file, decode once
    with `load_source` and use `slice_excerpt` instead -- seeking inside a
    compressed file per query is orders of magnitude slower.
    """
    y_full, out_sr = load_source(path, sr)
    return slice_excerpt(y_full, out_sr, duration, position), out_sr


def load_source(path: str | Path, sr: int = QUERY_SAMPLE_RATE) -> tuple[np.ndarray, int]:
    """Decode a whole source file once, mono, at `sr`."""
    import librosa

    y, out_sr = librosa.load(path, sr=sr, mono=True)
    return y, int(out_sr)


def slice_excerpt(
    y_full: np.ndarray, sr: int, duration: float, position: str
) -> np.ndarray:
    """Cut an excerpt out of already-decoded audio. Never exceeds the source."""
    total = len(y_full) / sr
    if total < duration:
        raise ValueError(f"source {total:.2f}s shorter than requested {duration:.2f}s")
    start = int(round(excerpt_offset(total, duration, position) * sr))
    n = int(round(duration * sr))
    start = max(0, min(start, len(y_full) - n))
    return y_full[start : start + n]


# ------------------------------------------------------------ transforms ----
def _pink_noise(n: int, rng: np.random.Generator) -> np.ndarray:
    """Pink (1/f) noise via spectral shaping. More music-like than white."""
    white = rng.standard_normal(n)
    spec = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n)
    scale = np.ones_like(freqs)
    scale[1:] = 1.0 / np.sqrt(freqs[1:])
    spec = spec * scale
    out = np.fft.irfft(spec, n=n)
    peak = np.max(np.abs(out))
    return (out / peak) if peak > 0 else out


def add_noise(
    y: np.ndarray, snr_db: float, noise_type: str, rng: np.random.Generator
) -> tuple[np.ndarray, dict]:
    """Mix in noise at a target signal-to-noise ratio (power-based)."""
    n = len(y)
    noise = rng.standard_normal(n) if noise_type == "white" else _pink_noise(n, rng)

    sig_p = float(np.mean(y**2))
    noise_p = float(np.mean(noise**2))
    if sig_p <= 0 or noise_p <= 0:
        return y.copy(), {"applied": False}

    scale = math.sqrt(sig_p / (noise_p * (10 ** (snr_db / 10.0))))
    out = y + scale * noise

    # Uniform gain trim if mixing pushed us past full scale. A uniform gain does
    # not change the SNR, and mirrors the AGC a real capture device applies.
    peak = float(np.max(np.abs(out)))
    gain = 1.0
    if peak > 0.99:
        gain = 0.99 / peak
        out = out * gain

    achieved = 10 * math.log10(sig_p / (noise_p * scale**2)) if scale > 0 else float("inf")
    return out.astype(np.float32), {
        "applied": True,
        "target_snr_db": snr_db,
        "achieved_snr_db": round(achieved, 3),
        "gain_trim": round(gain, 4),
    }


_MP3_LEVEL_CACHE: dict[tuple[int, int], float] = {}
_OPUS_LEVEL_CACHE: dict[int, float] = {}


def _calibrate_level(target_kbps: int, sr: int, fmt: str, subtype: str | None) -> float:
    """Find the compression_level that lands closest to a target bitrate.

    libsndfile exposes VBR/CBR quality as a 0..1 level, not a kbps value, and the
    mapping depends on sample rate. Probing once gives honest targeting; the
    bitrate actually achieved is measured per file and reported regardless.
    """
    dur = 6.0
    t = np.linspace(0, dur, int(sr * dur), endpoint=False)
    rs = np.random.RandomState(1234)
    probe = (
        0.30 * np.sin(2 * np.pi * 440 * t)
        + 0.20 * np.sin(2 * np.pi * 1310 * t)
        + 0.12 * rs.randn(len(t))
    ).astype(np.float32)
    probe = probe / (np.max(np.abs(probe)) + 1e-9) * 0.9

    best_level, best_err = 0.5, float("inf")
    for level in [i / 20 for i in range(0, 20)]:
        buf = io.BytesIO()
        try:
            kw = {"compression_level": level}
            if fmt == "MP3":
                kw["bitrate_mode"] = "CONSTANT"
            sf.write(buf, probe, sr, format=fmt, subtype=subtype, **kw)
        except Exception:  # noqa: BLE001 -- unsupported level, keep scanning
            continue
        kbps = len(buf.getvalue()) * 8 / dur / 1000
        err = abs(kbps - target_kbps)
        if err < best_err:
            best_level, best_err = level, err
    return best_level


def apply_codec(
    y: np.ndarray, sr: int, codec: str, target_kbps: int
) -> tuple[np.ndarray, int, dict]:
    """Encode and decode through a lossy codec; return the decoded audio.

    Uses libsndfile (via soundfile). ffmpeg is not installed in this
    environment, so only libsndfile-supported codecs are exercised.
    """
    dur = len(y) / sr
    if codec == "mp3":
        key = (target_kbps, sr)
        if key not in _MP3_LEVEL_CACHE:
            _MP3_LEVEL_CACHE[key] = _calibrate_level(target_kbps, sr, "MP3", None)
        level = _MP3_LEVEL_CACHE[key]
        buf = io.BytesIO()
        sf.write(
            buf, y, sr, format="MP3", compression_level=level, bitrate_mode="CONSTANT"
        )
    elif codec == "opus":
        # libsndfile's Opus encoder operates at 48 kHz.
        import librosa

        y48 = librosa.resample(y, orig_sr=sr, target_sr=48000)
        if target_kbps not in _OPUS_LEVEL_CACHE:
            _OPUS_LEVEL_CACHE[target_kbps] = _calibrate_level(
                target_kbps, 48000, "OGG", "OPUS"
            )
        level = _OPUS_LEVEL_CACHE[target_kbps]
        buf = io.BytesIO()
        sf.write(buf, y48, 48000, format="OGG", subtype="OPUS", compression_level=level)
    else:
        raise ValueError(f"unsupported codec: {codec}")

    encoded = buf.getvalue()
    measured_kbps = len(encoded) * 8 / dur / 1000
    buf.seek(0)
    decoded, dec_sr = sf.read(buf, dtype="float32", always_2d=False)
    if decoded.ndim > 1:
        decoded = decoded.mean(axis=1)
    return (
        decoded.astype(np.float32),
        int(dec_sr),
        {
            "codec": codec,
            "target_kbps": target_kbps,
            "measured_kbps": round(measured_kbps, 1),
            "compression_level": round(level, 3),
            "decoded_sr": int(dec_sr),
        },
    )


def apply_filter(y: np.ndarray, sr: int, kind: str) -> tuple[np.ndarray, dict]:
    """Band-limit the signal (causal Butterworth, like a real channel)."""
    nyq = sr / 2.0
    if kind == "telephone":
        lo, hi = 300.0, min(3400.0, nyq * 0.99)
        sos = sps.butter(4, [lo / nyq, hi / nyq], btype="bandpass", output="sos")
        info = {"kind": kind, "band_hz": [lo, hi]}
    elif kind == "lowpass_8k":
        cutoff = min(8000.0, nyq * 0.99)
        sos = sps.butter(8, cutoff / nyq, btype="lowpass", output="sos")
        info = {"kind": kind, "cutoff_hz": cutoff}
    else:
        raise ValueError(f"unknown filter: {kind}")
    return sps.sosfilt(sos, y).astype(np.float32), info


def change_speed(y: np.ndarray, sr: int, percent: float) -> tuple[np.ndarray, dict]:
    """Playback-rate change: tempo AND pitch shift together, as a tape/vinyl
    speed change or a slightly-off clock would produce. `percent` > 0 = faster.
    """
    import librosa

    factor = 1.0 + percent / 100.0
    intermediate = int(round(sr / factor))
    out = librosa.resample(y, orig_sr=sr, target_sr=intermediate)
    return out.astype(np.float32), {
        "percent": percent,
        "factor": round(factor, 5),
        "method": "resample_playback_rate",
    }


def shift_pitch(y: np.ndarray, sr: int, semitones: float) -> tuple[np.ndarray, dict]:
    """Pitch shift at constant tempo (librosa phase vocoder)."""
    import librosa

    out = librosa.effects.pitch_shift(y=y, sr=sr, n_steps=semitones)
    return out.astype(np.float32), {
        "semitones": semitones,
        "method": "librosa.effects.pitch_shift",
    }


# ------------------------------------------------------ condition matrix ----
def condition_matrix() -> list[tuple[str, str, dict]]:
    """(condition_label, family, params) for every degradation condition.

    Pitch and speed are held at a single duration; the noise/codec/filter axes
    are crossed with duration because those interact most with clip length.
    """
    conds: list[tuple[str, str, dict]] = [("clean", "clean", {})]

    for snr in (20, 10, 5, 0):
        conds.append(
            (f"noise_pink_snr{snr}db", "noise", {"noise_type": "pink", "snr_db": snr})
        )
    for snr in (20, 10, 5, 0):
        conds.append(
            (f"noise_white_snr{snr}db", "noise", {"noise_type": "white", "snr_db": snr})
        )

    for kbps in (128, 64, 32):
        conds.append((f"codec_mp3_{kbps}k", "codec", {"codec": "mp3", "target_kbps": kbps}))
    for kbps in (64, 32):
        conds.append(
            (f"codec_opus_{kbps}k", "codec", {"codec": "opus", "target_kbps": kbps})
        )

    conds.append(("filter_telephone", "filter", {"kind": "telephone"}))
    conds.append(("filter_lowpass8k", "filter", {"kind": "lowpass_8k"}))

    for pct in (-5, -2, 2, 5):
        conds.append((f"speed_{pct:+d}pct", "speed", {"percent": float(pct)}))

    for st in (-2, -1, 1, 2):
        conds.append((f"pitch_{st:+d}st", "pitch", {"semitones": float(st)}))

    return conds


def apply_condition(
    y: np.ndarray, sr: int, family: str, params: dict, rng: np.random.Generator
) -> tuple[np.ndarray, int, dict]:
    """Dispatch one condition. Returns (audio, sample_rate, measured_info)."""
    if family in ("clean", "negative"):
        # "negative" excerpts are undegraded audio that simply is not in the
        # catalog -- the cleanest possible false-accept test.
        return y, sr, {}
    if family == "noise":
        out, info = add_noise(y, params["snr_db"], params["noise_type"], rng)
        return out, sr, info
    if family == "codec":
        return apply_codec(y, sr, params["codec"], params["target_kbps"])
    if family == "filter":
        out, info = apply_filter(y, sr, params["kind"])
        return out, sr, info
    if family == "speed":
        out, info = change_speed(y, sr, params["percent"])
        return out, sr, info
    if family == "pitch":
        out, info = shift_pitch(y, sr, params["semitones"])
        return out, sr, info
    raise ValueError(f"unknown condition family: {family}")


def render(spec: QuerySpec, y: np.ndarray, sr: int, out_path: str | Path) -> Path:
    """Write a rendered query to disk as 16-bit PCM WAV."""
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    peak = float(np.max(np.abs(y))) if len(y) else 0.0
    if peak > 1.0:
        y = y / peak
    sf.write(p, y.astype(np.float32), sr, subtype="PCM_16")
    return p
