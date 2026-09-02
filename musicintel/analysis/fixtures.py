"""Deterministic synthetic audio with mathematically known BPM and key.

WHAT THESE ARE FOR
------------------
Functional validation only. A click track at exactly 120 BPM has a ground truth
that is true by construction, so a detector that cannot recover it is broken.
That is worth knowing early and cheaply.

WHAT THESE ARE NOT FOR
----------------------
**They cannot substantiate the roadmap's >=90% BPM or >=75% key acceptance
targets.** Those targets are claims about real music -- mixed, performed,
produced, sometimes rubato, sometimes modal. A synthetic click is the easiest
possible case and a I-IV-V-I is an unambiguous tonal statement. Scoring 100%
here says the code runs, not that it works. Any report using these fixtures must
say so.

DETERMINISM
-----------
Every generator is a pure function of its parameters. The only stochastic
element is a small amount of noise, drawn from a generator seeded from the
fixture id, so repeated generation is byte-identical. A test asserts that.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np

from musicintel.analysis.keys import Key, parse_key

DEFAULT_SR = 22050          # not the recognition rate; see docs/analysis-evaluation.md
A4_HZ = 440.0
A4_MIDI = 69


def _rng(fixture_id: str) -> np.random.Generator:
    """A generator seeded from the fixture id: same id, same noise, always."""
    digest = hashlib.sha256(fixture_id.encode("utf-8")).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "big"))


def midi_to_hz(midi: float) -> float:
    return A4_HZ * (2.0 ** ((midi - A4_MIDI) / 12.0))


# ------------------------------------------------------------------ tempo --
def click_track(bpm: float, seconds: float, *, sr: int = DEFAULT_SR,
                fixture_id: str = "click", accent_every: int = 4) -> np.ndarray:
    """A metronome at exactly `bpm`, with an accented downbeat every bar.

    Each click is a short exponentially-decaying noise burst -- broadband, so
    onset detection has something to find at every frequency, and short, so the
    onset is unambiguous in time. The accent gives a bar-level periodicity, which
    is what makes a half-time misreading possible and therefore worth testing.
    """
    if bpm <= 0:
        raise ValueError("bpm must be positive")
    rng = _rng(fixture_id)
    n = int(round(seconds * sr))
    out = np.zeros(n, dtype=np.float32)

    click_len = int(0.010 * sr)                      # 10 ms
    envelope = np.exp(-np.linspace(0, 8, click_len)).astype(np.float32)
    period = 60.0 / bpm

    beat = 0
    while True:
        start = int(round(beat * period * sr))
        if start >= n:
            break
        burst = rng.standard_normal(click_len).astype(np.float32) * envelope
        amp = 1.0 if (accent_every and beat % accent_every == 0) else 0.55
        end = min(start + click_len, n)
        out[start:end] += burst[: end - start] * amp
        beat += 1

    peak = float(np.max(np.abs(out))) or 1.0
    return (out / peak * 0.7).astype(np.float32)


# -------------------------------------------------------------------- key --
_MAJOR_TRIAD = (0, 4, 7)
_MINOR_TRIAD = (0, 3, 7)


def _triad(root_midi: float, minor: bool, seconds: float, sr: int,
           rng: np.random.Generator) -> np.ndarray:
    """One sustained triad, three notes with a couple of partials each."""
    n = int(round(seconds * sr))
    t = np.arange(n, dtype=np.float32) / sr
    out = np.zeros(n, dtype=np.float32)
    for semitone in (_MINOR_TRIAD if minor else _MAJOR_TRIAD):
        f0 = midi_to_hz(root_midi + semitone)
        for partial, weight in ((1, 1.0), (2, 0.35), (3, 0.15)):
            out += weight * np.sin(2 * np.pi * f0 * partial * t).astype(np.float32)
    # Gentle fade so chord boundaries do not click and invent onsets.
    fade = int(0.02 * sr)
    if fade * 2 < n:
        ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
        out[:fade] *= ramp
        out[-fade:] *= ramp[::-1]
    return out


def tonal_progression(key: Key, seconds: float = 12.0, *, sr: int = DEFAULT_SR,
                      fixture_id: str = "tonal") -> np.ndarray:
    """A cadence that states `key` unambiguously.

    Major: I - IV - V - I.  Minor: i - iv - V - i, with a MAJOR dominant, which
    is what actually establishes a minor tonality rather than its relative
    major. Ending on the tonic makes the tonal centre unmistakable -- these are
    the easiest possible key-detection cases, deliberately.
    """
    rng = _rng(fixture_id)
    root = 60 + key.pitch_class                        # around middle C
    if key.is_minor:
        chords = [(root, True), (root + 5, True), (root + 7, False), (root, True)]
    else:
        chords = [(root, False), (root + 5, False), (root + 7, False), (root, False)]

    each = seconds / len(chords)
    parts = [_triad(r, m, each, sr, rng) for r, m in chords]
    out = np.concatenate(parts).astype(np.float32)
    out += (rng.standard_normal(out.size).astype(np.float32) * 0.002)
    peak = float(np.max(np.abs(out))) or 1.0
    return (out / peak * 0.7).astype(np.float32)


# ------------------------------------------------------------- fixture set --
@dataclass(frozen=True)
class SyntheticFixture:
    fixture_id: str
    kind: str                       # "bpm" | "key"
    seconds: float
    sample_rate: int
    bpm: float | None = None
    key: str | None = None
    notes: str = ""
    tags: tuple[str, ...] = field(default_factory=tuple)

    def render(self) -> np.ndarray:
        if self.kind == "bpm":
            return click_track(self.bpm, self.seconds, sr=self.sample_rate,
                               fixture_id=self.fixture_id)
        if self.kind == "key":
            return tonal_progression(parse_key(self.key), self.seconds,
                                     sr=self.sample_rate, fixture_id=self.fixture_id)
        raise ValueError(f"unknown fixture kind {self.kind!r}")

    def to_dict(self) -> dict:
        return {"fixture_id": self.fixture_id, "kind": self.kind,
                "seconds": self.seconds, "sample_rate": self.sample_rate,
                "bpm": self.bpm, "key": self.key, "notes": self.notes,
                "tags": list(self.tags)}


# Tempi chosen so that every value has its double or half also present:
# (60,120) (70,140) (80,160) (90,180). A detector that reports 140 for a 70 BPM
# track is then visibly making an octave error rather than an unrelated one.
_BPM_VALUES = ((60.0, 30.0), (70.0, 30.0), (80.0, 20.0), (90.0, 20.0),
               (100.0, 20.0), (110.0, 20.0), (120.0, 30.0), (128.0, 20.0),
               (140.0, 30.0), (160.0, 20.0), (174.0, 20.0), (180.0, 20.0),
               (95.5, 20.0), (143.0, 10.0))


def synthetic_fixtures(sample_rate: int = DEFAULT_SR) -> list[SyntheticFixture]:
    """The full deterministic fixture set: 14 tempo + 24 key."""
    out: list[SyntheticFixture] = []
    for bpm, secs in _BPM_VALUES:
        half_or_double = any(
            abs(other - bpm * 2) < 1e-6 or abs(other - bpm / 2) < 1e-6
            for other, _ in _BPM_VALUES)
        out.append(SyntheticFixture(
            fixture_id=f"bpm_{bpm:g}", kind="bpm", seconds=secs,
            sample_rate=sample_rate, bpm=bpm,
            notes="click track; ground truth exact by construction",
            tags=("octave-pair",) if half_or_double else
                 (("short",) if secs <= 10.0 else ("non-integer",) if bpm % 1 else ())))
    for minor in (False, True):
        for pc in range(12):
            k = Key(pc, minor)
            out.append(SyntheticFixture(
                fixture_id=f"key_{k.tonic.replace('#','s')}_{k.mode}", kind="key",
                seconds=12.0, sample_rate=sample_rate, key=str(k),
                notes="I-IV-V-I cadence" if not minor else "i-iv-V-i cadence",
                tags=("relative-pair",)))
    return out


__all__ = [
    "A4_HZ", "DEFAULT_SR", "SyntheticFixture", "click_track", "midi_to_hz",
    "synthetic_fixtures", "tonal_progression",
]
