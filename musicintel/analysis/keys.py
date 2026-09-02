"""A 24-class musical key representation, and the relations metrics need.

WHY THIS EXISTS SEPARATELY
--------------------------
"Key accuracy" is meaningless without a declared representation. `C#` and `Db`
are the same pitch and different spellings; `C major` and `A minor` share every
note. A number that silently folds those together, or silently splits them,
cannot be compared to anyone else's number -- including the roadmap's 75%
target. So the policy lives here, in one place, and every metric reads it.

THE POLICY
----------
* **12 pitch classes, sharp spelling canonical.** `Db`, `C#` and `B##` all
  normalise to pitch class 1, printed `C#`. Spelling is notation; pitch class is
  what a detector can actually recover from audio.
* **2 modes**, major and minor. 12 x 2 = the 24 classes.
* **No key is a first-class value**, not a missing one. Atonal or ambiguous
  material is `NO_KEY`, and metrics count it as an exclusion rather than a
  wrong answer -- see `musicintel/analysis/evaluation.py`.

MIREX WEIGHTS
-------------
Exact 1.0, perfect fifth 0.5, relative major/minor 0.3, parallel major/minor
0.2, everything else 0.0. These are the long-standing MIREX audio-key-detection
weights, used here so the score is comparable to published work rather than
invented. `dominant` and `subdominant` both score 0.5 but are reported
separately, because they are different musical mistakes.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

PITCH_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")

# Every spelling that maps onto a pitch class, including double accidentals and
# the enharmonics that actually appear in key names.
_BASE = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
_MAJOR_WORDS = {"major", "maj", "M", "dur", ""}
_MINOR_WORDS = {"minor", "min", "m", "moll"}

MIREX_EXACT = 1.0
MIREX_FIFTH = 0.5
MIREX_RELATIVE = 0.3
MIREX_PARALLEL = 0.2
MIREX_OTHER = 0.0


class KeyParseError(ValueError):
    """The string is not a key this policy recognises."""


@dataclass(frozen=True, order=True)
class Key:
    """One of the 24 classes. `pitch_class` is 0-11 with C = 0."""

    pitch_class: int
    is_minor: bool

    def __post_init__(self) -> None:
        if not 0 <= self.pitch_class <= 11:
            raise KeyParseError(f"pitch class {self.pitch_class} outside 0-11")

    @property
    def mode(self) -> str:
        return "minor" if self.is_minor else "major"

    @property
    def tonic(self) -> str:
        return PITCH_NAMES[self.pitch_class]

    def __str__(self) -> str:
        return f"{self.tonic} {self.mode}"

    @property
    def index(self) -> int:
        """0-23, for confusion-matrix rows. Majors 0-11, then minors 12-23."""
        return self.pitch_class + (12 if self.is_minor else 0)

    # -- relations -------------------------------------------------------
    def relative(self) -> "Key":
        """C major <-> A minor. Same notes, different tonic."""
        if self.is_minor:
            return Key((self.pitch_class + 3) % 12, False)
        return Key((self.pitch_class + 9) % 12, True)

    def parallel(self) -> "Key":
        """C major <-> C minor. Same tonic, different mode."""
        return Key(self.pitch_class, not self.is_minor)

    def dominant(self) -> "Key":
        """A perfect fifth up, same mode."""
        return Key((self.pitch_class + 7) % 12, self.is_minor)

    def subdominant(self) -> "Key":
        """A perfect fourth up (fifth down), same mode."""
        return Key((self.pitch_class + 5) % 12, self.is_minor)


NO_KEY = "no_key"          # sentinel for atonal / no stable tonal centre


def parse_key(text: str) -> Key:
    """Parse a key name under the declared policy. Raises on anything else.

    Accepts `C`, `C major`, `Cmaj`, `C#m`, `Db minor`, `F# Minor`, `Bbm`, and
    the same with separators or different case. Rejects silently-wrong input
    rather than guessing -- a mis-parsed label is worse than a rejected one.
    """
    if text is None:
        raise KeyParseError("key is None")
    s = str(text).strip()
    if not s:
        raise KeyParseError("key is empty")

    m = re.match(r"^([A-Ga-g])([#b♯♭]*)\s*[-_ ]?\s*(.*)$", s)
    if not m:
        raise KeyParseError(f"cannot parse key {text!r}")
    letter, accidentals, rest = m.group(1).upper(), m.group(2), m.group(3).strip()

    pc = _BASE[letter]
    for ch in accidentals:
        if ch in ("#", "♯"):
            pc += 1
        elif ch in ("b", "♭"):
            pc -= 1
    pc %= 12

    word = rest.lower().replace(".", "")
    if word in {w.lower() for w in _MINOR_WORDS}:
        return Key(pc, True)
    if word in {w.lower() for w in _MAJOR_WORDS} or rest == "M":
        return Key(pc, False)
    raise KeyParseError(f"unrecognised mode {rest!r} in key {text!r}")


def normalize_key(text: str) -> str:
    """Canonical string form: sharp spelling, lowercase mode word."""
    return str(parse_key(text))


ALL_KEYS: tuple[Key, ...] = tuple(
    Key(pc, minor) for minor in (False, True) for pc in range(12))
KEY_LABELS: tuple[str, ...] = tuple(str(k) for k in ALL_KEYS)


def relation(truth: Key, predicted: Key) -> str:
    """How `predicted` is wrong, in musical terms.

    Order matters: a prediction can satisfy more than one description, and the
    closer relation wins. `exact` first, then `relative` and `parallel` (which
    are specific), then the fifths (which are broader).
    """
    if predicted == truth:
        return "exact"
    if predicted == truth.relative():
        return "relative"
    if predicted == truth.parallel():
        return "parallel"
    if predicted == truth.dominant():
        return "dominant"
    if predicted == truth.subdominant():
        return "subdominant"
    return "other"


def mirex_score(truth: Key, predicted: Key) -> float:
    """MIREX weight for one prediction."""
    return {
        "exact": MIREX_EXACT,
        "dominant": MIREX_FIFTH,
        "subdominant": MIREX_FIFTH,
        "relative": MIREX_RELATIVE,
        "parallel": MIREX_PARALLEL,
        "other": MIREX_OTHER,
    }[relation(truth, predicted)]


__all__ = [
    "ALL_KEYS", "KEY_LABELS", "MIREX_EXACT", "MIREX_FIFTH", "MIREX_PARALLEL",
    "MIREX_RELATIVE", "NO_KEY", "PITCH_NAMES", "Key", "KeyParseError",
    "mirex_score", "normalize_key", "parse_key", "relation",
]
