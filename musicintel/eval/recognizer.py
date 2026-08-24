"""Recognizer interface for the evaluation harness, plus the current baseline adapter.

The harness talks only to the `Recognizer` protocol, so the same benchmark can
score the current MFCC/FAISS prototype, a future landmark-fingerprint engine,
and anything after that without the benchmark changing.

Contract:
  * `prepare(tracks)` builds or loads whatever index the recognizer needs.
    Its cost is measured separately and is never counted as query latency.
  * `recognize(path)` returns ranked candidates, best first.
    An EMPTY candidate list means "no match" -- an explicit rejection.
    A recognizer that can never return an empty list has a false accept
    rate of 1.0 by construction, and the harness will report exactly that.
"""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, Sequence, runtime_checkable

import numpy as np

from musicintel.eval.manifest import Track
from musicintel.eval.provenance import (
    ALGORITHM_SOURCES,
    source_fingerprint,
    version_string,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Candidate:
    """One ranked hypothesis."""

    track_id: str
    score: float  # higher is better; recognizer-defined
    distance: float | None = None  # raw distance/cost, for threshold sweeps


@dataclass
class RecognitionResult:
    """Outcome of one query. Empty `candidates` means the recognizer rejected it."""

    candidates: list[Candidate] = field(default_factory=list)

    @property
    def is_match(self) -> bool:
        return len(self.candidates) > 0

    @property
    def top_id(self) -> str | None:
        return self.candidates[0].track_id if self.candidates else None

    def top_k_ids(self, k: int) -> list[str]:
        return [c.track_id for c in self.candidates[:k]]


@runtime_checkable
class Recognizer(Protocol):
    """Minimal interface every recognition engine must satisfy."""

    name: str
    version: str

    def prepare(self, tracks: Sequence[Track]) -> None:
        """Index the catalog. Not counted toward query latency."""
        ...

    def recognize(self, audio_path: str | Path) -> RecognitionResult:
        """Identify one audio file. Empty result == explicit no-match."""
        ...


def _load_legacy_module(rel_path: str, name: str):
    """Import a prototype module from src/ without polluting sys.path.

    The prototypes are not part of the `musicintel` package and are being kept
    untouched until their roadmap stage. Loading them by file path lets the
    adapter call the real code without importing or modifying it.
    """
    path = REPO_ROOT / rel_path
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


class MfccFaissRecognizer:
    """Adapter around the CURRENT prototype recognizer (src/build_index.py +
    src/music_recognition.py) so its real baseline can be measured.

    ALGORITHM -- reproduced exactly, nothing tuned or improved:
      reference: librosa.load(sr=22050, duration=30) -> mfcc(n_mfcc=13)
                 -> concat(mean, std) = 26-d
      query:     librosa.load(<no duration>, sr=22050) -> mfcc(n_mfcc=13)
                 -> concat(mean, std) = 26-d
      search:    faiss.IndexFlatL2(26), L2 nearest neighbour, k=3

    The 30 s reference / full-length query asymmetry is present in the original
    and is deliberately preserved.

    INFRASTRUCTURE-ONLY deviations (necessary to run a benchmark at all; each is
    reported in the baseline document):
      1. The index is built once in prepare() rather than rebuilt inside every
         call, as src/music_recognition.py:16 does. Rebuild cost is measured and
         reported separately instead of being charged to every query.
      2. Feature files are named with os.path.splitext rather than
         src/audio_processing.py:39's `file.split(".")[0]`, which collapses every
         dotted filename in a folder onto one .npy and would silently destroy the
         catalog.
      3. k is clamped to the catalog size and FAISS's -1 "not found" sentinel is
         filtered, instead of src/music_recognition.py:23's `song_names[-1]`,
         which silently aliases the last track.
      4. Ranked candidates and their L2 distances are returned; the original
         returns bare names. Needed for Recall@3 and the threshold sweep.

    None of these changes the features, the metric, or the ranking.
    """

    name = "mfcc_faiss_baseline"
    # Derived from git per instance (see __init__), never hardcoded: a pinned
    # literal keeps asserting a commit long after the code beneath it moved.
    VERSION_PREFIX = "prototype"
    version = "prototype@unknown"

    N_MFCC = 13
    SAMPLE_RATE = 22050
    REFERENCE_DURATION = 30  # seconds; matches audio_processing.extract_mfcc
    TOP_K = 3

    def __init__(self, feature_dir: str | Path | None = None) -> None:
        self.feature_dir = Path(feature_dir or REPO_ROOT / "data/eval/_mfcc_features")
        self._index = None
        self._track_ids: list[str] = []
        self.prepare_seconds: float = 0.0
        self.index_build_seconds: float = 0.0
        self.indexed_tracks: int = 0
        # Provenance of the algorithm actually being measured. The version comes
        # from git state (with a +dirty marker when the tree is not clean); the
        # fingerprint pins the exact src/ bytes even when no commit describes them.
        self.version = version_string(self.VERSION_PREFIX, REPO_ROOT)
        self.algorithm_sha256 = source_fingerprint(REPO_ROOT, ALGORITHM_SOURCES)

    # -- reference side (mirrors src/audio_processing.py:extract_mfcc) ----
    @classmethod
    def _reference_mfcc(cls, path: str | Path) -> np.ndarray:
        import librosa

        y, sr = librosa.load(path, sr=cls.SAMPLE_RATE, duration=cls.REFERENCE_DURATION)
        return librosa.feature.mfcc(y=y, sr=sr, n_mfcc=cls.N_MFCC)

    # -- query side (mirrors src/music_recognition.py:recognize_song) -----
    @classmethod
    def _query_embedding(cls, path: str | Path) -> np.ndarray:
        import librosa

        # NOTE: no `duration=` here. The original omits it on the query path.
        y, sr = librosa.load(path)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=cls.N_MFCC)
        mean = np.mean(mfcc, axis=1)
        std = np.std(mfcc, axis=1)
        return np.concatenate([mean, std]).astype("float32").reshape(1, -1)

    def prepare(self, tracks: Sequence[Track]) -> None:
        import time

        t0 = time.perf_counter()
        self.feature_dir.mkdir(parents=True, exist_ok=True)
        for old in self.feature_dir.glob("*.npy"):
            old.unlink()

        self._track_ids = []
        for t in tracks:
            mfcc = self._reference_mfcc(REPO_ROOT / t.path)
            np.save(self.feature_dir / f"{t.track_id}.npy", mfcc)
            self._track_ids.append(t.track_id)

        # Build the index with the prototype's own code, pointed at our folder.
        build_index = _load_legacy_module("src/build_index.py", "_legacy_build_index")
        build_index.FEATURE_FOLDER = str(self.feature_dir)
        t1 = time.perf_counter()
        index, names = build_index.build_faiss_index()
        self.index_build_seconds = time.perf_counter() - t1

        self._index = index
        # build_faiss_index returns the .npy filenames, in os.listdir order.
        self._track_ids = [Path(n).stem for n in names]
        # Recorded so a report can prove what was actually indexed. The feature
        # dir is cleared above, but a stale or file-sync-duplicated .npy would
        # otherwise silently enlarge the catalog leaving no trace in the report.
        self.indexed_tracks = len(self._track_ids)
        self.prepare_seconds = time.perf_counter() - t0

    def recognize(self, audio_path: str | Path) -> RecognitionResult:
        if self._index is None:
            raise RuntimeError("prepare() must be called before recognize()")
        q = self._query_embedding(audio_path)
        k = min(self.TOP_K, self._index.ntotal)
        if k == 0:
            return RecognitionResult([])
        distances, indices = self._index.search(q, k)  # type: ignore[union-attr]

        candidates: list[Candidate] = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0:  # FAISS "no neighbour" sentinel
                continue
            candidates.append(
                Candidate(
                    track_id=self._track_ids[idx],
                    score=float(-dist),  # higher is better
                    distance=float(dist),
                )
            )
        # No rejection stage exists in the prototype: it always answers.
        return RecognitionResult(candidates)
