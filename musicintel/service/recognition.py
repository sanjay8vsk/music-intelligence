"""End-to-end recognition over a stored catalog.

    audio -> catalog store -> frozen recognizer -> gated speed cascade -> verdict

This is the wiring layer, and only the wiring layer. It owns no DSP, no hashing,
no matching, no scoring and no thresholds of its own: it loads the right
catalog, calls the frozen pipeline, and attaches the catalog metadata the
recognizer never sees. Every accuracy number the system has was measured on the
components underneath, and nothing here changes them.

WHAT IT ADDS OVER CALLING THE CASCADE DIRECTLY
----------------------------------------------
1. TENANCY. A query names a catalog, and only that catalog's index is loaded, so
   isolation is structural rather than a filter that could be skipped.
2. IDENTITY. The recognizer returns a `track_id`; a caller wants the track --
   its source, duration, hash and metadata. That join lives here.
3. POSITION. The offset histogram already knows where in the reference the query
   landed; exposing it in seconds costs nothing and is the basis of timestamped
   recognition.
4. CACHING. Catalogs stay loaded between queries, because building an index
   costs seconds and a query costs milliseconds.

THRESHOLDS ARE CALIBRATED CONSTANTS, NOT DEFAULTS TO TASTE
-----------------------------------------------------------
The stage-1, gate and stage-2 thresholds below are the values derived on the
calibration split and recorded in eval/reports/phase1h_gated_benchmark.md. They
are reproduced here so a service instance behaves like the benchmarked system;
changing them invalidates every published number.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from musicintel.catalog.models import CatalogTrack
from musicintel.catalog.store import CatalogStore, LoadedCatalog
from musicintel.recognition.decision import Decision
from musicintel.recognition.fingerprint import FingerprintConfig, load_audio
from musicintel.recognition.gated_cascade import (
    GATED_RATE_GRID,
    PROBE_SECONDS,
    GatedCascadeConfig,
    identify_gated,
)
from musicintel.recognition.matcher import MatchConfig, warm_up as warm_up_matcher

# Calibrated on the calibration split; see eval/reports/phase1h_gated_benchmark.md.
STAGE1_THRESHOLD = 0.026316
GATE_THRESHOLD = 0.032520
STAGE2_THRESHOLD = 0.028571
MIN_ALIGNED_LANDMARKS = 5


def default_cascade_config() -> GatedCascadeConfig:
    """The benchmarked operating point, as a config object."""
    return GatedCascadeConfig(
        rate_grid=GATED_RATE_GRID,
        stage1_threshold=STAGE1_THRESHOLD,
        gate_threshold=GATE_THRESHOLD,
        probe_seconds=PROBE_SECONDS,
        stage2_threshold=STAGE2_THRESHOLD,
        min_aligned_landmarks=MIN_ALIGNED_LANDMARKS,
    )


@dataclass(frozen=True)
class Identification:
    """One verdict, with the track it names and the evidence behind it.

    `evidence_score` is a rate, not a probability -- the decision layer's
    semantics carry through unchanged. There is deliberately no confidence field.
    """

    decision: Decision
    catalog_id: str
    track_id: str | None
    track: CatalogTrack | None
    stage: int | None
    rate_percent: float
    evidence_score: float
    threshold: float
    aligned_landmarks: int
    query_landmarks: int
    offset_seconds: float | None
    escalated: bool
    latency_ms: float

    @property
    def is_match(self) -> bool:
        return self.decision is Decision.MATCH

    def to_dict(self) -> dict:
        return {
            "decision": self.decision.value,
            "catalog_id": self.catalog_id,
            "track_id": self.track_id,
            "track": self.track.to_dict() if self.track else None,
            "stage": self.stage,
            "rate_percent": self.rate_percent,
            "evidence_score": round(self.evidence_score, 6),
            "threshold": self.threshold,
            "aligned_landmarks": self.aligned_landmarks,
            "query_landmarks": self.query_landmarks,
            "offset_seconds": (round(self.offset_seconds, 3)
                               if self.offset_seconds is not None else None),
            "escalated": self.escalated,
            "latency_ms": round(self.latency_ms, 3),
        }


class RecognitionService:
    """Identify audio against a named catalog."""

    def __init__(
        self,
        store: CatalogStore,
        *,
        cascade_config: GatedCascadeConfig | None = None,
        match_config: MatchConfig | None = None,
        fingerprint_config: FingerprintConfig | None = None,
        cache_catalogs: bool = True,
        warm_up: bool = True,
    ) -> None:
        self.store = store
        self.cascade_config = cascade_config or default_cascade_config()
        self.match_config = match_config or MatchConfig()
        self.fingerprint_config = fingerprint_config or FingerprintConfig()
        self._cache: dict[str, LoadedCatalog] = {}
        self._cache_enabled = cache_catalogs
        # The matcher's hot loop is JIT-compiled, and the first call pays for it
        # (~1.1 s measured). Doing that here means it lands in process start-up
        # rather than in whichever query happens to arrive first. Anything that
        # constructs a service is warm before it serves.
        self.warm_up_seconds = warm_up_matcher() if warm_up else 0.0

    # -- catalogs ---------------------------------------------------------
    def catalogs(self) -> list[str]:
        return self.store.list_catalogs()

    def get(self, catalog_id: str) -> LoadedCatalog:
        """Load a catalog, from cache when possible.

        Only the named catalog is ever touched -- this is where tenant isolation
        actually holds, because no other catalog's index is in memory to search.
        """
        if self._cache_enabled and catalog_id in self._cache:
            return self._cache[catalog_id]
        loaded = self.store.load(catalog_id)
        if self._cache_enabled:
            self._cache[catalog_id] = loaded
        return loaded

    def unload(self, catalog_id: str | None = None) -> None:
        if catalog_id is None:
            self._cache.clear()
        else:
            self._cache.pop(catalog_id, None)

    # -- identify ----------------------------------------------------------
    def identify(
        self, y: np.ndarray, sr: int, catalog_id: str
    ) -> Identification:
        """Identify already-decoded audio against one catalog."""
        loaded = self.get(catalog_id)
        t0 = time.perf_counter()
        res = identify_gated(
            y, sr, loaded.index,
            config=self.cascade_config,
            match_config=self.match_config,
            fingerprint_config=self.fingerprint_config,
        )
        elapsed = (time.perf_counter() - t0) * 1000.0

        track = loaded.catalog.by_id(res.track_id) if res.track_id else None
        offset = None
        if res.is_match and res.best_offset is not None:
            cfg = loaded.index.config
            # The offset is measured against the reference index, so it is
            # already in reference time even when the query was rate-corrected.
            offset = res.best_offset * cfg.hop_length / cfg.sample_rate

        return Identification(
            decision=res.decision, catalog_id=catalog_id,
            track_id=res.track_id, track=track, stage=res.stage,
            rate_percent=res.rate_percent, evidence_score=res.evidence_score,
            threshold=res.threshold, aligned_landmarks=res.aligned_landmarks,
            query_landmarks=res.query_landmarks, offset_seconds=offset,
            escalated=res.escalated, latency_ms=elapsed,
        )

    def identify_file(self, path: str | Path, catalog_id: str) -> Identification:
        """Decode a file and identify it. Decode time is excluded from latency_ms,
        as it is in every benchmark this system has published."""
        y, sr = load_audio(path, self.fingerprint_config)
        return self.identify(y, sr, catalog_id)


__all__ = [
    "GATE_THRESHOLD", "MIN_ALIGNED_LANDMARKS", "STAGE1_THRESHOLD",
    "STAGE2_THRESHOLD", "Identification", "RecognitionService",
    "default_cascade_config",
]
