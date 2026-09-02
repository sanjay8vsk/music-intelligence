"""MusicBrainz metadata enrichment (Stage 2).

Offline only. Nothing in this package is imported by the API, and no lookup can
occur during start-up, in a request handler, or anywhere on the recognition
path. AcoustID and Chromaprint are deliberately absent.
"""

from __future__ import annotations

from musicintel.enrichment.musicbrainz import (
    ContactRequired,
    LookupResult,
    MusicBrainzClient,
    MusicBrainzError,
    PermanentError,
    RateLimiter,
    TransientError,
)
from musicintel.enrichment.normalize import NormalizedMetadata, normalize_recording

__all__ = [
    "ContactRequired", "LookupResult", "MusicBrainzClient", "MusicBrainzError",
    "NormalizedMetadata", "PermanentError", "RateLimiter", "TransientError",
    "normalize_recording",
]
