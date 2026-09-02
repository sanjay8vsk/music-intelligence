"""Immutable, content-addressed artifact storage (Stage 2).

Versions are `index_content_hash` values -- no numbering scheme is invented
here. Only a filesystem backend exists; no object-storage provider has been
selected and no cloud SDK is imported.
"""

from __future__ import annotations

from musicintel.storage.artifacts import (
    ARTIFACT_MEMBERS,
    ArtifactConflict,
    ArtifactIncomplete,
    ArtifactManifest,
    ArtifactNotFound,
    ArtifactStorage,
    StorageError,
    artifact_key,
    read_artifact_version,
    validate_version,
)
from musicintel.storage.local import LocalArtifactStorage, storage_from_url
from musicintel.storage.sync import (
    SyncError,
    SyncResult,
    parse_pins,
    resolve_version,
    sync_all,
    sync_catalog,
)

__all__ = [
    "ARTIFACT_MEMBERS", "ArtifactConflict", "ArtifactIncomplete",
    "ArtifactManifest", "ArtifactNotFound", "ArtifactStorage",
    "LocalArtifactStorage", "StorageError", "SyncError", "SyncResult",
    "artifact_key", "parse_pins", "read_artifact_version", "resolve_version",
    "storage_from_url", "sync_all", "sync_catalog", "validate_version",
]
