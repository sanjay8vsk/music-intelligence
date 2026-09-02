"""Boot-time artifact synchronisation.

WHY AT BOOT AND NOT ON DEMAND
-----------------------------
The 500-track artifact is 175.1 MB. Fetching it inside a request would put an
object-storage transfer on a path whose accepted p95 is 296.3 ms against a
300 ms bar. Nothing here is reachable from `/v1/identify`; `sync_catalogs()`
runs once, during start-up, before the application accepts traffic, and the
request path is unchanged.

WHICH VERSION, WITHOUT A MUTABLE POINTER
----------------------------------------
Versions are content hashes, so "which one should this instance serve" is a
separate question, and answering it with a mutable `latest` object would put a
mutable pointer back at the centre of an immutable design. Resolution order:

  1. an explicit pin in configuration -- the deployment names the exact hash;
  2. the `catalogs.index_content_hash` column, which the Stage 2 schema already
     records and which is therefore the existing statement of what is current;
  3. the only version in storage, if there is exactly one.

Ambiguity is an error, never a guess.

HOW A FETCH BECOMES ACTIVE
--------------------------
Staging directory -> manifest check -> `CatalogStore.load(verify=True)`, which
is the existing five-way drift verification -> a single rename. A partial or
corrupt download fails before the rename and never becomes the served artifact.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path

from musicintel.catalog.store import CatalogStore, CatalogStoreError
from musicintel.storage.artifacts import (
    ArtifactNotFound,
    StorageError,
    read_artifact_version,
    validate_version,
)

SYNC_DIRNAME = ".sync"


class SyncError(RuntimeError):
    """Synchronisation failed. Start-up must not continue past this."""


@dataclass(frozen=True)
class SyncResult:
    catalog_id: str
    version: str
    action: str          # "fetched" | "already-current" | "skipped"
    bytes_fetched: int = 0
    seconds: float = 0.0


def parse_pins(spec: str) -> dict[str, str]:
    """`acme=<hash>,demo=<hash>` -> mapping. Empty string means no pins."""
    pins: dict[str, str] = {}
    for pair in (spec or "").split(","):
        pair = pair.strip()
        if not pair:
            continue
        if "=" not in pair:
            raise SyncError(f"invalid artifact pin {pair!r}: expected catalog_id=hash")
        cid, _, version = pair.partition("=")
        pins[cid.strip()] = validate_version(version)
    return pins


def resolve_version(catalog_id: str, storage, *, pins: dict[str, str],
                    db_versions: dict[str, str] | None = None) -> str:
    """Pick the version this instance should serve. Never guesses."""
    if catalog_id in pins:
        return pins[catalog_id]
    if db_versions and catalog_id in db_versions and db_versions[catalog_id]:
        return validate_version(db_versions[catalog_id])
    available = storage.list_versions(catalog_id)
    if not available:
        raise SyncError(f"no artifact versions in storage for catalog {catalog_id!r}")
    if len(available) > 1:
        raise SyncError(
            f"catalog {catalog_id!r} has {len(available)} versions in storage and "
            f"no pin or database row says which to serve: {[v[:12] for v in available]}. "
            "Pin it with MUSICINTEL_ARTIFACT_PINS or record it in the catalogs table.")
    return available[0]


def _local_version(catalog_root: Path, catalog_id: str) -> str | None:
    try:
        return read_artifact_version(catalog_root / catalog_id)
    except Exception:
        return None


def sync_catalog(storage, catalog_root: Path, catalog_id: str, version: str) -> SyncResult:
    """Make one catalog available locally at `version`. Verified before active."""
    import time

    catalog_root = Path(catalog_root)
    started = time.perf_counter()

    if _local_version(catalog_root, catalog_id) == version:
        # Already the wanted bytes. Re-downloading 175 MB every boot would be
        # the most expensive no-op in the system.
        return SyncResult(catalog_id, version, "already-current",
                          seconds=time.perf_counter() - started)

    staging_root = catalog_root / SYNC_DIRNAME / f"{catalog_id}.{version[:12]}.{os.getpid()}"
    staging = staging_root / catalog_id
    if staging_root.exists():
        shutil.rmtree(staging_root, ignore_errors=True)
    staging.mkdir(parents=True, exist_ok=True)

    try:
        manifest = storage.get_artifact(catalog_id, version, staging)

        # The existing verification is the source of truth for coherence: the
        # manifest only proves the bytes arrived intact.
        loaded = CatalogStore(staging_root).load(catalog_id, verify=True)
        if loaded.artifact.get("index_content_hash") != version:
            raise SyncError(
                f"artifact for {catalog_id!r} declares version "
                f"{loaded.artifact.get('index_content_hash')!r}, expected {version!r}")

        target = catalog_root / catalog_id
        target.parent.mkdir(parents=True, exist_ok=True)
        previous = None
        if target.exists():
            previous = catalog_root / SYNC_DIRNAME / f"{catalog_id}.old.{os.getpid()}"
            if previous.exists():
                shutil.rmtree(previous, ignore_errors=True)
            os.rename(target, previous)
        try:
            os.rename(staging, target)
        except BaseException:
            if previous is not None and not target.exists():
                os.rename(previous, target)          # put the old one back
            raise
        if previous is not None:
            shutil.rmtree(previous, ignore_errors=True)

        return SyncResult(catalog_id, version, "fetched",
                          bytes_fetched=manifest.total_bytes,
                          seconds=time.perf_counter() - started)
    except (StorageError, CatalogStoreError) as exc:
        raise SyncError(f"catalog {catalog_id!r}: {exc}") from exc
    finally:
        shutil.rmtree(staging_root, ignore_errors=True)


def sync_all(storage, catalog_root: str | Path, *, pins: dict[str, str] | None = None,
             only: list[str] | None = None,
             db_versions: dict[str, str] | None = None,
             logger=None) -> list[SyncResult]:
    """Synchronise every required catalog. Raises SyncError on any failure.

    Failing loudly is the point: an instance that starts with a missing or
    unverifiable catalog would answer real queries against nothing, which is
    worse than not starting.
    """
    catalog_root = Path(catalog_root)
    pins = pins or {}
    wanted = list(only) if only else storage.list_catalogs()
    # A pinned catalog is required even if storage listing missed it.
    for cid in pins:
        if cid not in wanted:
            wanted.append(cid)

    results: list[SyncResult] = []
    for catalog_id in sorted(set(wanted)):
        try:
            version = resolve_version(catalog_id, storage, pins=pins,
                                      db_versions=db_versions)
            result = sync_catalog(storage, catalog_root, catalog_id, version)
        except ArtifactNotFound as exc:
            raise SyncError(f"catalog {catalog_id!r}: {exc}") from exc
        results.append(result)
        if logger is not None:
            logger.info("artifact.synced", catalog_id=catalog_id,
                        version=result.version[:12], action=result.action,
                        bytes=result.bytes_fetched,
                        seconds=round(result.seconds, 3))
    return results


__all__ = [
    "SYNC_DIRNAME", "SyncError", "SyncResult", "parse_pins", "resolve_version",
    "sync_all", "sync_catalog",
]
