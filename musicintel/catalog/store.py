"""Multi-tenant catalog storage: one immutable artifact per catalog.

WHY ISOLATION IS STRUCTURAL HERE
--------------------------------
The roadmap asks for `catalog_id` on every hash row with isolation tests. That
shape puts a tenant tag on each posting and filters at query time, which means
isolation holds only as long as every lookup path remembers to apply the filter
-- and one forgotten filter leaks another tenant's catalog.

This gives each catalog its own index artifact instead. A query against catalog
A loads A's index and physically cannot see B's postings, because they are not
in the array being searched. There is no filter to forget. It also leaves
`musicintel/recognition/index.py` untouched, which the freeze requires.

The cost is honest: one index per tenant means no cross-catalog query and more
resident memory when many catalogs are open at once. Both are acceptable at this
stage and neither is hidden.

ARTIFACT LAYOUT
---------------
    <root>/<catalog_id>/
        catalog.json     identity, provenance, licensing -- no audio
        index/           the Phase 1B index artifact, format unchanged
        artifact.json    binds the two together and versions the pair

`artifact.json` is what makes "reproducible from the manifest" checkable: it
records the catalog's content hash and the index's content hash, so a rebuild
from the same manifest can be proven identical, and drift between a catalog and
the index built from it is detected rather than silently served.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from musicintel.catalog.models import Catalog
from musicintel.recognition.fingerprint import FORMAT_VERSION
from musicintel.recognition.index import INDEX_FORMAT_VERSION, FingerprintIndex

ARTIFACT_VERSION = 1
CATALOG_FILENAME = "catalog.json"
INDEX_DIRNAME = "index"
ARTIFACT_FILENAME = "artifact.json"

# Catalog ids become directory names, so they must not be able to escape the
# store root or collide case-insensitively on a case-insensitive filesystem.
_VALID_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


class CatalogStoreError(RuntimeError):
    """The store could not serve a trustworthy catalog."""


def validate_catalog_id(catalog_id: str) -> str:
    """Reject anything that could traverse out of the store root."""
    if not isinstance(catalog_id, str) or not _VALID_ID.match(catalog_id):
        raise CatalogStoreError(
            f"invalid catalog_id {catalog_id!r}: must match {_VALID_ID.pattern}")
    if catalog_id in (".", "..") or "/" in catalog_id or "\\" in catalog_id:
        raise CatalogStoreError(f"invalid catalog_id {catalog_id!r}")
    return catalog_id


@dataclass(frozen=True)
class LoadedCatalog:
    """A catalog and the index built from it, verified to belong together."""

    catalog_id: str
    catalog: Catalog
    index: FingerprintIndex
    artifact: dict

    @property
    def track_count(self) -> int:
        return len(self.catalog)

    @property
    def fingerprint_count(self) -> int:
        return len(self.index)


class CatalogStore:
    """Reads and writes per-catalog artifacts under one root."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    # -- layout ---------------------------------------------------------
    def path_for(self, catalog_id: str) -> Path:
        return self.root / validate_catalog_id(catalog_id)

    def exists(self, catalog_id: str) -> bool:
        return (self.path_for(catalog_id) / ARTIFACT_FILENAME).is_file()

    def list_catalogs(self) -> list[str]:
        if not self.root.is_dir():
            return []
        return sorted(p.name for p in self.root.iterdir()
                      if (p / ARTIFACT_FILENAME).is_file())

    # -- write ----------------------------------------------------------
    def save(
        self,
        catalog: Catalog,
        index: FingerprintIndex,
        *,
        catalog_id: str | None = None,
        include_timestamp: bool = False,
    ) -> Path:
        """Write one catalog and its index as a single versioned artifact.

        The index's track ids must match the catalog's exactly. A mismatch means
        the index was built from different audio than the catalog describes, and
        serving that would let a query return a track the catalog cannot explain.
        """
        cid = validate_catalog_id(catalog_id or catalog.catalog_id)
        catalog.catalog_id = cid

        cat_ids, idx_ids = set(catalog.track_ids), set(index.track_ids)
        if cat_ids != idx_ids:
            only_c, only_i = sorted(cat_ids - idx_ids), sorted(idx_ids - cat_ids)
            raise CatalogStoreError(
                f"catalog and index disagree: {len(only_c)} only in catalog "
                f"{only_c[:3]}, {len(only_i)} only in index {only_i[:3]}")

        out = self.path_for(cid)
        out.mkdir(parents=True, exist_ok=True)
        catalog.save(out / CATALOG_FILENAME)
        index.save(out / INDEX_DIRNAME, include_timestamp=include_timestamp)

        artifact = {
            "artifact_version": ARTIFACT_VERSION,
            "catalog_id": cid,
            "catalog_content_hash": catalog.content_hash(),
            "index_content_hash": index.content_hash(),
            "track_count": len(catalog),
            "fingerprint_count": len(index),
            "fingerprint_format_version": FORMAT_VERSION,
            "index_format_version": INDEX_FORMAT_VERSION,
            "built_utc": (datetime.now(timezone.utc).isoformat(timespec="seconds")
                          if include_timestamp else None),
        }
        (out / ARTIFACT_FILENAME).write_text(
            json.dumps(artifact, indent=2, sort_keys=True) + "\n")
        return out

    # -- read -----------------------------------------------------------
    def load(self, catalog_id: str, *, verify: bool = True) -> LoadedCatalog:
        """Load one catalog. Nothing outside `<root>/<catalog_id>/` is read."""
        cid = validate_catalog_id(catalog_id)
        d = self.path_for(cid)
        ap = d / ARTIFACT_FILENAME
        if not ap.is_file():
            raise CatalogStoreError(f"no catalog {cid!r} under {self.root}")
        try:
            artifact = json.loads(ap.read_text())
        except json.JSONDecodeError as e:
            raise CatalogStoreError(f"{ARTIFACT_FILENAME} is not valid JSON: {e}") from e

        got = artifact.get("artifact_version")
        if got != ARTIFACT_VERSION:
            raise CatalogStoreError(
                f"artifact version {got!r}, this build reads {ARTIFACT_VERSION}")
        if artifact.get("catalog_id") != cid:
            raise CatalogStoreError(
                f"artifact claims catalog_id {artifact.get('catalog_id')!r} but "
                f"sits in {cid!r} -- the directory was renamed or copied")

        catalog = Catalog.load(d / CATALOG_FILENAME)
        index = FingerprintIndex.load(d / INDEX_DIRNAME)

        if verify:
            problems = self._drift(artifact, catalog, index, cid)
            if problems:
                raise CatalogStoreError(
                    f"catalog {cid!r} failed verification: " + "; ".join(problems))
        return LoadedCatalog(catalog_id=cid, catalog=catalog, index=index,
                             artifact=artifact)

    @staticmethod
    def _drift(artifact: dict, catalog: Catalog, index: FingerprintIndex,
               cid: str) -> list[str]:
        """Every way a stored catalog and its index can disagree."""
        out: list[str] = []
        if catalog.catalog_id != cid:
            out.append(f"catalog.json says {catalog.catalog_id!r}, not {cid!r}")
        if catalog.content_hash() != artifact.get("catalog_content_hash"):
            out.append("catalog content hash does not match the artifact")
        if index.content_hash() != artifact.get("index_content_hash"):
            out.append("index content hash does not match the artifact")
        if set(catalog.track_ids) != set(index.track_ids):
            out.append("catalog and index hold different track ids")
        if artifact.get("fingerprint_format_version") != FORMAT_VERSION:
            out.append(
                f"index holds fingerprint format "
                f"{artifact.get('fingerprint_format_version')!r}, this build "
                f"produces {FORMAT_VERSION}")
        return out

    def describe(self, catalog_id: str) -> dict:
        """Artifact metadata without loading the index arrays."""
        cid = validate_catalog_id(catalog_id)
        ap = self.path_for(cid) / ARTIFACT_FILENAME
        if not ap.is_file():
            raise CatalogStoreError(f"no catalog {cid!r} under {self.root}")
        return json.loads(ap.read_text())


__all__ = [
    "ARTIFACT_FILENAME", "ARTIFACT_VERSION", "CATALOG_FILENAME", "INDEX_DIRNAME",
    "CatalogStore", "CatalogStoreError", "LoadedCatalog", "validate_catalog_id",
]
