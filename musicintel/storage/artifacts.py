"""Content-addressed artifact storage: keys, manifests, and the interface.

WHAT AN ARTIFACT IS HERE
------------------------
Exactly what `CatalogStore` already writes -- `artifact.json`, `catalog.json`
and `index/` -- uploaded file for file. No archive, no new packaging format.

Two reasons not to tar it. The index arrays are `.npy` files that `np.load`
reads from real paths, so a package would have to be unpacked on arrival
anyway; and packaging would create a second representation of an artifact whose
on-disk layout is already verified by `CatalogStore.load()`. Atomicity does not
require a package: a download lands in a staging directory, is verified there by
the existing machinery, and is moved into place with a single rename. A partial
download simply never reaches the rename.

VERSIONS ARE CONTENT HASHES
---------------------------
A version *is* `index_content_hash` -- no counter, no timestamp, no scheme
invented here. That hash already covers the index format version, the
fingerprint format version, the complete fingerprint configuration, per-track
entries and all three array payloads, and deliberately excludes `built_utc`, so
rebuilding the same catalog with the same configuration yields the same
version.

A CONSEQUENCE WORTH KNOWING
--------------------------
Neither content hash covers `title` or `artist`. `index_content_hash` is built
from `TrackEntry(track_id, fingerprint_count, duration_sec)`, and
`catalog_content_hash` from `(track_id, sha256)` pairs. So editing only a
track's title changes neither hash while changing `catalog.json`'s bytes.
Publishing that produces the same key with different content, which is refused
as `ArtifactConflict` rather than silently overwriting. That is the correct
behaviour for a content-addressed store -- see `docs/` and the report -- but it
means metadata-only edits need a rebuild to publish.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Protocol, runtime_checkable

from musicintel.catalog.store import (
    ARTIFACT_FILENAME,
    CATALOG_FILENAME,
    INDEX_DIRNAME,
    validate_catalog_id,
)

MANIFEST_FILENAME = "MANIFEST.json"
KEY_PREFIX = "catalogs"

# Everything a complete artifact consists of, relative to the catalog directory.
ARTIFACT_MEMBERS: tuple[str, ...] = (
    ARTIFACT_FILENAME,
    CATALOG_FILENAME,
    f"{INDEX_DIRNAME}/meta.json",
    f"{INDEX_DIRNAME}/hashes.npy",
    f"{INDEX_DIRNAME}/track_ords.npy",
    f"{INDEX_DIRNAME}/anchor_frames.npy",
)

_HEX64 = set("0123456789abcdef")


class StorageError(RuntimeError):
    """Base class for every storage failure."""


class ArtifactNotFound(StorageError):
    pass


class ArtifactConflict(StorageError):
    """A key already holds different bytes. Never resolved by overwriting."""


class ArtifactIncomplete(StorageError):
    """A stored or fetched artifact is missing members or fails its manifest."""


def validate_version(version: str) -> str:
    """A version is an index content hash: 64 lowercase hex characters."""
    v = (version or "").strip().lower()
    if len(v) != 64 or not set(v) <= _HEX64:
        raise StorageError(
            f"invalid artifact version {version!r}: expected a 64-character "
            "index content hash")
    return v


def artifact_key(catalog_id: str, version: str) -> str:
    """`catalogs/<catalog_id>/<index_content_hash>` -- the immutable prefix.

    `catalog_id` is validated with the store's own rule, so a key can never
    traverse outside its catalog. That is what keeps per-catalog isolation
    structural in storage as well as on disk.
    """
    return f"{KEY_PREFIX}/{validate_catalog_id(catalog_id)}/{validate_version(version)}"


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


@dataclass(frozen=True)
class ArtifactManifest:
    """Per-member digests, so a fetch can be checked before it is trusted.

    This is not a replacement for `CatalogStore.load(verify=True)` -- that
    remains the source of truth for whether an artifact is coherent. The
    manifest answers a narrower question the store cannot: did every byte we
    stored arrive.
    """

    catalog_id: str
    version: str
    members: dict[str, dict]        # relative path -> {"sha256": ..., "bytes": ...}

    @property
    def total_bytes(self) -> int:
        return sum(int(m["bytes"]) for m in self.members.values())

    def to_json(self) -> str:
        return json.dumps(
            {"catalog_id": self.catalog_id, "version": self.version,
             "members": self.members},
            indent=2, sort_keys=True) + "\n"

    @classmethod
    def from_json(cls, text: str) -> "ArtifactManifest":
        try:
            d = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ArtifactIncomplete(f"manifest is not valid JSON: {exc}") from exc
        try:
            return cls(catalog_id=d["catalog_id"], version=d["version"],
                       members=d["members"])
        except KeyError as exc:
            raise ArtifactIncomplete(f"manifest missing {exc}") from exc

    @classmethod
    def build(cls, directory: Path, catalog_id: str, version: str) -> "ArtifactManifest":
        members: dict[str, dict] = {}
        for rel in ARTIFACT_MEMBERS:
            p = directory / rel
            if not p.is_file():
                raise ArtifactIncomplete(
                    f"artifact for {catalog_id!r} is missing {rel}")
            members[rel] = {"sha256": sha256_file(p), "bytes": p.stat().st_size}
        return cls(catalog_id=catalog_id, version=validate_version(version),
                   members=members)

    def verify_directory(self, directory: Path) -> list[str]:
        """Every way `directory` fails to match this manifest."""
        problems: list[str] = []
        for rel, meta in sorted(self.members.items()):
            p = directory / rel
            if not p.is_file():
                problems.append(f"missing {rel}")
                continue
            size = p.stat().st_size
            if size != int(meta["bytes"]):
                problems.append(
                    f"{rel}: {size} bytes, manifest says {meta['bytes']}")
                continue
            if sha256_file(p) != meta["sha256"]:
                problems.append(f"{rel}: content hash does not match the manifest")
        return problems

    def differs_from(self, other: "ArtifactManifest") -> bool:
        return self.members != other.members


@runtime_checkable
class ArtifactStorage(Protocol):
    """The narrowest interface a backend must provide.

    Deliberately not a filesystem abstraction: four operations over immutable,
    content-addressed artifacts. An S3-compatible backend implements the same
    four without any change to callers.
    """

    def put_artifact(self, catalog_id: str, version: str, source: Path) -> str: ...
    def get_artifact(self, catalog_id: str, version: str, destination: Path) -> ArtifactManifest: ...
    def exists(self, catalog_id: str, version: str) -> bool: ...
    def list_versions(self, catalog_id: str) -> list[str]: ...
    def list_catalogs(self) -> list[str]: ...


def read_artifact_version(directory: Path) -> str:
    """The version of an artifact on disk, from its own descriptor."""
    ap = directory / ARTIFACT_FILENAME
    if not ap.is_file():
        raise ArtifactIncomplete(f"no {ARTIFACT_FILENAME} in {directory}")
    try:
        descriptor = json.loads(ap.read_text())
    except json.JSONDecodeError as exc:
        raise ArtifactIncomplete(f"{ARTIFACT_FILENAME} is not valid JSON") from exc
    version = descriptor.get("index_content_hash")
    if not version:
        raise ArtifactIncomplete(
            f"{ARTIFACT_FILENAME} has no index_content_hash to use as a version")
    return validate_version(version)


def iter_members(directory: Path) -> Iterable[tuple[str, Path]]:
    for rel in ARTIFACT_MEMBERS:
        yield rel, directory / rel


__all__ = [
    "ARTIFACT_MEMBERS", "ArtifactConflict", "ArtifactIncomplete",
    "ArtifactManifest", "ArtifactNotFound", "ArtifactStorage",
    "MANIFEST_FILENAME", "StorageError", "artifact_key", "iter_members",
    "read_artifact_version", "sha256_file", "validate_version",
]
