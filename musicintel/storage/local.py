"""Filesystem backend for artifact storage.

Not a stand-in for object storage -- it is the reference implementation of the
same four operations, and it is what the tests exercise. An S3-compatible
backend replaces `_write_member`/`_read_member`/`_list` and nothing else;
callers, key layout, manifests and the sync logic are unchanged.

Writes are staged and renamed. A `put` that dies halfway leaves a temporary
directory, never a half-populated key that a later `get` would treat as
complete.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

from musicintel.catalog.store import validate_catalog_id
from musicintel.storage.artifacts import (
    ARTIFACT_MEMBERS,
    MANIFEST_FILENAME,
    ArtifactConflict,
    ArtifactIncomplete,
    ArtifactManifest,
    ArtifactNotFound,
    artifact_key,
    validate_version,
)


class LocalArtifactStorage:
    """Content-addressed artifacts under one root directory."""

    scheme = "file"

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    # -- layout ----------------------------------------------------------
    def _dir_for(self, catalog_id: str, version: str) -> Path:
        return self.root / artifact_key(catalog_id, version)

    def describe(self) -> str:
        return f"file://{self.root}"

    # -- write -----------------------------------------------------------
    def put_artifact(self, catalog_id: str, version: str, source: Path) -> str:
        """Publish an artifact. Idempotent for identical content.

        Same key and identical bytes is a no-op. Same key and different bytes
        raises `ArtifactConflict` -- a content-addressed store that overwrote
        would make every previously fetched copy unreproducible.
        """
        source = Path(source)
        version = validate_version(version)
        catalog_id = validate_catalog_id(catalog_id)
        manifest = ArtifactManifest.build(source, catalog_id, version)
        target = self._dir_for(catalog_id, version)

        if target.is_dir() and (target / MANIFEST_FILENAME).is_file():
            existing = ArtifactManifest.from_json(
                (target / MANIFEST_FILENAME).read_text())
            if existing.differs_from(manifest):
                changed = sorted(
                    rel for rel in set(existing.members) | set(manifest.members)
                    if existing.members.get(rel) != manifest.members.get(rel))
                raise ArtifactConflict(
                    f"{artifact_key(catalog_id, version)} already holds different "
                    f"content; differing members: {changed}. A content-addressed "
                    "key is never overwritten -- rebuild the index so the version "
                    "changes, or publish under the correct version.")
            return artifact_key(catalog_id, version)      # idempotent no-op

        target.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=f".put-{version[:12]}-",
                                        dir=target.parent))
        try:
            for rel in ARTIFACT_MEMBERS:
                dest = staging / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source / rel, dest)
            (staging / MANIFEST_FILENAME).write_text(manifest.to_json())
            problems = manifest.verify_directory(staging)
            if problems:                                   # pragma: no cover
                raise ArtifactIncomplete(
                    f"staged copy does not match its manifest: {problems}")
            try:
                os.rename(staging, target)
            except OSError:
                # Another publisher won the race with identical content; that is
                # exactly the idempotent case, so verify rather than fail.
                if not target.is_dir():
                    raise
                existing = ArtifactManifest.from_json(
                    (target / MANIFEST_FILENAME).read_text())
                if existing.differs_from(manifest):
                    raise ArtifactConflict(
                        f"{artifact_key(catalog_id, version)} was concurrently "
                        "published with different content")
                shutil.rmtree(staging, ignore_errors=True)
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise
        return artifact_key(catalog_id, version)

    # -- read ------------------------------------------------------------
    def get_artifact(self, catalog_id: str, version: str,
                     destination: Path) -> ArtifactManifest:
        """Copy an artifact into `destination`, checking it against its manifest.

        `destination` is a staging directory chosen by the caller; this method
        never touches a live catalog root.
        """
        src = self._dir_for(catalog_id, version)
        mf = src / MANIFEST_FILENAME
        if not src.is_dir() or not mf.is_file():
            raise ArtifactNotFound(
                f"no artifact {artifact_key(catalog_id, version)} under {self.root}")
        manifest = ArtifactManifest.from_json(mf.read_text())

        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=True)
        for rel in ARTIFACT_MEMBERS:
            member = src / rel
            if not member.is_file():
                raise ArtifactIncomplete(
                    f"stored artifact {artifact_key(catalog_id, version)} is "
                    f"missing {rel}")
            dest = destination / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(member, dest)

        problems = manifest.verify_directory(destination)
        if problems:
            raise ArtifactIncomplete(
                f"fetched artifact {artifact_key(catalog_id, version)} does not "
                f"match its manifest: {problems}")
        return manifest

    def exists(self, catalog_id: str, version: str) -> bool:
        d = self._dir_for(catalog_id, version)
        return d.is_dir() and (d / MANIFEST_FILENAME).is_file()

    def list_versions(self, catalog_id: str) -> list[str]:
        d = self.root / "catalogs" / validate_catalog_id(catalog_id)
        if not d.is_dir():
            return []
        return sorted(
            p.name for p in d.iterdir()
            if p.is_dir() and (p / MANIFEST_FILENAME).is_file()
            and len(p.name) == 64)

    def list_catalogs(self) -> list[str]:
        d = self.root / "catalogs"
        if not d.is_dir():
            return []
        out = []
        for p in sorted(d.iterdir()):
            if not p.is_dir():
                continue
            try:
                validate_catalog_id(p.name)
            except Exception:
                continue
            if self.list_versions(p.name):
                out.append(p.name)
        return out


def storage_from_url(url: str) -> LocalArtifactStorage:
    """Build a backend from a URL. Only `file://` is implemented.

    The URL form exists so configuration does not change when an S3-compatible
    backend is added; no provider has been chosen, and no SDK is imported.
    """
    if url.startswith("file://"):
        return LocalArtifactStorage(url[len("file://"):])
    if "://" not in url:
        return LocalArtifactStorage(url)
    scheme = url.split("://", 1)[0]
    raise NotImplementedError(
        f"artifact storage scheme {scheme!r} is not implemented; only 'file://' "
        "is available. No object-storage provider has been selected.")


__all__ = ["LocalArtifactStorage", "storage_from_url"]
