#!/usr/bin/env python
"""Publish catalog artifacts from a CatalogStore into artifact storage.

The version is the artifact's own `index_content_hash` -- nothing is numbered,
nothing is tagged, and republishing identical bytes is a no-op.

    python scripts/publish_artifact.py --store data/catalogs \
        --storage-url file:///var/lib/musicintel/artifacts --catalog acme

    python scripts/publish_artifact.py --store data/catalogs \
        --storage-url file:///srv/artifacts --all --dry-run
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from musicintel.catalog.store import CatalogStore                      # noqa: E402
from musicintel.storage.artifacts import (                             # noqa: E402
    ArtifactConflict, StorageError, read_artifact_version,
)
from musicintel.storage.local import storage_from_url                  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--store", required=True, type=Path,
                    help="CatalogStore root holding built artifacts")
    ap.add_argument("--storage-url",
                    default=os.environ.get("MUSICINTEL_ARTIFACT_STORAGE_URL"),
                    help="artifact storage URL (only file:// is implemented)")
    ap.add_argument("--catalog", action="append", dest="catalogs",
                    help="catalog to publish (repeatable)")
    ap.add_argument("--all", action="store_true", help="publish every catalog")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    if not args.storage_url and not args.dry_run:
        ap.error("--storage-url or MUSICINTEL_ARTIFACT_STORAGE_URL is required")
    if not args.catalogs and not args.all:
        ap.error("pass --catalog or --all")

    store = CatalogStore(args.store)
    catalog_ids = args.catalogs or store.list_catalogs()
    if not catalog_ids:
        print(f"no catalogs under {args.store}")
        return 0

    storage = storage_from_url(args.storage_url) if args.storage_url else None
    published = skipped = 0
    for cid in catalog_ids:
        directory = store.path_for(cid)
        try:
            version = read_artifact_version(directory)
        except StorageError as exc:
            print(f"{cid}: {exc}", file=sys.stderr)
            return 1
        if args.dry_run:
            print(f"[dry-run] {cid} -> catalogs/{cid}/{version[:12]}…")
            continue
        try:
            already = storage.exists(cid, version)
            key = storage.put_artifact(cid, version, directory)
        except ArtifactConflict as exc:
            print(f"{cid}: CONFLICT {exc}", file=sys.stderr)
            return 2
        except StorageError as exc:
            print(f"{cid}: {exc}", file=sys.stderr)
            return 1
        if already:
            print(f"{cid}: already published at {key} (no-op)")
            skipped += 1
        else:
            print(f"{cid}: published {key}")
            published += 1

    if not args.dry_run:
        print(f"{published} published, {skipped} already present")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
