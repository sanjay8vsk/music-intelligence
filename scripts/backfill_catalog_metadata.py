#!/usr/bin/env python
"""Attach owner-supplied title/artist to an existing catalog, in place.

Why this exists: the evaluation corpus was fetched with title and artist
recorded in its manifest, but `ingest` had no way to carry them, so every one of
the 500 catalog entries has `title=None`. MusicBrainz enrichment needs artist and
title to look anything up, so this backfills them from the manifest that already
contains them.

WHAT IT WILL NOT DO
-------------------
Neither `Catalog.content_hash()` (over `(track_id, sha256)` pairs) nor
`FingerprintIndex.content_hash()` (over TrackEntry + config + arrays) reads
title or artist, so attaching them cannot change catalog identity, index
identity or recognition. This script does not take that on trust: it computes
both hashes before and after and **refuses to write if either moves**.

Only `catalog.json` is rewritten. The index is untouched, and `artifact.json`
stays valid because the hash it records is unchanged. Published object-storage
artifacts are NOT rewritten -- republishing metadata-only edits is refused by
design, and nothing here republishes.

    python scripts/backfill_catalog_metadata.py \
        --store data/eval/scale_store --catalog scale500 \
        --manifest eval/fixtures/scale_corpus_manifest.json --dry-run
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from musicintel.catalog.ingest import load_sidecar                     # noqa: E402
from musicintel.catalog.models import Catalog, CatalogTrack            # noqa: E402
from musicintel.catalog.store import CATALOG_FILENAME, CatalogStore    # noqa: E402


def apply_metadata(catalog: Catalog, sidecar) -> tuple[Catalog, int, int]:
    """Return (new catalog, attached, unchanged). Deterministic and order-preserving."""
    tracks: list[CatalogTrack] = []
    attached = unchanged = 0
    for t in catalog.tracks:
        meta = sidecar.lookup(t.track_id, t.sha256)
        title = (meta or {}).get("title") or t.title
        artist = (meta or {}).get("artist") or t.artist
        if title == t.title and artist == t.artist:
            unchanged += 1
            tracks.append(t)
            continue
        attached += 1
        tracks.append(dataclasses.replace(t, title=title, artist=artist))
    return (Catalog(tracks=tracks, version=catalog.version,
                    catalog_id=catalog.catalog_id), attached, unchanged)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--store", required=True, type=Path)
    ap.add_argument("--catalog", required=True)
    ap.add_argument("--manifest", required=True, type=Path,
                    help="read-only metadata source (e.g. a corpus manifest)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    store = CatalogStore(args.store)
    loaded = store.load(args.catalog, verify=True)
    catalog, index = loaded.catalog, loaded.index

    before_catalog = catalog.content_hash()
    before_index = index.content_hash()

    sidecar = load_sidecar(args.manifest)
    updated, attached, unchanged = apply_metadata(catalog, sidecar)

    after_catalog = updated.content_hash()
    after_index = before_index          # the index is never touched

    print(f"catalog          : {args.catalog}")
    print(f"sidecar entries  : {len(sidecar)} "
          f"({len(sidecar.by_sha256)} by sha256, {len(sidecar.by_track_id)} by track_id)")
    print(f"tracks           : {len(catalog)}")
    print(f"metadata attached: {attached}")
    print(f"already present  : {unchanged}")
    print(f"catalog_content_hash before: {before_catalog}")
    print(f"catalog_content_hash after : {after_catalog}")
    print(f"index_content_hash         : {before_index}")

    if after_catalog != before_catalog:
        print("\nREFUSING TO WRITE: catalog_content_hash changed. Metadata must "
              "never alter catalog identity.", file=sys.stderr)
        return 2

    if args.dry_run:
        print("\ndry run: nothing written")
        return 0
    if attached == 0:
        print("\nnothing to do")
        return 0

    updated.save(store.path_for(args.catalog) / CATALOG_FILENAME)

    # Prove the artifact still verifies end to end after the write.
    reloaded = store.load(args.catalog, verify=True)
    assert reloaded.catalog.content_hash() == before_catalog
    assert reloaded.index.content_hash() == before_index
    with_title = sum(1 for t in reloaded.catalog.tracks if t.title)
    with_artist = sum(1 for t in reloaded.catalog.tracks if t.artist)
    print(f"\nwritten. reloaded and verified: {with_title} titles, "
          f"{with_artist} artists, both content hashes unchanged")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
