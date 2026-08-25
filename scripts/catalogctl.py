#!/usr/bin/env python
"""Manage multi-tenant catalogs and identify audio against them.

    python scripts/catalogctl.py add      STORE CATALOG_ID AUDIO_DIR
    python scripts/catalogctl.py list     STORE
    python scripts/catalogctl.py describe STORE CATALOG_ID
    python scripts/catalogctl.py identify STORE CATALOG_ID AUDIO_FILE

Each catalog gets its own index artifact under STORE/CATALOG_ID/, so a query
against one catalog physically cannot see another's postings. Audio is never
copied; only its hash, duration and path are recorded.

`scripts/ingest_catalog.py` remains the single-catalog tool; this is the
multi-tenant front end over the same ingestion code.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from musicintel.catalog.ingest import (  # noqa: E402
    IngestError, build_catalog_index, discover_audio, ingest_paths,
)
from musicintel.catalog.store import CatalogStore, CatalogStoreError  # noqa: E402
from musicintel.recognition.fingerprint import FingerprintConfig  # noqa: E402
from musicintel.service.recognition import RecognitionService  # noqa: E402


def cmd_add(args) -> int:
    store = CatalogStore(args.store)
    audio_dir = Path(args.audio_dir)
    cfg = FingerprintConfig()
    try:
        paths = discover_audio(audio_dir)
    except IngestError as e:
        print(f"ERROR: {e}"); return 2
    if not paths:
        print(f"ERROR: no audio under {audio_dir}"); return 2
    print(f"Discovered {len(paths)} audio files under {audio_dir}")
    try:
        rep = ingest_paths(paths, root=audio_dir, config=cfg,
                           id_mode=args.id_mode,
                           cache_dir=(None if args.no_cache
                                      else Path(args.store) / "_fpcache"),
                           on_duplicate_id=args.on_duplicate_id, verbose=True)
    except IngestError as e:
        print(f"ERROR: {e}\n  (pass --on-duplicate-id skip to continue)"); return 2

    problems = rep.catalog.verify(audio_dir, check_hashes=args.verify)
    if problems:
        print(f"ERROR: catalog failed verification ({len(problems)} problems)")
        for p in problems[:10]:
            print("   ", p)
        return 2
    index = build_catalog_index(rep.catalog, rep.fingerprints, config=cfg)
    try:
        out = store.save(rep.catalog, index, catalog_id=args.catalog_id)
    except CatalogStoreError as e:
        print(f"ERROR: {e}"); return 2

    print("\n" + "=" * 60)
    for k, v in rep.summary().items():
        print(f"  {k:<18}{v}")
    print(f"  {'catalog_id':<18}{args.catalog_id}")
    print(f"  {'catalog hash':<18}{rep.catalog.content_hash()[:16]}...")
    print(f"  {'index postings':<18}{len(index):,} ({index.nbytes/1e6:.1f} MB)")
    print(f"  {'index hash':<18}{index.content_hash()[:16]}...")
    dupes = rep.catalog.duplicate_content()
    if dupes:
        print(f"  {'duplicate audio':<18}{len(dupes)} hash(es) under >1 id")
    if rep.skipped:
        print(f"  {'skipped':<18}{len(rep.skipped)}")
        for s in rep.skipped[:5]:
            print(f"      {Path(s['path']).name}: {s['reason'][:66]}")
    print("=" * 60)
    print(f"Wrote {out}")
    return 0


def cmd_list(args) -> int:
    store = CatalogStore(args.store)
    ids = store.list_catalogs()
    if not ids:
        print(f"no catalogs under {args.store}"); return 0
    print(f"{'catalog_id':<24}{'tracks':>8}{'fingerprints':>14}{'index hash':>18}")
    for cid in ids:
        a = store.describe(cid)
        print(f"{cid:<24}{a['track_count']:>8}{a['fingerprint_count']:>14,}"
              f"{a['index_content_hash'][:16]:>18}")
    return 0


def cmd_describe(args) -> int:
    try:
        print(json.dumps(CatalogStore(args.store).describe(args.catalog_id), indent=2))
    except CatalogStoreError as e:
        print(f"ERROR: {e}"); return 2
    return 0


def cmd_identify(args) -> int:
    try:
        svc = RecognitionService(CatalogStore(args.store))
        r = svc.identify_file(args.audio_file, args.catalog_id)
    except CatalogStoreError as e:
        print(f"ERROR: {e}"); return 2
    print(json.dumps(r.to_dict(), indent=2))
    return 0 if r.is_match else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("add", help="ingest a directory as a catalog")
    a.add_argument("store"); a.add_argument("catalog_id"); a.add_argument("audio_dir")
    a.add_argument("--id-mode", default="stem", choices=("stem", "content"))
    a.add_argument("--on-duplicate-id", default="error", choices=("error", "skip"))
    a.add_argument("--no-cache", action="store_true")
    a.add_argument("--verify", action="store_true")
    a.set_defaults(fn=cmd_add)

    l = sub.add_parser("list", help="list catalogs in a store")
    l.add_argument("store"); l.set_defaults(fn=cmd_list)

    d = sub.add_parser("describe", help="show one catalog's artifact metadata")
    d.add_argument("store"); d.add_argument("catalog_id"); d.set_defaults(fn=cmd_describe)

    i = sub.add_parser("identify", help="identify audio against one catalog")
    i.add_argument("store"); i.add_argument("catalog_id"); i.add_argument("audio_file")
    i.set_defaults(fn=cmd_identify)

    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
