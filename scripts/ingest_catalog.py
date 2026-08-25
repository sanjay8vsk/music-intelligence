#!/usr/bin/env python
"""Ingest a directory of audio into a catalog and a fingerprint index.

Replaces the prototype's `src/audio_processing.py` batch loop. That script
derived identity with `file.split(".")[0]`, which collapses dotted filenames --
`mix.1.wav` and `mix.2.wav` both became `mix`, so ingesting both silently lost
one. Identity here comes from `Path.stem` (or the content hash) and a collision
is an error, not an overwrite.

Audio is never copied or moved; only its hash, duration and path are recorded.

    python scripts/ingest_catalog.py AUDIO_DIR --out catalog/main
    python scripts/ingest_catalog.py AUDIO_DIR --out catalog/main --id-mode content
    python scripts/ingest_catalog.py AUDIO_DIR --out catalog/main --verify
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
from musicintel.catalog.models import Catalog  # noqa: E402
from musicintel.recognition.fingerprint import FingerprintConfig  # noqa: E402
from musicintel.recognition.index import FingerprintIndex  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("audio_dir", help="directory to scan, recursively")
    ap.add_argument("--out", required=True,
                    help="output directory for catalog.json and index/")
    ap.add_argument("--cache", default=None,
                    help="fingerprint cache dir (default: <out>/cache)")
    ap.add_argument("--id-mode", default="stem", choices=("stem", "content"),
                    help="track identity: filename stem, or content hash prefix")
    ap.add_argument("--on-duplicate-id", default="error", choices=("error", "skip"))
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--verify", action="store_true",
                    help="re-hash every file after ingestion")
    args = ap.parse_args(argv)

    audio_dir = Path(args.audio_dir)
    out = Path(args.out)
    cache = None if args.no_cache else Path(args.cache or (out / "cache"))
    cfg = FingerprintConfig()

    try:
        paths = discover_audio(audio_dir)
    except IngestError as e:
        print(f"ERROR: {e}")
        return 2
    if not paths:
        print(f"ERROR: no audio found under {audio_dir}")
        return 2
    print(f"Discovered {len(paths)} audio files under {audio_dir}")

    try:
        report = ingest_paths(paths, root=audio_dir, config=cfg, id_mode=args.id_mode,
                              cache_dir=cache, on_duplicate_id=args.on_duplicate_id,
                              verbose=True)
    except IngestError as e:
        print(f"ERROR: {e}")
        print("  (pass --on-duplicate-id skip to keep the first and continue)")
        return 2

    cat = report.catalog
    problems = cat.verify(audio_dir, check_hashes=args.verify)
    if problems:
        print(f"ERROR: catalog failed verification ({len(problems)} problems)")
        for p in problems[:10]:
            print("   ", p)
        return 2

    index = build_catalog_index(cat, report.fingerprints, config=cfg)
    out.mkdir(parents=True, exist_ok=True)
    cat.save(out / "catalog.json")
    index.save(out / "index")

    dupes = cat.duplicate_content()
    print("\n" + "=" * 62)
    for k, v in report.summary().items():
        print(f"  {k:<18}{v}")
    print(f"  {'catalog hash':<18}{cat.content_hash()[:16]}...")
    print(f"  {'index postings':<18}{len(index):,}  ({index.nbytes/1e6:.1f} MB)")
    print(f"  {'index hash':<18}{index.content_hash()[:16]}...")
    if dupes:
        print(f"  {'duplicate audio':<18}{len(dupes)} hash(es) under >1 id:")
        for h, ids in list(dupes.items())[:5]:
            print(f"      {h[:12]}... -> {ids}")
    if report.skipped:
        print(f"  {'skipped':<18}{len(report.skipped)}")
        for s in report.skipped[:5]:
            print(f"      {Path(s['path']).name}: {s['reason'][:70]}")
    print("=" * 62)
    print(f"Wrote {out/'catalog.json'}")
    print(f"Wrote {out/'index'}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
