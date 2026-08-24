#!/usr/bin/env python3
"""Materialize the degradation matrix as audio files, without running a benchmark.

Reads the fixture manifest, renders every planned query (clean, degraded and
negative) to data/eval/queries/, and writes index.jsonl carrying full
machine-readable condition metadata for each file.

The reference corpus is never modified: every transform reads the source and
writes a new file.

Usage:
    python scripts/make_degradations.py [--holdout 12] [--limit-tracks N]
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from musicintel.eval import degradation as dg  # noqa: E402
from musicintel.eval.manifest import Manifest  # noqa: E402
from musicintel.eval.recognition import (  # noqa: E402
    plan_heldout_negatives,
    plan_positive_queries,
    render_queries,
    synthesize_negatives,
    write_query_index,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", default="eval/fixtures/manifest.json")
    ap.add_argument("--out", default="data/eval/queries")
    ap.add_argument("--holdout", type=int, default=12)
    ap.add_argument("--limit-tracks", type=int, default=0)
    args = ap.parse_args()

    manifest_path = REPO_ROOT / args.manifest
    if not manifest_path.exists():
        print(f"ERROR: no manifest at {manifest_path}")
        print("Run: python scripts/fetch_fixture_corpus.py --tracks 50")
        return 2

    manifest = Manifest.load(manifest_path)
    problems = manifest.verify(REPO_ROOT)
    if problems:
        print(f"ERROR: corpus incomplete ({len(problems)} problems)")
        for p in problems[:10]:
            print("   ", p)
        return 2

    manifest.assign_holdout(args.holdout)
    catalog = manifest.catalog
    if args.limit_tracks:
        catalog = catalog[: args.limit_tracks]

    out_dir = REPO_ROOT / args.out
    print(f"Corpus     : {len(manifest)} tracks "
          f"(catalog {len(catalog)}, held out {len(manifest.held_out)})")
    print(f"Conditions : {len(dg.condition_matrix())}")
    print(f"Output     : {out_dir}\n")

    specs = plan_positive_queries(catalog) + plan_heldout_negatives(manifest.held_out)
    print(f"Planned {len(specs)} track-derived queries; rendering...")
    rendered = render_queries(specs, manifest, out_dir)

    print("Synthesizing speech / silence / noise negatives...")
    rendered += synthesize_negatives(out_dir)

    index_path = out_dir / "index.jsonl"
    write_query_index(rendered, index_path)

    fam = Counter(s.family for s, _ in rendered)
    total_bytes = sum(p.stat().st_size for _, p in rendered if p.exists())
    print(f"\nRendered {len(rendered)} query files ({total_bytes / 1e6:.1f} MB)")
    for k, v in sorted(fam.items()):
        print(f"  {k:10s} {v}")
    print(f"\nIndex -> {index_path}")
    print("Audio is git-ignored; only the index metadata describes it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
