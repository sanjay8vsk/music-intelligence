#!/usr/bin/env python
"""Screen a freshly fetched corpus into the Stage 2 scale-validation catalog.

`scripts/fetch_fixture_corpus.py` verifies licences per item but performs NO
content deduplication -- it will happily return the same audio under two
identifiers, and because its search is deterministically ordered from page 1 it
re-fetches everything the Phase 0/1 corpora already hold.

This screens the raw fetch by SHA-256 against:
  * eval/fixtures/manifest.json          (the frozen 44-track evaluation corpus)
  * eval/fixtures/negatives_manifest.json (the frozen negative source corpus)
  * itself                                (two identifiers, one recording)

and writes a separate manifest. Neither frozen manifest is read for anything
but comparison, and neither is written.

Every accepted, rejected and failed candidate is recorded, so the corpus can be
explained rather than merely counted.

    python scripts/build_scale_corpus.py --raw data/eval/corpus500_raw.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from musicintel.eval.manifest import Manifest  # noqa: E402
from musicintel.eval.negatives import NegativeSet  # noqa: E402

FROZEN_MANIFEST = "eval/fixtures/manifest.json"
FROZEN_NEGATIVES = "eval/fixtures/negatives_manifest.json"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", default="data/eval/corpus500_raw.json")
    ap.add_argument("--out", default="eval/fixtures/scale_corpus_manifest.json")
    ap.add_argument("--rejects", default="data/eval/corpus500_rejects.json")
    ap.add_argument("--target", type=int, default=500)
    args = ap.parse_args(argv)

    raw_path = REPO_ROOT / args.raw
    if not raw_path.is_file():
        print(f"ERROR: no raw manifest at {raw_path}")
        return 2
    raw = Manifest.load(raw_path)
    print(f"Raw fetch: {len(raw)} tracks")

    # -- what the frozen corpora already hold -----------------------------
    frozen = Manifest.load(REPO_ROOT / FROZEN_MANIFEST)
    negs = NegativeSet.load(REPO_ROOT / FROZEN_NEGATIVES)
    frozen_sha = {t.sha256 for t in frozen.tracks} | {s.sha256 for s in negs.sources}
    frozen_ids = {t.track_id for t in frozen.tracks} | {s.track_id for s in negs.sources}
    print(f"Screening against {len(frozen_sha)} existing recordings "
          f"({len(frozen)} eval corpus + {len(negs.sources)} negative sources)")

    accepted, rejected = [], []
    seen_sha: dict[str, str] = {}
    seen_id: set[str] = set()
    for t in raw.tracks:
        reason = None
        if not t.sha256 or len(t.sha256) != 64:
            reason = "missing or malformed sha256"
        elif t.duration_sec <= 0:
            reason = "non-positive duration"
        elif not (REPO_ROOT / t.path).is_file():
            reason = "audio file missing on disk (download failed)"
        elif t.sha256 in frozen_sha:
            reason = "sha256 duplicate of a frozen-corpus recording"
        elif t.track_id in frozen_ids:
            reason = "track_id already used by a frozen-corpus recording"
        elif t.sha256 in seen_sha:
            reason = f"sha256 duplicate of accepted candidate {seen_sha[t.sha256]}"
        elif t.track_id in seen_id:
            reason = "duplicate track_id within the fetch"
        if reason:
            rejected.append({"track_id": t.track_id, "sha256": t.sha256[:16],
                             "reason": reason})
            continue
        seen_sha[t.sha256] = t.track_id
        seen_id.add(t.track_id)
        accepted.append(t)

    reasons = Counter(r["reason"] for r in rejected)
    print(f"\nAccepted {len(accepted)} | rejected {len(rejected)}")
    for k, v in reasons.most_common():
        print(f"   {v:>4}  {k}")

    (REPO_ROOT / args.rejects).parent.mkdir(parents=True, exist_ok=True)
    (REPO_ROOT / args.rejects).write_text(json.dumps(
        {"raw_count": len(raw), "accepted": len(accepted),
         "rejected": len(rejected), "reasons": dict(reasons),
         "rejected_detail": rejected}, indent=2) + "\n")

    if len(accepted) < args.target:
        print(f"\nSHORTFALL: {len(accepted)} distinct recordings, target {args.target}.")
        print("  Not writing a manifest. Re-run the fetch with a larger --tracks;")
        print("  downloads already on disk are reused, so this is cheap to extend.")
        print(f"  Reject detail: {args.rejects}")
        return 1

    # Deterministic: keep the first `target` by sorted track_id, so the corpus
    # is a function of the fetch, not of thread completion order.
    accepted = sorted(accepted, key=lambda t: t.track_id)[: args.target]
    out = Manifest(tracks=accepted)
    out.save(REPO_ROOT / args.out)

    durations = sorted(t.duration_sec for t in accepted)
    total = sum(durations)
    print(f"\n{'=' * 62}")
    print(f"  distinct recordings   {len(out)}")
    print(f"  licences              {out.license_counts()}")
    print(f"  distinct artists      {len({t.artist for t in accepted})}")
    print(f"  total audio           {total/3600:.2f} h ({total/60:.0f} min)")
    print(f"  median duration       {durations[len(durations)//2]:.0f} s")
    print(f"  audio on disk         {sum(t.bytes for t in accepted)/1e9:.2f} GB")
    print(f"  manifest content hash {out.content_hash()}")
    print(f"{'=' * 62}")
    print(f"Wrote {args.out}")
    print(f"Wrote {args.rejects}  (git-ignored)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
