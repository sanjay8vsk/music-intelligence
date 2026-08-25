#!/usr/bin/env python
"""Build the Phase 1G expanded negative set.

Phase 1E measured FAR against 63 held-out negatives, where one false accept was
worth 1.59 percentage points. This assembles a larger, leakage-screened negative
set so FAR becomes measurable nearer the 0.1% scale.

It does NOT touch the positive corpus, the 44-track source manifest, or the
original 126 negatives -- those are carried forward unchanged by the Phase 1G
benchmark so the comparison to Phase 1E survives.

Audio goes to data/eval/negative_queries/ (git-ignored). Only the negative-set
manifest -- identity, provenance, licensing, hashes, no audio -- is committed.

    python scripts/build_negative_set.py --raw data/eval/negatives_manifest_raw.json
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from musicintel.eval import degradation as dg  # noqa: E402
from musicintel.eval.manifest import Manifest  # noqa: E402
from musicintel.eval.negatives import (  # noqa: E402
    CAT_MUSIC, CAT_NEAR_SILENCE, CAT_PINK, CAT_SILENCE, CAT_SPEECH, CAT_WHITE,
    NegativeExcerpt, NegativeSet, NegativeSource, assign_splits,
    interleave_calibration, norm_text, plan_disjoint_excerpts, screen_candidates,
)
from musicintel.recognition.decision import DecisionConfig, decide  # noqa: E402
from musicintel.recognition.fingerprint import (  # noqa: E402
    FingerprintConfig, fingerprint, load_audio,
)
from musicintel.recognition.index import build_index  # noqa: E402
from musicintel.recognition.matcher import match  # noqa: E402

SR = dg.QUERY_SAMPLE_RATE
DURATIONS = (3.0, 5.0, 10.0)
# Extra speech lines for this phase. The harness's own six stay untouched.
SPEECH_LINES = [
    "Trains to the northern terminal are running twenty minutes behind schedule.",
    "Mix the dry ingredients thoroughly before adding the melted butter.",
    "The library closes at six on weekdays and at noon on Saturdays.",
    "Renewable generation exceeded demand for the first time last quarter.",
    "Take the second exit at the roundabout and follow signs for the harbour.",
    "Your appointment has been confirmed for Thursday the fourteenth at ten.",
    "Rainfall this month was well above the seasonal average across the region.",
    "Please ensure all electronic devices are switched to flight mode.",
    "The exhibition runs until the end of March and admission is free.",
    "Sort the recycling into paper, glass and household plastics.",
    "A moderate breeze from the south west is expected to continue overnight.",
    "The committee will publish its findings before the end of the year.",
]


def _say(text: str, out: Path) -> bool:
    say = shutil.which("say")
    if not say:
        return False
    try:
        subprocess.run([say, "-o", str(out), "--data-format=LEI16@22050", text],
                       check=True, capture_output=True, timeout=60)
        return True
    except Exception:  # noqa: BLE001
        return False


def synth_negatives(out_dir: Path, per_category: int) -> list[tuple[NegativeExcerpt, np.ndarray]]:
    """Silence, near-silence, pink/white noise and speech. Never a catalog track."""
    made = []
    for i in range(per_category):
        dur = DURATIONS[i % len(DURATIONS)]
        n = int(SR * dur)
        rng = np.random.default_rng(1000 + i)
        made.append((NegativeExcerpt(f"synth__{CAT_SILENCE}__{i:03d}", CAT_SILENCE,
                                     None, 0.0, dur), np.zeros(n, np.float32)))
        made.append((NegativeExcerpt(f"synth__{CAT_NEAR_SILENCE}__{i:03d}", CAT_NEAR_SILENCE,
                                     None, 0.0, dur),
                     (rng.standard_normal(n) * 1e-4).astype(np.float32)))
        made.append((NegativeExcerpt(f"synth__{CAT_WHITE}__{i:03d}", CAT_WHITE,
                                     None, 0.0, dur),
                     (rng.standard_normal(n) * 0.1).astype(np.float32)))
        pink = dg._pink_noise(n, np.random.default_rng(2000 + i))
        made.append((NegativeExcerpt(f"synth__{CAT_PINK}__{i:03d}", CAT_PINK,
                                     None, 0.0, dur), (pink * 0.1).astype(np.float32)))
    for i, line in enumerate(SPEECH_LINES):
        tmp = out_dir / f"_speech_src_{i}.wav"
        if not _say(line, tmp):
            continue
        import librosa
        y, _ = librosa.load(tmp, sr=SR, mono=True)
        tmp.unlink(missing_ok=True)
        if y.size < int(SR * 2.0):
            continue
        dur = min(DURATIONS[i % len(DURATIONS)], y.size / SR)
        made.append((NegativeExcerpt(f"synth__{CAT_SPEECH}__{i:03d}", CAT_SPEECH,
                                     None, 0.0, float(dur)),
                     y[: int(SR * dur)].astype(np.float32)))
    return made


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", default="eval/fixtures/manifest.json")
    ap.add_argument("--raw", default="data/eval/negatives_manifest_raw.json")
    ap.add_argument("--out-manifest", default="eval/fixtures/negatives_manifest.json")
    ap.add_argument("--query-dir", default="data/eval/negative_queries")
    ap.add_argument("--holdout", type=int, default=12)
    ap.add_argument("--synth-per-category", type=int, default=18)
    args = ap.parse_args(argv)

    corpus = Manifest.load(REPO_ROOT / args.manifest)
    corpus.assign_holdout(args.holdout)
    catalog = {t.track_id for t in corpus.catalog}
    heldout = corpus.held_out
    print(f"Corpus: {len(corpus)} tracks | catalog {len(catalog)} | held out {len(heldout)}")

    # -- 1. candidate sources from the freshly fetched corpus --------------
    raw_path = REPO_ROOT / args.raw
    candidates: list[NegativeSource] = []
    if raw_path.exists():
        raw = Manifest.load(raw_path)
        raw_meta = {t["track_id"]: t for t in json.loads(raw_path.read_text())["tracks"]}
        for t in raw.tracks:
            md = raw_meta.get(t.track_id, {})
            candidates.append(NegativeSource(
                track_id=t.track_id, path=t.path, sha256=t.sha256,
                duration_sec=t.duration_sec, license=t.license,
                license_url=t.license_url, source=t.source,
                source_url=md.get("source_url"), artist=md.get("artist"),
                title=md.get("title"), origin="fetched"))
    print(f"Fetched candidates: {len(candidates)}")

    # -- 2. metadata leakage screen ----------------------------------------
    kept, rejected = screen_candidates(
        candidates,
        catalog_ids=catalog,
        catalog_sha256={t.sha256 for t in corpus.tracks},
        catalog_artist_title={(norm_text(t.artist), norm_text(t.title))
                              for t in corpus.catalog},
        catalog_artists={norm_text(t.artist) for t in corpus.catalog if t.artist},
        corpus_ids={t.track_id for t in corpus.tracks},
    )
    print(f"  metadata screen: kept {len(kept)}, rejected {len(rejected)}")
    reasons: dict[str, int] = {}
    for r in rejected:
        reasons[r["reason"]] = reasons.get(r["reason"], 0) + 1
    for k, v in sorted(reasons.items()):
        print(f"     {v:>3}  {k}")

    # -- 3. content leakage gate: run them through the recognizer ----------
    # A re-encode or a retitled upload defeats every metadata gate. The only
    # way to be sure a candidate carries no indexed audio is to ask the
    # recognizer, so anything the catalog index MATCHES is thrown out.
    fp_cfg = FingerprintConfig()
    print(f"  building catalog index over {len(catalog)} tracks for the content gate...")
    t0 = time.perf_counter()
    idx = build_index([(t.track_id, fingerprint(*load_audio(REPO_ROOT / t.path, fp_cfg), fp_cfg))
                       for t in corpus.catalog], config=fp_cfg)
    print(f"    {len(idx):,} postings in {time.perf_counter()-t0:.0f}s")
    dcfg = DecisionConfig(threshold=0.018087855297157620, min_aligned_landmarks=5)
    content_rejected = []
    survivors: list[NegativeSource] = []
    for i, s in enumerate(kept, 1):
        try:
            y, sr = load_audio(REPO_ROOT / s.path, fp_cfg)
            d = decide(match(fingerprint(y, sr, fp_cfg), idx), config=dcfg)
        except Exception as e:  # noqa: BLE001
            content_rejected.append({"track_id": s.track_id, "reason": f"decode failed: {e}"})
            continue
        if d.is_match:
            content_rejected.append({"track_id": s.track_id,
                                     "reason": f"recognizer MATCHED catalog track {d.track_id}"})
            continue
        survivors.append(s)
        if i % 10 == 0:
            print(f"    content gate {i}/{len(kept)}")
    print(f"  content gate: kept {len(survivors)}, rejected {len(content_rejected)}")
    for r in content_rejected:
        print(f"     {r['track_id'][:40]}: {r['reason']}")

    # -- 4. held-out tracks are sources too (already Phase 1E negatives) ----
    sources = survivors + [
        NegativeSource(track_id=t.track_id, path=t.path, sha256=t.sha256,
                       duration_sec=t.duration_sec, license=t.license,
                       license_url=t.license_url, source=t.source,
                       source_url=t.source_url, artist=t.artist, title=t.title,
                       origin="heldout")
        for t in heldout
    ]
    print(f"Negative sources: {len(sources)} "
          f"({len(survivors)} fetched + {len(heldout)} held out), "
          f"{sum(s.duration_sec for s in sources)/60:.1f} min")

    # -- 5. plan disjoint excerpts + synthetics -----------------------------
    excerpts: list[NegativeExcerpt] = []
    for s in sources:
        excerpts += plan_disjoint_excerpts(s, DURATIONS)
    print(f"Music excerpts (disjoint): {len(excerpts)}")

    qdir = REPO_ROOT / args.query_dir
    qdir.mkdir(parents=True, exist_ok=True)
    synth = synth_negatives(qdir, args.synth_per_category)
    excerpts += [e for e, _ in synth]
    print(f"Synthetic negatives: {len(synth)}")

    # -- 6. split by SOURCE TRACK, preserving the Phase 1E held-out sides ----
    cal_sources = (interleave_calibration([t.track_id for t in heldout])
                   | interleave_calibration([s.track_id for s in survivors]))
    excerpts = assign_splits(excerpts, calibration_sources=cal_sources)

    # -- 7. render -----------------------------------------------------------
    print("Rendering...")
    by_src: dict[str, list[NegativeExcerpt]] = {}
    for e in excerpts:
        if e.source_track:
            by_src.setdefault(e.source_track, []).append(e)
    src_by_id = {s.track_id: s for s in sources}
    rendered: dict[str, str] = {}
    for i, (tid, group) in enumerate(sorted(by_src.items()), 1):
        try:
            y, sr = dg.load_source(REPO_ROOT / src_by_id[tid].path, SR)
        except Exception as e:  # noqa: BLE001
            print(f"    decode failed {tid}: {e}")
            continue
        for e in group:
            a = int(round(e.start_sec * sr)); b = a + int(round(e.duration * sr))
            if b > len(y):
                continue
            p = qdir / f"{e.query_id}.wav"
            sf.write(p, y[a:b].astype(np.float32), sr, subtype="PCM_16")
            rendered[e.query_id] = str(p.relative_to(REPO_ROOT))
        if i % 10 == 0:
            print(f"    {i}/{len(by_src)} sources")
    for e, audio in synth:
        p = qdir / f"{e.query_id}.wav"
        sf.write(p, audio, SR, subtype="PCM_16")
        rendered[e.query_id] = str(p.relative_to(REPO_ROOT))

    excerpts = [NegativeExcerpt(**{**e.to_dict(), "rendered_path": rendered.get(e.query_id)})
                for e in excerpts if e.query_id in rendered]

    ns = NegativeSet(sources=sources, excerpts=excerpts)
    problems = ns.verify(catalog)
    if problems:
        print(f"ERROR: negative set failed verification ({len(problems)} problems)")
        for p in problems[:10]:
            print("   ", p)
        return 2
    ns.save(REPO_ROOT / args.out_manifest)

    print("\n" + "=" * 62)
    print(f"  negative excerpts : {len(ns.excerpts)}")
    print(f"  by category       : {ns.counts_by_category()}")
    print(f"  by split          : {ns.counts_by_split()}")
    print(f"  source recordings : {ns.source_counts()}")
    print(f"  content hash      : {ns.content_hash()[:16]}...")
    print("=" * 62)
    print(f"Wrote {args.out_manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
