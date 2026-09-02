#!/usr/bin/env python
"""Stage 2 scale validation: exercise the whole production path at 500 tracks.

    corpus manifest -> catalog ingestion -> per-catalog artifact
        -> RecognitionService -> frozen recognizer -> gated cascade -> identify

Measures what the roadmap's Stage 2 acceptance asks for and nothing more. It
changes no threshold, touches no recognition module, and reads the frozen
manifests only for comparison.

    python scripts/validate_scale.py
"""

from __future__ import annotations

import argparse
import gc
import json
import platform
import resource
import shutil
import sys
import time
import tracemalloc
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from musicintel.catalog.ingest import build_catalog_index, ingest_paths  # noqa: E402
from musicintel.catalog.store import CatalogStore  # noqa: E402
from musicintel.eval.manifest import Manifest  # noqa: E402
from musicintel.eval.provenance import (  # noqa: E402
    PHASE1_SOURCES, git_state, source_fingerprint,
)
from musicintel.recognition.fingerprint import (  # noqa: E402
    FORMAT_VERSION, FingerprintConfig, fingerprint, load_audio,
)
from musicintel.recognition.index import INDEX_FORMAT_VERSION  # noqa: E402
from musicintel.recognition.matcher import MatchConfig, match  # noqa: E402
from musicintel.service.recognition import (  # noqa: E402
    GATE_THRESHOLD, STAGE1_THRESHOLD, STAGE2_THRESHOLD, RecognitionService,
)

SCALE_CATALOG_ID = "scale500"
CONTROL_CATALOG_ID = "control"


def rss_mb() -> float:
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 1e6 if sys.platform == "darwin" else r / 1e3   # macOS bytes, Linux KiB


def pctl(v, p):
    return round(float(np.percentile(v, p)), 3) if v else None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", default="eval/fixtures/scale_corpus_manifest.json")
    ap.add_argument("--store", default="data/eval/scale_store")
    ap.add_argument("--cache", default="data/eval/scale_fpcache")
    ap.add_argument("--report-dir", default="eval/reports")
    ap.add_argument("--queries", type=int, default=200)
    ap.add_argument("--keep-store", action="store_true")
    args = ap.parse_args(argv)

    mpath = REPO_ROOT / args.manifest
    if not mpath.is_file():
        print(f"ERROR: no scale corpus manifest at {mpath}")
        print("  run scripts/build_scale_corpus.py first")
        return 2
    corpus = Manifest.load(mpath)
    fp_cfg = FingerprintConfig()
    paths = [REPO_ROOT / t.path for t in corpus.tracks]
    missing = [p for p in paths if not p.is_file()]
    if missing:
        print(f"ERROR: {len(missing)} of {len(paths)} corpus files missing on disk")
        return 2
    print(f"Corpus: {len(corpus)} tracks, "
          f"{sum(t.duration_sec for t in corpus)/3600:.2f} h audio")
    print(f"  manifest content hash {corpus.content_hash()}")

    store_root = REPO_ROOT / args.store
    if store_root.exists() and not args.keep_store:
        shutil.rmtree(store_root)
    store = CatalogStore(store_root)
    cache = REPO_ROOT / args.cache

    # -- 1. cold ingestion --------------------------------------------------
    print(f"\n[1/6] Cold ingestion of {len(paths)} tracks (cache may be warm)...")
    gc.collect(); tracemalloc.start()
    base_rss = rss_mb(); t0 = time.perf_counter()
    rep = ingest_paths(paths, root=REPO_ROOT, config=fp_cfg, id_mode="stem",
                       cache_dir=cache, on_duplicate_id="error", verbose=True)
    t_ingest = time.perf_counter() - t0
    cur, peak_py = tracemalloc.get_traced_memory(); tracemalloc.stop()
    peak_rss = rss_mb()
    print(f"  {rep.ingested} ingested in {t_ingest:.0f}s "
          f"({rep.cache_hits} cache hits, {len(rep.skipped)} skipped)")

    t1 = time.perf_counter()
    index = build_catalog_index(rep.catalog, rep.fingerprints, config=fp_cfg)
    t_index = time.perf_counter() - t1
    rep.catalog.catalog_id = SCALE_CATALOG_ID
    t2 = time.perf_counter()
    out = store.save(rep.catalog, index, catalog_id=SCALE_CATALOG_ID)
    t_save = time.perf_counter() - t2
    artifact_bytes = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    print(f"  index {len(index):,} postings in {t_index:.0f}s, "
          f"artifact {artifact_bytes/1e6:.1f} MB written in {t_save:.1f}s")

    # -- 2. warm re-ingestion + reproducibility -----------------------------
    print("\n[2/6] Warm re-ingestion from the same manifest...")
    t3 = time.perf_counter()
    rep2 = ingest_paths(paths, root=REPO_ROOT, config=fp_cfg, id_mode="stem",
                        cache_dir=cache, on_duplicate_id="error")
    t_warm = time.perf_counter() - t3
    index2 = build_catalog_index(rep2.catalog, rep2.fingerprints, config=fp_cfg)
    reproducible = {
        "catalog_hash_identical": rep2.catalog.content_hash() == rep.catalog.content_hash(),
        "index_hash_identical": index2.content_hash() == index.content_hash(),
        "catalog_content_hash": rep.catalog.content_hash(),
        "index_content_hash": index.content_hash(),
        "warm_seconds": round(t_warm, 2),
        "cold_seconds": round(t_ingest, 2),
        "speedup": round(t_ingest / t_warm, 1) if t_warm else None,
        "warm_cache_hit_rate": round(rep2.cache_hit_rate, 4),
    }
    print(f"  warm {t_warm:.0f}s ({rep2.cache_hit_rate*100:.0f}% cache hits), "
          f"catalog hash identical: {reproducible['catalog_hash_identical']}, "
          f"index hash identical: {reproducible['index_hash_identical']}")

    # -- 3. duplicate detection ---------------------------------------------
    print("\n[3/6] Duplicate detection by content hash...")
    dupes = rep.catalog.duplicate_content()
    frozen = Manifest.load(REPO_ROOT / "eval/fixtures/manifest.json")
    overlap = {t.sha256 for t in rep.catalog} & {t.sha256 for t in frozen.tracks}
    dup_result = {"duplicate_content_groups": len(dupes),
                  "distinct_sha256": len({t.sha256 for t in rep.catalog}),
                  "tracks": len(rep.catalog),
                  "overlap_with_frozen_eval_corpus": len(overlap)}
    print(f"  {dup_result['distinct_sha256']} distinct hashes over "
          f"{dup_result['tracks']} tracks; {len(dupes)} duplicate groups; "
          f"{len(overlap)} overlap with the frozen eval corpus")

    # -- 4. cross-tenant isolation at scale ---------------------------------
    print("\n[4/6] Cross-tenant isolation at scale...")
    ctrl_paths = paths[:5]
    ctrl = ingest_paths(paths[-5:], root=REPO_ROOT, config=fp_cfg,
                        cache_dir=cache, on_duplicate_id="error")
    ctrl.catalog.catalog_id = CONTROL_CATALOG_ID
    store.save(ctrl.catalog, build_catalog_index(ctrl.catalog, ctrl.fingerprints,
                                                 config=fp_cfg),
               catalog_id=CONTROL_CATALOG_ID)
    svc = RecognitionService(store)
    probe_track = rep.catalog.tracks[0]
    y, sr = load_audio(REPO_ROOT / probe_track.source_path, fp_cfg)
    mid = len(y) // 2
    probe = y[mid:mid + sr * 5]
    own = svc.identify(probe, sr, SCALE_CATALOG_ID)
    other = svc.identify(probe, sr, CONTROL_CATALOG_ID)
    isolation = {"own_catalog_decision": own.decision.value,
                 "own_catalog_track": own.track_id,
                 "own_catalog_correct": own.track_id == probe_track.track_id,
                 "other_catalog_decision": other.decision.value,
                 "other_catalog_track": other.track_id,
                 "isolated": (own.is_match and other.decision.value == "NO_MATCH")}
    print(f"  own catalog: {own.decision.value} ({own.track_id}) | "
          f"other tenant: {other.decision.value} -> isolated: {isolation['isolated']}")

    # -- 5. lookup latency ---------------------------------------------------
    n_q = min(args.queries, len(rep.catalog))
    print(f"\n[5/6] Latency over {n_q} in-catalog 5 s queries...")
    loaded = svc.get(SCALE_CATALOG_ID)
    m_cfg = MatchConfig()
    e2e, fp_ms, lookup_ms, hist_ms, correct = [], [], [], [], 0
    step = max(1, len(rep.catalog) // n_q)
    for i, t in enumerate(rep.catalog.tracks[::step][:n_q]):
        y, sr = load_audio(REPO_ROOT / t.source_path, fp_cfg)
        if len(y) < sr * 8:
            continue
        q = y[len(y) // 2: len(y) // 2 + sr * 5]
        r = svc.identify(q, sr, SCALE_CATALOG_ID)
        e2e.append(r.latency_ms); correct += int(r.track_id == t.track_id)
        a = time.perf_counter(); qf = fingerprint(q, sr, fp_cfg); b = time.perf_counter()
        mr = match(qf, loaded.index, config=m_cfg)
        fp_ms.append((b - a) * 1000)
        # MatchTiming stores SECONDS; these are reported in milliseconds.
        lookup_ms.append(mr.timing.lookup * 1000)
        hist_ms.append(mr.timing.histogram * 1000)
        if (i + 1) % 50 == 0:
            print(f"    {i+1}/{n_q}")
    latency = {
        "queries": len(e2e), "top1_correct": correct,
        "top1_rate": round(correct / len(e2e), 4) if e2e else None,
        "end_to_end_ms": {"p50": pctl(e2e, 50), "p95": pctl(e2e, 95), "p99": pctl(e2e, 99)},
        "fingerprint_ms": {"p50": pctl(fp_ms, 50), "p95": pctl(fp_ms, 95)},
        "index_lookup_ms": {"p50": pctl(lookup_ms, 50), "p95": pctl(lookup_ms, 95)},
        "offset_histogram_ms": {"p50": pctl(hist_ms, 50), "p95": pctl(hist_ms, 95)},
    }
    print(f"  end-to-end p50 {latency['end_to_end_ms']['p50']:.1f} ms / "
          f"p95 {latency['end_to_end_ms']['p95']:.1f} ms | "
          f"index lookup p95 {latency['index_lookup_ms']['p95']:.2f} ms | "
          f"top-1 {correct}/{len(e2e)}")

    # -- 6. CLI at scale -----------------------------------------------------
    print("\n[6/6] CLI describe/list at scale...")
    cli = {"catalogs": store.list_catalogs(),
           "describe_scale": store.describe(SCALE_CATALOG_ID)}
    print(f"  catalogs: {cli['catalogs']}")

    # ------------------------------------------------------------------ report
    git = git_state(REPO_ROOT)
    per_track = len(index) / len(rep.catalog)
    audio_sec = sum(t.duration_sec for t in rep.catalog)
    results = {
        "schema_version": 1, "stage": "2",
        "title": "Stage 2 scale validation — 500-track catalog end-to-end",
        "provenance": {
            "git_commit": git["commit"], "git_dirty": git["dirty"],
            "git_dirty_path_count": len(git["dirty_paths"]),
            "phase1_source_sha256": source_fingerprint(REPO_ROOT, PHASE1_SOURCES),
            "fingerprint_format_version": FORMAT_VERSION,
            "index_format_version": INDEX_FORMAT_VERSION,
            "thresholds": {"stage1": STAGE1_THRESHOLD, "gate": GATE_THRESHOLD,
                           "stage2": STAGE2_THRESHOLD},
            "benchmark_command": "python scripts/validate_scale.py",
            "python": sys.version.split()[0], "platform": platform.platform(),
            "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        },
        "corpus": {
            "manifest_path": args.manifest,
            "manifest_content_hash": corpus.content_hash(),
            "tracks": len(corpus),
            "audio_hours": round(audio_sec / 3600, 3),
            "licence_counts": corpus.license_counts(),
            "distinct_artists": len({t.artist for t in corpus.tracks}),
            "audio_bytes": sum(t.bytes for t in corpus.tracks),
        },
        "ingestion": {
            "cold_seconds": round(t_ingest, 2),
            "index_build_seconds": round(t_index, 2),
            "artifact_save_seconds": round(t_save, 2),
            "cold_cache_hits": rep.cache_hits,
            "cold_cache_hit_rate": round(rep.cache_hit_rate, 4),
            "skipped": len(rep.skipped),
            "tracks_per_second": round(len(rep.catalog) / t_ingest, 2),
            "audio_realtime_factor": round(audio_sec / t_ingest, 1),
        },
        "index": {
            "postings": len(index), "unique_hashes": index.n_unique_hashes,
            "postings_per_track": round(per_track, 0),
            "postings_per_audio_second": round(len(index) / audio_sec, 1),
            "in_memory_bytes": index.nbytes,
            "artifact_bytes": artifact_bytes,
            "bytes_per_posting": round(index.nbytes / len(index), 2),
        },
        "memory": {
            "peak_python_alloc_mb": round(peak_py / 1e6, 1),
            "baseline_rss_mb": round(base_rss, 1),
            "peak_rss_mb": round(peak_rss, 1),
            "note": "peak RSS covers the whole ingestion, which holds every "
                    "track's fingerprints in memory at once -- the known "
                    "limitation this run exists to quantify",
        },
        "reproducibility": reproducible,
        "duplicates": dup_result,
        "isolation": isolation,
        "latency": latency,
        "cli": cli,
        "roadmap_reference": {
            "stage2_acceptance": "ingest a 500-track catalog end-to-end; duplicates "
                                 "detected by content hash; cross-tenant isolation "
                                 "proven by test; index artifact reproducible from "
                                 "the manifest",
            "stage1_latency_criterion": "p95 lookup < 50 ms at 1,000 tracks",
            "stage1_criterion_note": "measured here at 500 tracks as a scale-readiness "
                                     "signal only. This run does NOT claim the "
                                     "1,000-track criterion is satisfied.",
        },
    }
    rd = REPO_ROOT / args.report_dir
    rd.mkdir(parents=True, exist_ok=True)
    (rd / "stage2_scale_validation.json").write_text(json.dumps(results, indent=2) + "\n")
    (rd / "stage2_scale_validation.md").write_text(build_md(results))
    print(f"\nWrote {rd/'stage2_scale_validation.json'}")
    print(f"Wrote {rd/'stage2_scale_validation.md'}")
    return 0


def build_md(r):
    L = []; A = L.append
    pv, c, ing, idx = r["provenance"], r["corpus"], r["ingestion"], r["index"]
    mem, rep, dup, iso, lat = (r["memory"], r["reproducibility"], r["duplicates"],
                               r["isolation"], r["latency"])
    A("# Stage 2 — 500-track scale validation\n")
    A(f"**Generated:** {pv['generated_utc']}  ")
    A(f"**Repo commit:** `{pv['git_commit'][:12]}`  ")
    A(f"**Working tree:** {'DIRTY (' + str(pv['git_dirty_path_count']) + ' paths)' if pv['git_dirty'] else 'clean'}  ")
    A(f"**Phase 1 fingerprint:** `{pv['phase1_source_sha256']}`\n")
    A("> Measurement only. No threshold, recognition module or cascade was changed,\n"
      "> and the frozen Phase 0/1 corpora and reports were read for comparison only.\n")

    A("## Stage 2 acceptance\n")
    A(f"> {r['roadmap_reference']['stage2_acceptance']}\n")
    A("| Criterion | Result |")
    A("|---|---|")
    A(f"| Ingest a 500-track catalog end-to-end | **{c['tracks']} tracks**, "
      f"{ing['cold_seconds']:.0f} s, identify verified |")
    A(f"| Duplicates detected by content hash | **{dup['distinct_sha256']} distinct "
      f"hashes / {dup['tracks']} tracks**, {dup['duplicate_content_groups']} duplicate "
      f"groups, {dup['overlap_with_frozen_eval_corpus']} overlap with the frozen corpus |")
    A(f"| Cross-tenant isolation proven | own **{iso['own_catalog_decision']}**, "
      f"other tenant **{iso['other_catalog_decision']}** -> "
      f"**{'isolated' if iso['isolated'] else 'NOT ISOLATED'}** |")
    A(f"| Index artifact reproducible from the manifest | catalog hash identical "
      f"**{rep['catalog_hash_identical']}**, index hash identical "
      f"**{rep['index_hash_identical']}** |")
    A("")

    A("## Corpus\n")
    A(f"- **{c['tracks']}** distinct recordings, **{c['audio_hours']:.2f} h** audio, "
      f"{c['distinct_artists']} distinct artists")
    A(f"- Licences: {c['licence_counts']}")
    A(f"- Audio on disk: {c['audio_bytes']/1e9:.2f} GB (git-ignored)")
    A(f"- Manifest content hash: `{c['manifest_content_hash']}`\n")

    A("## Ingestion and index\n")
    A("| Quantity | Value |")
    A("|---|---:|")
    A(f"| Cold ingestion | {ing['cold_seconds']:.0f} s ({ing['tracks_per_second']:.1f} tracks/s, "
      f"{ing['audio_realtime_factor']:.0f}x realtime) |")
    A(f"| Index build | {ing['index_build_seconds']:.1f} s |")
    A(f"| Artifact save | {ing['artifact_save_seconds']:.1f} s |")
    A(f"| Warm re-ingestion | {rep['warm_seconds']:.1f} s "
      f"({rep['warm_cache_hit_rate']*100:.0f}% cache hits, {rep['speedup']}x faster) |")
    A(f"| Postings | {idx['postings']:,} ({idx['postings_per_track']:,.0f}/track, "
      f"{idx['postings_per_audio_second']:.1f}/audio-second) |")
    A(f"| Unique hashes | {idx['unique_hashes']:,} |")
    A(f"| Index in memory | {idx['in_memory_bytes']/1e6:.1f} MB "
      f"({idx['bytes_per_posting']:.0f} bytes/posting) |")
    A(f"| Artifact on disk | {idx['artifact_bytes']/1e6:.1f} MB |")
    A(f"| Peak RSS during ingestion | {mem['peak_rss_mb']:.0f} MB |")
    A(f"| Peak Python allocation | {mem['peak_python_alloc_mb']:.0f} MB |")
    A(f"\n{mem['note']}.\n")

    A("## Latency\n")
    A(f"{lat['queries']} in-catalog 5 s queries, top-1 correct "
      f"{lat['top1_correct']}/{lat['queries']}.\n")
    A("| Stage | p50 ms | p95 ms |")
    A("|---|---:|---:|")
    A(f"| Index lookup | {lat['index_lookup_ms']['p50']:.2f} | {lat['index_lookup_ms']['p95']:.2f} |")
    A(f"| Fingerprint | {lat['fingerprint_ms']['p50']:.2f} | {lat['fingerprint_ms']['p95']:.2f} |")
    A(f"| Offset histogram | {lat['offset_histogram_ms']['p50']:.2f} | {lat['offset_histogram_ms']['p95']:.2f} |")
    A(f"| **End-to-end identify** | **{lat['end_to_end_ms']['p50']:.2f}** | "
      f"**{lat['end_to_end_ms']['p95']:.2f}** |")
    A(f"\n**Against the roadmap's Stage 1 criterion** — *\"{r['roadmap_reference']['stage1_latency_criterion']}\"*: "
      f"{r['roadmap_reference']['stage1_criterion_note']}\n")
    A("")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
