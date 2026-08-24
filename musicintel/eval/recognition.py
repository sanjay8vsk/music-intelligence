"""Recognition benchmark driver.

One command runs the whole thing:

    python -m musicintel.eval.recognition

It loads the fixture manifest, generates a deterministic query set (clean +
degraded + negative), runs a recognizer over it, and writes machine-readable
JSON plus a human-readable Markdown report to eval/reports/.

The driver is recognizer-agnostic: anything satisfying
musicintel.eval.recognizer.Recognizer can be scored without touching this file.
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from musicintel.eval import degradation as dg
from musicintel.eval.manifest import Manifest, Track, sha256_file
from musicintel.eval.metrics import (
    QueryOutcome,
    by_condition_and_duration,
    group_by,
    summarize,
    threshold_sweep,
    worst_and_best,
)
from musicintel.eval.provenance import (
    HARNESS_SOURCES,
    git_state,
    source_fingerprint,
)
from musicintel.eval.recognizer import Recognizer

REPO_ROOT = Path(__file__).resolve().parents[2]
# The manifest is committed (it holds no audio, only identity + provenance +
# licensing), so a fresh clone can reproduce the exact corpus. The audio it
# points at lives under data/, which is git-ignored.
DEFAULT_MANIFEST = REPO_ROOT / "eval/fixtures/manifest.json"
DEFAULT_QUERY_DIR = REPO_ROOT / "data/eval/queries"
DEFAULT_REPORT_DIR = REPO_ROOT / "eval/reports"

# Query plan. Degradations are crossed with duration where clip length is known
# to interact (noise, codec, filtering); speed and pitch are held at 5 s to keep
# the matrix affordable, which is stated in the report's Limitations section.
CLEAN_DURATIONS = dg.DURATIONS
CLEAN_POSITIONS = dg.POSITIONS
CROSSED_DURATIONS = dg.DURATIONS
SINGLE_DURATION = (5.0,)
DEFAULT_POSITION = "middle"

SPEECH_LINES = [
    "The quick brown fox jumps over the lazy dog near the river bank.",
    "Please leave your message after the tone and we will call you back.",
    "Weather today will be cloudy with a chance of rain in the afternoon.",
    "This is a test recording used to evaluate an audio recognition system.",
    "Turn left at the next intersection and continue for two kilometres.",
    "All passengers for the delayed service should proceed to platform nine.",
]


def _repo_relative(path: Path) -> str:
    """Repo-relative path string, falling back to absolute if outside the repo."""
    p = Path(path).resolve()
    try:
        return str(p.relative_to(REPO_ROOT))
    except ValueError:
        return str(p)


# ------------------------------------------------------- query planning ----
def plan_positive_queries(tracks: list[Track]) -> list[dg.QuerySpec]:
    """Deterministic positive query plan for the indexed catalog."""
    specs: list[dg.QuerySpec] = []
    for t in tracks:
        for cond, family, params in dg.condition_matrix():
            if family == "clean":
                combos = [(d, p) for d in CLEAN_DURATIONS for p in CLEAN_POSITIONS]
            elif family in ("noise", "codec", "filter"):
                if params.get("noise_type") == "white":
                    combos = [(d, DEFAULT_POSITION) for d in SINGLE_DURATION]
                else:
                    combos = [(d, DEFAULT_POSITION) for d in CROSSED_DURATIONS]
            else:  # speed, pitch
                combos = [(d, DEFAULT_POSITION) for d in SINGLE_DURATION]

            for duration, position in combos:
                if duration > t.duration_sec:
                    continue  # never request more audio than the source has
                qid = dg.make_query_id(t.track_id, duration, position, cond)
                specs.append(
                    dg.QuerySpec(
                        query_id=qid,
                        track_id=t.track_id,
                        duration=duration,
                        position=position,
                        condition=cond,
                        family=family,
                        params=dict(params),
                        seed=dg.derive_seed(qid),
                        source_hash=t.sha256,
                        is_negative=False,
                    )
                )
    return specs


def plan_heldout_negatives(tracks: list[Track]) -> list[dg.QuerySpec]:
    """Out-of-catalog music: real tracks deliberately never indexed."""
    specs: list[dg.QuerySpec] = []
    for t in tracks:
        for duration in CLEAN_DURATIONS:
            for position in CLEAN_POSITIONS:
                if duration > t.duration_sec:
                    continue
                qid = dg.make_query_id(t.track_id, duration, position, "neg_music")
                specs.append(
                    dg.QuerySpec(
                        query_id=qid,
                        track_id=None,
                        duration=duration,
                        position=position,
                        condition="negative_out_of_catalog_music",
                        family="negative",
                        params={"source_track": t.track_id},
                        seed=dg.derive_seed(qid),
                        source_hash=t.sha256,
                        is_negative=True,
                    )
                )
    return specs


def synthesize_negatives(out_dir: Path) -> list[tuple[dg.QuerySpec, Path]]:
    """Speech, silence and noise -- audio that must never match a catalog track.

    Speech is generated locally with macOS `say`, so no third-party speech
    corpus and no licensing question is involved.
    """
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    made: list[tuple[dg.QuerySpec, Path]] = []
    sr = dg.QUERY_SAMPLE_RATE

    def emit(cond: str, duration: float, y: np.ndarray, tag: str) -> None:
        qid = f"synthetic__{cond}__d{duration:g}s__{tag}"
        spec = dg.QuerySpec(
            query_id=qid,
            track_id=None,
            duration=duration,
            position="beginning",
            condition=cond,
            family="negative",
            params={"synthetic": True},
            seed=dg.derive_seed(qid),
            source_hash="",
            is_negative=True,
        )
        path = out_dir / f"{qid}.wav"
        dg.render(spec, y, sr, path)
        spec = dg.QuerySpec(
            **{**spec.to_dict(), "rendered_path": _repo_relative(path)}
        )
        made.append((spec, path))

    # --- speech -----------------------------------------------------
    say = shutil.which("say")
    if say:
        import librosa

        for i, line in enumerate(SPEECH_LINES):
            raw = out_dir / f"_speech_src_{i}.wav"
            try:
                subprocess.run(
                    [say, "-o", str(raw), "--data-format=LEI16@22050", line],
                    check=True, capture_output=True, timeout=60,
                )
            except Exception:  # noqa: BLE001
                continue
            if not raw.exists():
                continue
            total = librosa.get_duration(path=str(raw))
            for duration in CLEAN_DURATIONS:
                if duration > total:
                    continue
                y, _ = librosa.load(raw, sr=sr, duration=duration, mono=True)
                emit("negative_speech", duration, y, f"utt{i}")
            raw.unlink(missing_ok=True)

    # --- silence, near-silence, noise --------------------------------
    for duration in CLEAN_DURATIONS:
        n = int(sr * duration)
        emit("negative_silence", duration, np.zeros(n, dtype=np.float32), "digital")
        rng = np.random.default_rng(dg.derive_seed(f"nearsil{duration}"))
        emit("negative_near_silence", duration,
             (rng.standard_normal(n) * 1e-4).astype(np.float32), "dither")
        rng = np.random.default_rng(dg.derive_seed(f"white{duration}"))
        emit("negative_noise_white", duration,
             (rng.standard_normal(n) * 0.1).astype(np.float32), "w")
        rng = np.random.default_rng(dg.derive_seed(f"pink{duration}"))
        emit("negative_noise_pink", duration,
             (dg._pink_noise(n, rng) * 0.1).astype(np.float32), "p")

    return made


# ------------------------------------------------------------ rendering ----
def render_queries(
    specs: list[dg.QuerySpec], manifest: Manifest, out_dir: Path, *, verbose: bool = True
) -> list[tuple[dg.QuerySpec, Path]]:
    """Materialize every planned query to disk, deterministically.

    Specs are grouped by source track so each track is decoded exactly once and
    every excerpt is sliced from memory. Seeking inside a compressed source per
    query is orders of magnitude slower.
    """
    from collections import defaultdict

    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    rendered: list[tuple[dg.QuerySpec, Path]] = []
    failures = 0

    by_track: dict[str, list[dg.QuerySpec]] = defaultdict(list)
    for spec in specs:
        src_id = spec.track_id or spec.params.get("source_track")
        if src_id:
            by_track[src_id].append(spec)
        else:
            failures += 1

    done = 0
    for t_i, (track_id, group) in enumerate(sorted(by_track.items()), start=1):
        track = manifest.by_id(track_id)
        if track is None:
            failures += len(group)
            continue
        try:
            y_full, sr = dg.load_source(REPO_ROOT / track.path)
        except Exception as e:  # noqa: BLE001
            failures += len(group)
            if verbose:
                print(f"    decode failed {track_id}: {type(e).__name__}: {e}")
            continue

        for spec in group:
            try:
                y = dg.slice_excerpt(y_full, sr, spec.duration, spec.position)
                rng = np.random.default_rng(spec.seed)
                y2, sr2, measured = dg.apply_condition(
                    y, sr, spec.family, spec.params, rng
                )
                path = out_dir / f"{spec.query_id}.wav"
                dg.render(spec, y2, sr2, path)
                rendered.append(
                    (
                        dg.QuerySpec(
                            **{
                                **spec.to_dict(),
                                "rendered_path": _repo_relative(path),
                                "measured": measured,
                            }
                        ),
                        path,
                    )
                )
                done += 1
            except Exception as e:  # noqa: BLE001
                failures += 1
                if verbose and failures <= 5:
                    print(f"    render failed {spec.query_id}: {type(e).__name__}: {e}")
        if verbose:
            print(f"    [{t_i}/{len(by_track)}] {track_id[:40]:40s} {done} rendered")

    if verbose:
        print(f"    rendered {len(rendered)} queries ({failures} failed)")
    return rendered


def write_query_index(rendered: list[tuple[dg.QuerySpec, Path]], path: Path) -> None:
    """One JSON object per query: full machine-readable condition metadata."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        for spec, _ in rendered:
            fh.write(json.dumps(spec.to_dict(), sort_keys=True) + "\n")


# ------------------------------------------------------------- execution ----
def run_queries(
    recognizer: Recognizer, rendered: list[tuple[dg.QuerySpec, Path]], *, verbose=True
) -> list[QueryOutcome]:
    """Execute every query, timing only the recognize() call."""
    outcomes: list[QueryOutcome] = []
    for i, (spec, path) in enumerate(rendered):
        if verbose and i and i % 250 == 0:
            print(f"    {i}/{len(rendered)} queries")
        err = None
        returned: list[str] = []
        top_distance = None
        t0 = time.perf_counter()
        try:
            result = recognizer.recognize(path)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            returned = [c.track_id for c in result.candidates]
            if result.candidates and result.candidates[0].distance is not None:
                top_distance = result.candidates[0].distance
        except Exception as e:  # noqa: BLE001
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            err = f"{type(e).__name__}: {e}"
        outcomes.append(
            QueryOutcome(
                query_id=spec.query_id,
                condition=spec.condition,
                family=spec.family,
                duration=spec.duration,
                position=spec.position,
                is_negative=spec.is_negative,
                latency_ms=round(elapsed_ms, 3),
                returned_ids=returned,
                truth_track_id=spec.track_id,
                top_distance=top_distance,
                error=err,
            )
        )
    return outcomes


def _repo_relative(path: Path) -> str:
    """Repo-relative path for the report, falling back to absolute.

    A manifest kept outside the repo is legitimate; it must be recorded as it
    is rather than crashing the report write after the whole run has completed.
    """
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path.resolve())


def environment_info(venv_python: str) -> dict:
    import importlib

    versions = {}
    for mod in ("librosa", "numpy", "scipy", "soundfile", "faiss", "sklearn"):
        try:
            m = importlib.import_module(mod)
            versions[mod] = getattr(m, "__version__", "unknown")
        except Exception:  # noqa: BLE001
            versions[mod] = "not installed"
    git = git_state(REPO_ROOT)
    return {
        "python": sys.version.split()[0],
        "executable": venv_python,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": versions,
        "ffmpeg_available": shutil.which("ffmpeg") is not None,
        "git_commit": git["commit"],
        "git_commit_short": git["commit_short"],
        # When the tree is dirty, `git_commit` does NOT describe the code that
        # ran -- `harness_sha256` is what identifies it. Say so rather than
        # letting the commit imply a state a reader could check out.
        "git_dirty": git["dirty"],
        "git_dirty_paths": git["dirty_paths"],
        "harness_sha256": source_fingerprint(REPO_ROOT, HARNESS_SOURCES),
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


# --------------------------------------------------------------- reports ----
def _fmt(v, pct=False, nd=1):
    if v is None:
        return "—"
    if pct:
        return f"{v * 100:.{nd}f}%"
    return f"{v:.{nd}f}"


def _table(rows: list[dict], caption: str) -> str:
    if not rows:
        return f"_{caption}: no data._\n"
    out = ["| Condition | Queries | Recall@1 | Recall@3 | No-match | FAR | p50 ms | p95 ms |",
           "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        out.append(
            f"| `{r['condition']}` | {r['queries']} | {_fmt(r.get('recall_at_1'), True)} "
            f"| {_fmt(r.get('recall_at_3'), True)} | {_fmt(r.get('no_match_rate'), True)} "
            f"| {_fmt(r.get('far'), True)} | {_fmt(r.get('p50_ms'))} | {_fmt(r.get('p95_ms'))} |"
        )
    return "\n".join(out) + "\n"


def build_markdown(results: dict) -> str:
    env = results["environment"]
    ds = results["dataset"]
    ov = results["overall"]
    L: list[str] = []
    A = L.append

    A("# Recognition Baseline\n")
    A(f"**Recognizer:** `{results['recognizer']['name']}` "
      f"(version `{results['recognizer']['version']}`)  ")
    A(f"**Generated:** {env['generated_utc']}  ")
    A(f"**Repo commit:** `{env['git_commit'][:12]}`  ")
    if env.get("git_dirty"):
        A(f"**Working tree: DIRTY** ({len(env['git_dirty_paths'])} uncommitted "
          f"paths) -- the commit above does NOT contain the exact code that ran. "
          f"The harness fingerprint below is what identifies it.\n")
    else:
        A("**Working tree:** clean\n")

    A("## Environment\n")
    A(f"- Python **{env['python']}** on {env['platform']} ({env['machine']})")
    for k, v in env["packages"].items():
        A(f"- {k}: `{v}`")
    A(f"- ffmpeg available: **{env['ffmpeg_available']}**")
    A(f"- Harness source fingerprint: `{env.get('harness_sha256', 'unknown')}`")
    A(f"- Algorithm source fingerprint: "
      f"`{results['recognizer'].get('algorithm_sha256') or 'unknown'}`")
    A(f"- Tracks actually indexed: "
      f"**{results['recognizer'].get('indexed_tracks', 'unknown')}**")
    A("")

    A("## Dataset\n")
    A(f"- Corpus: **{ds['source']}**")
    A(f"- Tracks total: **{ds['track_count']}** "
      f"(indexed catalog: **{ds['catalog_count']}**, held out: **{ds['heldout_count']}**)")
    A(f"- Licenses: {ds['license_counts']}")
    A(f"- Manifest content hash: `{ds['manifest_hash']}`")
    A(f"- Total reference audio: {ds['total_audio_minutes']:.1f} minutes")
    A("\nAudio is **not** committed. Reproduce with "
      "`python scripts/fetch_fixture_corpus.py --tracks 50`.\n")

    A("## Methodology\n")
    A(f"- **{ov['positive_queries']}** positive queries, "
      f"**{ov['negative_queries']}** negative queries "
      f"(**{ov['total_queries']}** total).")
    A("- Excerpts taken at three positions (beginning / middle / end) and three "
      "durations (3 s / 5 s / 10 s). A query is never longer than its source.")
    A("- Clean conditions are crossed with all durations x all positions. "
      "Noise, codec and filtering are crossed with all durations at the middle "
      "position. Speed and pitch are evaluated at 5 s, middle.")
    A("- All randomness is seeded from a SHA-256 of the query id, so the query "
      "set is byte-reproducible.")
    A("- Latency times **only** the `recognize()` call, using `time.perf_counter()`. "
      "Index construction is excluded and reported separately.")
    A(f"- Index build (catalog of {ds['catalog_count']}): "
      f"**{results['recognizer']['prepare_seconds']:.1f} s** total, of which "
      f"**{results['recognizer']['index_build_seconds']:.3f} s** was FAISS index "
      "construction.")
    A("")

    A("## Clean Results\n")
    A(_table(results["tables"]["clean"], "clean"))

    A("### Positive queries by excerpt position and duration\n")
    A("| Slice | Queries | Recall@1 | Recall@3 |")
    A("|---|---:|---:|---:|")
    for label, key in (("position", "by_position"), ("duration", "by_duration")):
        for name, v in sorted(results[key].items()):
            A(f"| {label} = {name} | {v['queries']} | "
              f"{_fmt(v.get('recall_at_1'), True)} | {_fmt(v.get('recall_at_3'), True)} |")
    A("")

    A("## Degradation Results\n")
    for fam in ("noise", "codec", "filter", "speed", "pitch"):
        rows = results["tables"].get(fam) or []
        if rows:
            A(f"### {fam.capitalize()}\n")
            A(_table(rows, fam))

    A("## Negative Results\n")
    A(f"**False Accept Rate: {_fmt(ov.get('far'), True)}** — "
      f"{ov.get('false_accepts')} of {ov['negative_queries']} negative queries "
      "returned a catalog track.")
    A(f"**Correct rejection rate: {_fmt(ov.get('correct_rejection_rate'), True)}**\n")
    A(_table(results["tables"]["negative"], "negative"))

    sweep = results.get("threshold_sweep", {})
    if sweep.get("available"):
        A("### Would a score threshold have helped?\n")
        A("The prototype has no rejection stage, so its FAR is 1.0 by "
          "construction. This sweep asks a different question: if a distance "
          "threshold *were* added, what could it achieve?\n")
        A("| Max FAR allowed | Best Recall@1 | at distance |")
        A("|---|---:|---:|")
        for label, pt in sweep["operating_points"].items():
            if pt:
                A(f"| {label.replace('far_le_', '≤ ')} | "
                  f"{_fmt(pt['recall_at_1'], True)} | {pt['tau']:.3f} |")
            else:
                A(f"| {label.replace('far_le_', '≤ ')} | unreachable | — |")
        dd = sweep["distance_distribution"]
        A("\nL2 distance distributions (overlap = inseparability):\n")
        A("| Set | n | min | p05 | median | p95 | max |")
        A("|---|---:|---:|---:|---:|---:|---:|")
        for key, label in (("correct_positive", "Correct matches"),
                           ("negative", "Negatives (no true match)")):
            d = dd.get(key)
            if d:
                A(f"| {label} | {d['n']} | {d['min']} | {d['p05']} | "
                  f"{d['median']} | {d['p95']} | {d['max']} |")
        A("")

    A("## Worst Conditions (measured)\n")
    A(_table(results["worst_conditions"], "worst"))
    A("## Best Conditions (measured)\n")
    A(_table(results["best_conditions"], "best"))

    A("## Current Recognizer Assessment\n")
    A(results["assessment"])
    A("")

    A("## Limitations\n")
    for lim in results["limitations"]:
        A(f"- {lim}")
    A("")
    return "\n".join(L)


def build_assessment(results: dict) -> str:
    ov = results["overall"]
    clean = results["tables"]["clean"]
    clean5 = next(
        (r for r in clean if r["base_condition"] == "clean" and r["duration"] == 5.0), None
    )
    r1 = clean5["recall_at_1"] if clean5 else None
    sweep = results.get("threshold_sweep", {})
    best = None
    if sweep.get("available"):
        best = sweep["operating_points"].get("far_le_0.01")

    lines = []
    lines.append(
        f"On **clean 5-second excerpts** — the easiest condition in the whole "
        f"matrix — the prototype reaches **Recall@1 = {_fmt(r1, True)}** against a "
        f"catalog of only **{results['dataset']['catalog_count']}** tracks."
    )
    lines.append("")
    lines.append(
        f"Its **False Accept Rate is {_fmt(ov.get('far'), True)}**: it returns "
        "catalog tracks for speech, silence and pure noise, because "
        "`src/music_recognition.py` has no rejection stage at all — it returns "
        "`k=3` unconditionally. That alone makes it unusable as a product."
    )
    lines.append("")
    if best:
        lines.append(
            "The threshold sweep shows this is **not merely a missing threshold**. "
            f"Even with an oracle distance cut-off tuned on the test data itself, "
            f"the best achievable Recall@1 at FAR ≤ 1% is "
            f"**{_fmt(best['recall_at_1'], True)}**. The 26-dimensional "
            "MFCC mean/std representation cannot separate a matching recording "
            "from a non-matching one."
        )
    else:
        lines.append(
            "The threshold sweep could not be computed (insufficient scored "
            "positives and negatives)."
        )
    # The position breakdown exposes something the aggregate numbers hide.
    pos = results.get("by_position", {})
    beg = (pos.get("beginning") or {}).get("recall_at_1")
    mid = (pos.get("middle") or {}).get("recall_at_1")
    end = (pos.get("end") or {}).get("recall_at_1")
    if beg is not None and mid is not None and end is not None:
        lines.append("")
        lines.append(
            f"**The position breakdown exposes what the recognizer is actually "
            f"doing.** Recall@1 is {_fmt(beg, True)} for excerpts from the "
            f"*beginning* of a track, {_fmt(mid, True)} from the *middle*, and "
            f"{_fmt(end, True)} from the *end*. The reference side indexes only "
            "the first 30 seconds of each track (`librosa.load(..., duration=30)` "
            "in `src/audio_processing.py`), and the median corpus track is "
            f"{results['dataset'].get('median_duration_sec', 0):.0f} s long — so "
            "middle and end excerpts share **no audio content at all** with what "
            "was indexed. Any correct answer there cannot come from recognising "
            "the content; it comes from the two excerpts happening to have a "
            "similar overall spectral character. That is the empirical "
            "confirmation that 26-d MFCC mean/std is a **timbre-similarity "
            "descriptor, not an identity fingerprint**."
        )
    lines.append("")
    lines.append(
        "**Verdict: the current MFCC/FAISS approach is not a viable foundation "
        "for Phase 1.** It should be replaced by spectral-peak landmark hashing "
        "with time-offset consistency scoring, not tuned. The measured numbers "
        "above are the baseline every future engine must beat."
    )
    return "\n".join(lines)


# ------------------------------------------------------------------ main ----
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Run the recognition benchmark.")
    ap.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    ap.add_argument("--query-dir", default=str(DEFAULT_QUERY_DIR))
    ap.add_argument("--report-dir", default=str(DEFAULT_REPORT_DIR))
    ap.add_argument("--holdout", type=int, default=12,
                    help="tracks excluded from the index, used as negatives")
    ap.add_argument("--limit-tracks", type=int, default=0,
                    help="cap catalog size (0 = all); for quick runs")
    ap.add_argument("--reuse-queries", action="store_true",
                    help="reuse already-rendered queries if present")
    ap.add_argument("--recognizer", default="mfcc_faiss",
                    choices=["mfcc_faiss"], help="which recognizer to score")
    args = ap.parse_args(argv)

    # Resolve to an absolute path before anything else. A relative --manifest
    # (the documented invocation passes one) otherwise blows up later, when the
    # report tries to record it relative to the repo root.
    manifest_path = Path(args.manifest).expanduser()
    if not manifest_path.is_absolute():
        cwd_candidate = (Path.cwd() / manifest_path).resolve()
        repo_candidate = (REPO_ROOT / manifest_path).resolve()
        manifest_path = cwd_candidate if cwd_candidate.exists() else repo_candidate
    if not manifest_path.exists():
        print(f"ERROR: manifest not found at {manifest_path}")
        print("Run: python scripts/fetch_fixture_corpus.py --tracks 50")
        return 2

    t_start = time.perf_counter()
    manifest = Manifest.load(manifest_path)
    problems = manifest.verify(REPO_ROOT)
    if problems:
        print(f"ERROR: corpus incomplete ({len(problems)} problems):")
        for p in problems[:10]:
            print("   ", p)
        return 2

    manifest.assign_holdout(args.holdout)
    catalog = manifest.catalog
    if args.limit_tracks:
        catalog = catalog[: args.limit_tracks]
    heldout = manifest.held_out
    print(f"Corpus: {len(manifest)} tracks | catalog {len(catalog)} | held out {len(heldout)}")

    # -- plan + render -------------------------------------------------
    query_dir = Path(args.query_dir)
    print("Planning queries...")
    pos_specs = plan_positive_queries(catalog)
    neg_specs = plan_heldout_negatives(heldout)
    print(f"  positive plan: {len(pos_specs)}  |  held-out-music negatives: {len(neg_specs)}")

    if args.reuse_queries and (query_dir / "index.jsonl").exists():
        print("Reusing existing rendered queries...")
        rendered = []
        for line in (query_dir / "index.jsonl").read_text().splitlines():
            d = json.loads(line)
            p = REPO_ROOT / d["rendered_path"]
            if p.exists():
                rendered.append((dg.QuerySpec(**d), p))
    else:
        print("Rendering queries (this is the slow part)...")
        rendered = render_queries(pos_specs + neg_specs, manifest, query_dir)
        print("  synthesizing speech/silence/noise negatives...")
        rendered += synthesize_negatives(query_dir)
        write_query_index(rendered, query_dir / "index.jsonl")

    n_pos = sum(1 for s, _ in rendered if not s.is_negative)
    n_neg = sum(1 for s, _ in rendered if s.is_negative)
    print(f"  ready: {n_pos} positive, {n_neg} negative")

    # -- prepare recognizer --------------------------------------------
    from musicintel.eval.recognizer import MfccFaissRecognizer

    recognizer = MfccFaissRecognizer()
    print(f"Preparing recognizer '{recognizer.name}' over {len(catalog)} tracks...")
    recognizer.prepare(catalog)
    print(f"  prepare: {recognizer.prepare_seconds:.1f}s "
          f"(faiss index build {recognizer.index_build_seconds:.3f}s)")

    # -- run -------------------------------------------------------------
    print("Running queries...")
    outcomes = run_queries(recognizer, rendered)
    total_seconds = time.perf_counter() - t_start

    # -- aggregate --------------------------------------------------------
    print("Aggregating...")
    positives = [o for o in outcomes if not o.is_negative]
    negatives = [o for o in outcomes if o.is_negative]
    all_rows = by_condition_and_duration(outcomes)

    tables: dict[str, list[dict]] = {}
    for fam in ("clean", "noise", "codec", "filter", "speed", "pitch", "negative"):
        tables[fam] = [r for r in all_rows if r["family"] == fam]

    worst, best = worst_and_best([r for r in all_rows if r["family"] != "negative"])

    overall = summarize(outcomes)
    overall.update(
        {
            "positive_queries": len(positives),
            "negative_queries": len(negatives),
            "total_queries": len(outcomes),
        }
    )
    pos_sum = summarize(positives)
    neg_sum = summarize(negatives)
    overall["recall_at_1"] = pos_sum.get("recall_at_1")
    overall["recall_at_3"] = pos_sum.get("recall_at_3")
    overall["far"] = neg_sum.get("far")
    overall["correct_rejection_rate"] = neg_sum.get("correct_rejection_rate")
    overall["false_accepts"] = neg_sum.get("false_accepts")

    total_audio_min = sum(t.duration_sec for t in manifest) / 60.0
    results = {
        "schema_version": 1,
        "recognizer": {
            "name": recognizer.name,
            "version": recognizer.version,
            "algorithm_sha256": getattr(recognizer, "algorithm_sha256", None),
            # What the index really held, as opposed to what the split intended.
            "indexed_tracks": getattr(recognizer, "indexed_tracks", None),
            "prepare_seconds": round(recognizer.prepare_seconds, 3),
            "index_build_seconds": round(recognizer.index_build_seconds, 4),
        },
        "environment": environment_info(sys.executable),
        "dataset": {
            "source": "archive.org netlabels (CC-BY / CC-BY-SA only)",
            "manifest_path": _repo_relative(manifest_path),
            "manifest_hash": manifest.content_hash(),
            "split_hash": manifest.split_hash(),
            "track_count": len(manifest),
            "catalog_count": len(catalog),
            "heldout_count": len(heldout),
            "license_counts": manifest.license_counts(),
            "total_audio_minutes": round(total_audio_min, 2),
            "median_duration_sec": round(
                float(np.median([t.duration_sec for t in manifest])), 1
            ),
        },
        "overall": overall,
        "by_position": {k: summarize(v) for k, v in group_by(positives, "position").items()},
        "by_duration": {k: summarize(v) for k, v in group_by(positives, "duration").items()},
        "by_family": {k: summarize(v) for k, v in group_by(outcomes, "family").items()},
        "tables": tables,
        "threshold_sweep": threshold_sweep(outcomes),
        "worst_conditions": worst,
        "best_conditions": best,
        "performance": {
            "total_evaluation_seconds": round(total_seconds, 2),
            "query_count": len(outcomes),
            **{k: v for k, v in summarize(outcomes).items()
               if k in ("mean_ms", "p50_ms", "p95_ms", "min_ms", "max_ms")},
        },
        "limitations": [
            "Catalog is tiny ({} tracks). Recognition difficulty grows sharply with "
            "catalog size, so these numbers are an OPTIMISTIC upper bound; a "
            "10,000-track catalog would be materially harder."
            .format(len(catalog)),
            "Corpus is CC-BY/CC-BY-SA netlabel electronic-leaning music from "
            "archive.org, not a genre-balanced sample of mainstream commercial music.",
            "Degradations are synthetic. Real phone captures add room acoustics, "
            "handset response, AGC and codec chains simultaneously; no real-world "
            "recordings are included in this run.",
            "ffmpeg is not installed, so codec tests use libsndfile's MP3 and Opus "
            "encoders. Bitrates are VBR/CBR-quality targeted and the achieved "
            "bitrate is measured per file rather than exactly dialled in.",
            "Speed and pitch conditions are evaluated only at 5 s / middle position "
            "to keep the matrix affordable.",
            "Latency is measured on one machine with a warm filesystem cache and "
            "excludes index construction; it is not a server-side SLA figure.",
            "The threshold sweep is fitted on the evaluation set itself, so it is an "
            "OPTIMISTIC upper bound on what any real threshold could achieve.",
            "Negative speech is synthetic (macOS `say`), not natural human speech.",
            "Family-level aggregates are NOT directly comparable to each other: "
            "`clean` spans all three positions while codec/speed/pitch are "
            "middle-position only, and position strongly affects the score. "
            "Compare conditions at the condition level, not the family level.",
            "The reference side indexes only the first 30 s of each track, an "
            "existing property of the prototype that was preserved. This is a "
            "limitation OF THE RECOGNIZER being measured, not of the benchmark.",
        ],
    }
    results["assessment"] = build_assessment(results)

    # -- write ------------------------------------------------------------
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "baseline.json").write_text(json.dumps(results, indent=2) + "\n")
    (report_dir / "baseline.md").write_text(build_markdown(results))

    print("\n" + "=" * 66)
    print(f"  positive queries : {overall['positive_queries']}")
    print(f"  negative queries : {overall['negative_queries']}")
    print(f"  Recall@1         : {_fmt(overall['recall_at_1'], True)}")
    print(f"  Recall@3         : {_fmt(overall['recall_at_3'], True)}")
    print(f"  FAR              : {_fmt(overall['far'], True)}")
    print(f"  p50 / p95 latency: {_fmt(overall['p50_ms'])} ms / {_fmt(overall['p95_ms'])} ms")
    print(f"  total time       : {total_seconds:.1f}s")
    print("=" * 66)
    print(f"\nWrote {report_dir / 'baseline.json'}")
    print(f"Wrote {report_dir / 'baseline.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
