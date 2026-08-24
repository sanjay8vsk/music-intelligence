#!/usr/bin/env python3
"""Build the evaluation fixture corpus from openly-licensed audio on archive.org.

Only PERMISSIVE licenses are accepted (CC0 / public domain / CC-BY / CC-BY-SA).
Licenses carrying NoDerivatives (-nd) are rejected because the benchmark creates
derivative works (noise mixing, transcoding, pitch/speed shifting). Licenses
carrying NonCommercial (-nc) are rejected because this corpus supports the
development of a commercial product.

The license of every track is verified individually against that item's own
metadata endpoint -- never assumed from the collection or the search result.

Audio is written to data/eval/corpus/ which is git-ignored. Only the manifest
(metadata + hashes, no audio) is intended for commit.

Usage:
    python scripts/fetch_fixture_corpus.py [--tracks 50] [--out data/eval/corpus]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from musicintel.eval.manifest import Manifest, Track, sha256_file  # noqa: E402

SEARCH_URL = "https://archive.org/advancedsearch.php"
METADATA_URL = "https://archive.org/metadata/{identifier}"
DOWNLOAD_URL = "https://archive.org/download/{identifier}/{filename}"
USER_AGENT = "musicintel-eval-corpus/0.1 (research benchmark; contact via repo)"

# Permissive license fragments. Anything not matching one of these is rejected.
ALLOWED_LICENSE_PATTERNS = (
    "creativecommons.org/publicdomain/zero",
    "creativecommons.org/publicdomain/mark",
    "creativecommons.org/licenses/by/",
    "creativecommons.org/licenses/by-sa/",
)
# Explicit rejects, checked first (a URL may contain "by" as a substring).
FORBIDDEN_LICENSE_TOKENS = ("-nd", "-nc", "/nc-", "by-nc", "by-nd", "nc-sa", "nc-nd")

MIN_DURATION_SEC = 45.0
MAX_DURATION_SEC = 600.0
MAX_BYTES = 18 * 1024 * 1024


_SESSION = requests.Session()
_SESSION.headers.update({"User-Agent": USER_AGENT})

# (connect, read). The read timeout bounds the gap BETWEEN bytes, so a stalled
# archive.org node fails fast instead of hanging the run indefinitely.
_TIMEOUT = (10, 20)


def _get(url: str, attempts: int = 2, max_seconds: float = 90.0) -> bytes:
    """GET with bounded retries and a hard wall-clock cap on the whole transfer."""
    last: Exception | None = None
    for _ in range(attempts):
        started = time.monotonic()
        try:
            with _SESSION.get(url, timeout=_TIMEOUT, stream=True) as resp:
                resp.raise_for_status()
                chunks = bytearray()
                for chunk in resp.iter_content(chunk_size=1 << 16):
                    chunks.extend(chunk)
                    if time.monotonic() - started > max_seconds:
                        raise TimeoutError(f"exceeded {max_seconds}s budget")
                return bytes(chunks)
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(0.5)
    raise last if last else RuntimeError("unreachable")


def _get_json(url: str) -> dict:
    return json.loads(_get(url, max_seconds=30.0).decode("utf-8", errors="replace"))


def license_is_permissive(license_url: str | None) -> bool:
    """True only for CC0 / public-domain / CC-BY / CC-BY-SA."""
    if not license_url:
        return False
    u = str(license_url).lower()
    if any(tok in u for tok in FORBIDDEN_LICENSE_TOKENS):
        return False
    return any(p in u for p in ALLOWED_LICENSE_PATTERNS)


def license_short_name(license_url: str) -> str:
    u = str(license_url).lower()
    if "publicdomain/zero" in u:
        return "CC0-1.0"
    if "publicdomain/mark" in u:
        return "PublicDomainMark-1.0"
    if "/licenses/by-sa/" in u:
        return "CC-BY-SA"
    if "/licenses/by/" in u:
        return "CC-BY"
    return "UNKNOWN"


def search_candidates(rows: int, page: int) -> list[dict]:
    """Search archive.org for audio items with an explicitly permissive license."""
    license_clause = " OR ".join(
        [
            'licenseurl:"http://creativecommons.org/publicdomain/zero/1.0/"',
            'licenseurl:"http://creativecommons.org/licenses/by/3.0/"',
            'licenseurl:"http://creativecommons.org/licenses/by/4.0/"',
            'licenseurl:"http://creativecommons.org/licenses/by-sa/3.0/"',
            'licenseurl:"http://creativecommons.org/licenses/by-sa/4.0/"',
        ]
    )
    query = f"collection:netlabels AND mediatype:audio AND ({license_clause})"
    params = [
        ("q", query),
        ("rows", str(rows)),
        ("page", str(page)),
        ("output", "json"),
        ("sort[]", "identifier asc"),  # deterministic ordering
    ]
    for field in ("identifier", "title", "creator", "licenseurl", "year", "genre"):
        params.append(("fl[]", field))
    url = SEARCH_URL + "?" + urllib.parse.urlencode(params)
    return _get_json(url)["response"]["docs"]


def pick_audio_files(meta: dict, limit: int = 2) -> list[dict]:
    """Choose up to `limit` suitable MP3s from an item's file list.

    Taking more than one track per item amortizes the metadata round-trip. It
    also makes the catalog slightly harder in a realistic way: same-release
    tracks share production characteristics, which is exactly the near-duplicate
    case an exact recognizer must not confuse.
    """
    ok: list[tuple[float, dict]] = []
    for f in meta.get("files", []):
        if f.get("format") not in (
            "VBR MP3", "128Kbps MP3", "256Kbps MP3", "320Kbps MP3", "64Kbps MP3"
        ):
            continue
        try:
            size = int(f.get("size", 0))
            length = float(f.get("length", 0) or 0)
        except (TypeError, ValueError):
            continue
        if not (MIN_DURATION_SEC <= length <= MAX_DURATION_SEC):
            continue
        if size <= 0 or size > MAX_BYTES:
            continue
        ok.append((length, f))
    # Shortest first: smaller downloads, same evidential value.
    ok.sort(key=lambda x: x[0])
    return [f for _, f in ok[:limit]]


def probe_item(doc: dict) -> tuple[str, dict, str] | None:
    """Fetch and license-verify one item. Returns (identifier, metadata, license)."""
    identifier = doc.get("identifier")
    if not identifier:
        return None
    try:
        meta = _get_json(METADATA_URL.format(identifier=identifier))
    except Exception:  # noqa: BLE001
        return None
    lic = meta.get("metadata", {}).get("licenseurl")
    if isinstance(lic, list):
        lic = lic[0] if lic else None
    if not license_is_permissive(lic):
        return None
    return identifier, meta, str(lic)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tracks", type=int, default=50, help="how many tracks to fetch")
    ap.add_argument("--out", default="data/eval/corpus", help="audio output directory")
    # The manifest carries no audio -- only identity, provenance and licensing --
    # so it lives outside the git-ignored data/ tree and IS committed.
    ap.add_argument("--manifest", default="eval/fixtures/manifest.json")
    ap.add_argument("--max-pages", type=int, default=20)
    args = ap.parse_args()

    out_dir = REPO_ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = REPO_ROOT / args.manifest
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    tracks: list[Track] = []
    seen_creators: dict[str, int] = {}
    rejected_license = 0
    rejected_nofile = 0
    page = 1

    print(f"Target: {args.tracks} tracks with permissive licenses only")
    print("Accepted: CC0, Public Domain Mark, CC-BY, CC-BY-SA")
    print("Rejected: anything with -nc (NonCommercial) or -nd (NoDerivatives)\n")

    while len(tracks) < args.tracks and page <= args.max_pages:
        try:
            docs = search_candidates(rows=100, page=page)
        except Exception as e:  # noqa: BLE001
            print(f"  search page {page} failed: {type(e).__name__}: {e}")
            break
        if not docs:
            break
        print(f"[page {page}] {len(docs)} candidate items")

        # -- stage 1: probe + license-verify in parallel (I/O bound) --------
        with ThreadPoolExecutor(max_workers=8) as pool:
            probed = list(pool.map(probe_item, docs))
        rejected_license += sum(1 for p in probed if p is None)

        # -- stage 2: select sequentially so quotas stay deterministic ------
        plan: list[dict] = []
        for item in probed:
            if item is None:
                continue
            if len(tracks) + len(plan) >= args.tracks:
                break
            identifier, meta, lic = item
            item_meta = meta.get("metadata", {})

            creator = item_meta.get("creator") or "unknown"
            if isinstance(creator, list):
                creator = creator[0] if creator else "unknown"
            creator = str(creator)[:80]
            # Cap tracks per artist: artist repetition is a known evaluation hazard.
            remaining_for_creator = 2 - seen_creators.get(creator, 0)
            if remaining_for_creator <= 0:
                continue

            files = pick_audio_files(meta, limit=remaining_for_creator)
            if not files:
                rejected_nofile += 1
                continue

            title = item_meta.get("title") or identifier
            if isinstance(title, list):
                title = title[0] if title else identifier
            genre = item_meta.get("genre")
            if isinstance(genre, list):
                genre = genre[0] if genre else None

            for idx, f in enumerate(files):
                if len(tracks) + len(plan) >= args.tracks:
                    break
                suffix = "" if idx == 0 else f"_{idx}"
                plan.append(
                    {
                        "identifier": identifier,
                        "file": f,
                        "creator": creator,
                        "title": str(title)[:160],
                        "genre": str(genre)[:60] if genre else None,
                        "license": lic,
                        "track_id": f"ia_{identifier}{suffix}"[:96],
                    }
                )
                seen_creators[creator] = seen_creators.get(creator, 0) + 1

        # -- stage 3: download in parallel ---------------------------------
        def fetch_one(job: dict) -> Track | None:
            fname = job["file"]["name"]
            dest = out_dir / f"{job['track_id']}{Path(fname).suffix or '.mp3'}"
            if not dest.exists():
                url = DOWNLOAD_URL.format(
                    identifier=job["identifier"], filename=urllib.parse.quote(fname)
                )
                try:
                    dest.write_bytes(_get(url, attempts=2, max_seconds=120.0))
                except Exception:  # noqa: BLE001
                    return None
            return Track(
                track_id=job["track_id"],
                path=str(dest.relative_to(REPO_ROOT)),
                title=job["title"],
                artist=job["creator"],
                genre=job["genre"],
                source="archive.org/netlabels",
                source_url=f"https://archive.org/details/{job['identifier']}",
                license=license_short_name(job["license"]),
                license_url=str(job["license"]),
                duration_sec=float(job["file"].get("length", 0) or 0),
                sha256=sha256_file(dest),
                bytes=dest.stat().st_size,
            )

        with ThreadPoolExecutor(max_workers=6) as pool:
            for t in pool.map(fetch_one, plan):
                if t is None:
                    continue
                tracks.append(t)
                print(
                    f"  [{len(tracks):2d}/{args.tracks}] {t.track_id[:44]:44s} "
                    f"{t.license:14s} {t.duration_sec:6.1f}s"
                )
        page += 1

    if not tracks:
        print("\nNo tracks obtained. Nothing written.")
        return 1

    manifest = Manifest(tracks=tracks)
    manifest.save(manifest_path)

    print(f"\nFetched {len(tracks)} tracks")
    print(f"  rejected (license not permissive): {rejected_license}")
    print(f"  rejected (no suitable audio file): {rejected_nofile}")
    print(f"  licenses: {manifest.license_counts()}")
    print(f"  audio    -> {out_dir}  (git-ignored)")
    print(f"  manifest -> {manifest_path}")
    print(f"  manifest sha256: {manifest.content_hash()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
