#!/usr/bin/env python
"""Enrich a catalog's tracks with MusicBrainz metadata. Offline worker.

Deliberately a CLI and not part of the API. MusicBrainz asks for at most one
request per second, so 500 tracks take at least 8 minutes -- that cannot live in
application start-up and certainly not in a request. Nothing in the identify
path touches this.

Restartable: already-enriched tracks are skipped unless `--force` or
`--max-age-days` says otherwise, so an interrupted run resumes where it stopped.

    MUSICINTEL_MUSICBRAINZ_CONTACT=you@example.com \
    python scripts/enrich_musicbrainz.py --dsn "$DSN" --catalog acme
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from musicintel.db.pool import DatabaseUnavailable, connection        # noqa: E402
from musicintel.db.repositories import TrackMetadataRepository        # noqa: E402
from musicintel.enrichment.musicbrainz import (                       # noqa: E402
    MIN_REQUEST_INTERVAL, ContactRequired, MusicBrainzClient,
)
from musicintel.enrichment.normalize import normalize_recording       # noqa: E402

SOURCE = "musicbrainz"


def enrich(conn, client, catalog_id: str, *, force: bool = False,
           max_age_days: int | None = None, limit: int | None = None,
           progress_every: int = 25, echo=print) -> dict:
    """Enrich one catalog. Returns a counts summary. Never raises per track."""
    repo = TrackMetadataRepository(conn)
    pending = repo.pending(catalog_id, source=SOURCE, force=force,
                           max_age_days=max_age_days, limit=limit)
    missing = repo.without_metadata(catalog_id)

    # Tracks with no title/artist are recorded as `skipped`, not guessed at:
    # searching MusicBrainz by a filename-derived track_id produces confident
    # wrong answers, which is worse than no answer.
    for track_id in missing:
        repo.upsert(catalog_id, track_id, source=SOURCE, match_status="skipped",
                    query_used=None)

    counts = {"eligible": len(pending), "matched": 0, "no_match": 0,
              "ambiguous": 0, "error": 0, "skipped_missing_metadata": len(missing),
              "rows_written": len(missing)}
    started = time.monotonic()

    # Snapshot the client's LIFETIME counters. They are never reset -- correctly
    # so, they belong to the client -- but a second enrich() on the same client
    # would otherwise report the first run's requests too. Measured before this
    # was fixed: a second run that dispatched 3 requests reported 6.
    requests_before = client.requests_made
    waits_before = client.limiter.waits

    dispatches = 0
    first_lookup_at: float | None = None
    last_lookup_end: float | None = None

    for i, (track_id, title, artist) in enumerate(pending, start=1):
        if first_lookup_at is None:
            first_lookup_at = time.monotonic()
        result = client.search_recording(artist, title)
        last_lookup_end = time.monotonic()
        # `attempts` is the dispatch count. Each attempt issues one request, and
        # the wall-clock cap returns BEFORE incrementing, so an attempt that was
        # prevented is never counted. Unlike `requests_made` this includes a
        # request that was sent and then timed out.
        dispatches += result.attempts
        # `attempts` is recorded for EVERY outcome, not only failures. On a
        # successful row it is the only trace that a retry was needed --
        # transient trouble that resolved itself, which `requests_made` cannot
        # show because it never counts a timed-out request. `error_detail` is
        # written only when there is one, so successful rows keep it NULL.
        fields: dict = {"query_used": result.query, "attempts": result.attempts}
        if result.error:
            fields["error_detail"] = result.error
        if result.status in ("matched", "ambiguous") and result.candidates:
            fields.update(normalize_recording(result.candidates[0]).as_fields())
        repo.upsert(catalog_id, track_id, source=SOURCE,
                    match_status=result.status, raw_response=result.raw, **fields)
        counts[result.status] = counts.get(result.status, 0) + 1
        counts["rows_written"] += 1
        if progress_every and i % progress_every == 0:
            # Lookups per second, which is NOT the dispatch rate: a lookup that
            # retried issued several requests. Labelled accordingly.
            pace = i / max(time.monotonic() - started, 1e-9)
            echo(f"    {i}/{len(pending)}  {pace:.2f} lookups/s")

    counts["elapsed_seconds"] = round(time.monotonic() - started, 2)
    # Per-run deltas, not the client's lifetime totals.
    counts["requests_made"] = client.requests_made - requests_before
    counts["rate_limit_waits"] = client.limiter.waits - waits_before
    counts["http_dispatches"] = dispatches

    # HTTP dispatches per second across this run's request window.
    #
    # (dispatches - 1) / window, not dispatches / window: n requests occupy n-1
    # intervals, so dividing the count by the span they cover reports a rate
    # higher than the one actually sustained. That was the defect -- 4 requests
    # at a 0.5 s interval reported 2.632/s when the sustained rate was 2.0/s.
    #
    # The window is measured at LOOKUP boundaries (first lookup start to last
    # lookup end), because the client exposes no per-dispatch timestamps and
    # adding them would be instrumentation this metric does not need. It is
    # therefore slightly WIDER than the true first-to-last dispatch span -- by
    # roughly the final response time -- which makes the reported rate a little
    # conservative. It is never overstated.
    window = ((last_lookup_end - first_lookup_at)
              if first_lookup_at is not None and last_lookup_end is not None
              else 0.0)
    counts["dispatch_window_seconds"] = round(window, 3)
    # Undefined below two dispatches: one request spans no interval, and a
    # zero-length window has no rate. Reported as 0.0 rather than invented.
    counts["http_dispatches_per_sec"] = (
        round((dispatches - 1) / window, 3)
        if dispatches >= 2 and window > 0 else 0.0)
    return counts


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dsn", default=os.environ.get("MUSICINTEL_DATABASE_URL"))
    ap.add_argument("--catalog", required=True)
    ap.add_argument("--contact",
                    default=os.environ.get("MUSICINTEL_MUSICBRAINZ_CONTACT"))
    ap.add_argument("--base-url",
                    default=os.environ.get("MUSICINTEL_MUSICBRAINZ_BASE_URL"))
    ap.add_argument("--limit", type=int)
    ap.add_argument("--force", action="store_true",
                    help="re-enrich tracks that already have metadata")
    ap.add_argument("--max-age-days", type=int,
                    help="re-enrich anything fetched longer ago than this")
    ap.add_argument("--min-interval", type=float,
                    default=MIN_REQUEST_INTERVAL,
                    help="seconds between requests (default: the shipped "
                         "operating interval)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    if not args.dsn:
        ap.error("--dsn or MUSICINTEL_DATABASE_URL is required")

    kwargs = {"min_interval": args.min_interval}
    if args.base_url:
        kwargs["base_url"] = args.base_url
    try:
        client = MusicBrainzClient(args.contact, **kwargs)
    except ContactRequired as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    try:
        with connection(args.dsn) as conn:
            repo = TrackMetadataRepository(conn)
            if args.dry_run:
                pending = repo.pending(args.catalog, source=SOURCE,
                                       force=args.force,
                                       max_age_days=args.max_age_days,
                                       limit=args.limit)
                missing = repo.without_metadata(args.catalog)
                print(f"[dry-run] catalog {args.catalog}")
                print(f"  eligible for enrichment      : {len(pending)}")
                print(f"  skipped (no title/artist)    : {len(missing)}")
                print(f"  User-Agent                   : {client.user_agent}")
                print(f"  minimum request interval     : {client.limiter.min_interval}s")
                return 0
            counts = enrich(conn, client, args.catalog, force=args.force,
                            max_age_days=args.max_age_days, limit=args.limit)
    except DatabaseUnavailable as exc:
        print(f"database unavailable: {exc}", file=sys.stderr)
        return 1

    print(f"\ncatalog {args.catalog}")
    for key in ("eligible", "matched", "no_match", "ambiguous", "error",
                "skipped_missing_metadata", "rows_written", "requests_made",
                "http_dispatches", "elapsed_seconds", "dispatch_window_seconds",
                "http_dispatches_per_sec", "rate_limit_waits"):
        print(f"  {key:<26} {counts.get(key)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
