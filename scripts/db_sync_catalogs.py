#!/usr/bin/env python
"""Sync a CatalogStore into the Stage 2 PostgreSQL tables.

The on-disk artifact stays the source of truth for recognition -- this copies
the *identity* of what it contains into the database so it can be queried,
audited and billed against without loading a 175 MB index.

    python scripts/db_sync_catalogs.py --store data/catalogs --tenant acme
    python scripts/db_sync_catalogs.py --store data/catalogs --map acme=acme,demo=trial

Run migrations first:

    python -m musicintel.db.migrate "$MUSICINTEL_DATABASE_URL"
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from musicintel.catalog.store import CatalogStore                # noqa: E402
from musicintel.db.pool import DatabaseUnavailable, connection   # noqa: E402
from musicintel.db.repositories import CatalogRepository         # noqa: E402


def _tenant_for(catalog_id: str, default: str | None, mapping: dict[str, str]) -> str:
    if catalog_id in mapping:
        return mapping[catalog_id]
    if default:
        return default
    # Refusing beats guessing: an unowned catalog is an isolation question, and
    # inventing a tenant would answer it wrongly and silently.
    raise SystemExit(
        f"no tenant for catalog {catalog_id!r}: pass --tenant or add it to --map")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--store", required=True, type=Path)
    ap.add_argument("--dsn", default=os.environ.get("MUSICINTEL_DATABASE_URL"))
    ap.add_argument("--tenant", help="tenant for every catalog without a --map entry")
    ap.add_argument("--map", default="", help="catalog_id=tenant pairs, comma separated")
    ap.add_argument("--catalog", action="append", dest="only",
                    help="sync only this catalog (repeatable)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    if not args.dsn and not args.dry_run:
        ap.error("--dsn or MUSICINTEL_DATABASE_URL is required")

    mapping = dict(
        pair.split("=", 1) for pair in args.map.split(",") if "=" in pair)

    store = CatalogStore(args.store)
    catalog_ids = args.only or store.list_catalogs()
    if not catalog_ids:
        print(f"no catalogs under {args.store}")
        return 0

    # A dry run reads the store and resolves tenants; it must not need a
    # database to tell you what it would do.
    if args.dry_run:
        for cid in catalog_ids:
            artifact = store.describe(cid)
            tenant = _tenant_for(cid, args.tenant, mapping)
            print(f"[dry-run] {cid}: {artifact['track_count']} tracks "
                  f"-> tenant={tenant}, "
                  f"content_hash={artifact['catalog_content_hash'][:12]}")
        print("dry run complete")
        return 0

    synced = 0
    try:
        with connection(args.dsn) as conn:
            repo = CatalogRepository(conn)
            for cid in catalog_ids:
                loaded = store.load(cid)
                artifact = store.describe(cid)
                tenant = _tenant_for(cid, args.tenant, mapping)
                repo.sync(loaded.catalog, tenant=tenant, artifact=artifact,
                          catalog_id=cid)
                row = repo.get(cid)
                print(f"synced {cid}: {row['track_count']} tracks, "
                      f"tenant={row['tenant']}, content_hash={row['content_hash'][:12]}")
                synced += 1
    except DatabaseUnavailable as exc:
        print(f"database unavailable: {exc}", file=sys.stderr)
        return 1
    print(f"{synced} catalog(s) synced")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
