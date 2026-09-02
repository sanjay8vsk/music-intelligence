"""Durable identity, configuration and usage history (Stage 2).

This package holds no fingerprints. Catalog isolation remains structural --
each catalog is its own on-disk index artifact — and nothing here changes what
the recogniser searches. These tables record what exists, who owns it, and what
was consumed.

`psycopg` is an optional dependency (`pip install '.[db]'`). Import this package
only where a database is actually configured; the API degrades to its Stage 3
behaviour when `MUSICINTEL_DATABASE_URL` is unset.
"""

from __future__ import annotations

from musicintel.db.migrate import apply_migrations, applied_migrations
from musicintel.db.pool import DatabaseUnavailable, close_pool, connection, open_pool
from musicintel.db.repositories import (
    ApiKeyRepository,
    CatalogRepository,
    UsageRepository,
    UsageRow,
)

__all__ = [
    "ApiKeyRepository",
    "CatalogRepository",
    "DatabaseUnavailable",
    "UsageRepository",
    "UsageRow",
    "apply_migrations",
    "applied_migrations",
    "close_pool",
    "connection",
    "open_pool",
]
