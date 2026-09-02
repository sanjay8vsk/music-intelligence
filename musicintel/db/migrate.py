"""Migration runner.

Plain SQL files applied in filename order, each recorded in `schema_migrations`
so re-running is a no-op. Not Alembic: one migration does not justify the
dependency, autogenerate would be actively unhelpful against a hand-written
schema full of CHECK constraints, and the whole runner is forty lines.

Each file runs inside one transaction. A migration that fails leaves no partial
schema and no row claiming it succeeded.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

MIGRATIONS_DIR = Path(__file__).parent / "migrations"

_BOOTSTRAP = """
CREATE TABLE IF NOT EXISTS schema_migrations (
    name        text        PRIMARY KEY,
    sha256      char(64)    NOT NULL,
    applied_at  timestamptz NOT NULL DEFAULT now()
)
"""


def _migration_files(directory: Path | None = None) -> list[Path]:
    d = directory or MIGRATIONS_DIR
    return sorted(p for p in d.glob("*.sql") if p.is_file())


def applied_migrations(conn) -> dict[str, str]:
    """name -> sha256 for everything already applied."""
    with conn.cursor() as cur:
        cur.execute(_BOOTSTRAP)
        cur.execute("SELECT name, sha256 FROM schema_migrations")
        return dict(cur.fetchall())
    

def apply_migrations(conn, *, directory: Path | None = None) -> list[str]:
    """Apply pending migrations. Returns the names newly applied."""
    already = applied_migrations(conn)
    conn.commit()

    newly: list[str] = []
    for path in _migration_files(directory):
        sql = path.read_text()
        digest = hashlib.sha256(sql.encode("utf-8")).hexdigest()
        recorded = already.get(path.name)
        if recorded is not None:
            if recorded != digest:
                # An applied migration was edited. Silently ignoring that is how
                # environments drift apart without anyone noticing.
                raise RuntimeError(
                    f"migration {path.name} changed after it was applied "
                    f"(recorded {recorded[:12]}, file {digest[:12]}). "
                    "Add a new migration instead of editing an applied one."
                )
            continue
        with conn.cursor() as cur:
            cur.execute(sql)
            cur.execute(
                "INSERT INTO schema_migrations (name, sha256) VALUES (%s, %s)",
                (path.name, digest),
            )
        conn.commit()
        newly.append(path.name)
    return newly


__all__ = ["MIGRATIONS_DIR", "apply_migrations", "applied_migrations"]


def _main(argv: list[str]) -> int:
    """`python -m musicintel.db.migrate <dsn>` -- the deployment release step."""
    import os
    import sys

    dsn = argv[1] if len(argv) > 1 else os.environ.get("MUSICINTEL_DATABASE_URL")
    if not dsn:
        print("usage: python -m musicintel.db.migrate <dsn>\n"
              "   or: MUSICINTEL_DATABASE_URL=... python -m musicintel.db.migrate",
              file=sys.stderr)
        return 2
    from musicintel.db.pool import connection
    with connection(dsn) as conn:
        applied = apply_migrations(conn)
        current = sorted(applied_migrations(conn))
    for name in applied:
        print(f"applied {name}")
    if not applied:
        print("no pending migrations")
    print(f"schema at: {current[-1] if current else '(empty)'}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    import sys
    sys.exit(_main(sys.argv))
