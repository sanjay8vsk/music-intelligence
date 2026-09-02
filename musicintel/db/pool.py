"""Connection management.

A single process-wide pool. Sync rather than async on purpose: the two callers
are start-up configuration reads and a background usage writer, neither of which
is on the request path. Introducing an async driver would buy nothing and would
put `await` points inside the identify handler, which is exactly where the Stage
3 latency budget has no room -- the measured p95 margin is under 4 ms.
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from typing import Iterator

_pool = None
_lock = threading.Lock()


class DatabaseUnavailable(RuntimeError):
    """The database is not configured, not installed, or not reachable."""


def _require_psycopg():
    try:
        import psycopg  # noqa: F401
        from psycopg_pool import ConnectionPool
    except ModuleNotFoundError as exc:  # pragma: no cover - deployment error
        raise DatabaseUnavailable(
            "psycopg is not installed; install with: pip install '.[db]'"
        ) from exc
    return ConnectionPool


def open_pool(
    dsn: str, *, min_size: int = 1, max_size: int = 8, timeout: float = 5.0
):
    """Open the process-wide pool. Idempotent; returns the live pool.

    `check` is set so a connection killed by a database restart is replaced
    rather than handed out dead -- the usage writer is long-lived and would
    otherwise hold a broken connection indefinitely.
    """
    global _pool
    ConnectionPool = _require_psycopg()
    with _lock:
        if _pool is None:
            _pool = ConnectionPool(
                dsn, min_size=min_size, max_size=max_size, timeout=timeout,
                open=False, check=ConnectionPool.check_connection,
            )
            _pool.open(wait=True, timeout=timeout)
        return _pool


def close_pool() -> None:
    global _pool
    with _lock:
        if _pool is not None:
            _pool.close()
            _pool = None


def is_open() -> bool:
    return _pool is not None


@contextmanager
def connection(dsn: str | None = None) -> Iterator:
    """A pooled connection, or a standalone one when `dsn` is given.

    Raises DatabaseUnavailable rather than a driver-specific error so callers
    have one exception to handle regardless of why the database is missing.
    """
    if dsn is not None:
        import psycopg
        try:
            with psycopg.connect(dsn) as conn:
                yield conn
        except Exception as exc:
            raise DatabaseUnavailable(str(exc)) from exc
        return

    if _pool is None:
        raise DatabaseUnavailable("the database pool is not open")
    try:
        with _pool.connection() as conn:
            yield conn
    except DatabaseUnavailable:
        raise
    except Exception as exc:
        raise DatabaseUnavailable(str(exc)) from exc


__all__ = ["DatabaseUnavailable", "close_pool", "connection", "is_open", "open_pool"]
