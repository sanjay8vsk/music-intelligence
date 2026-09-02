"""Write-behind usage persistence.

WHY NOT A SYNCHRONOUS WRITE
---------------------------
Recording usage inline would put a database round trip inside `/v1/identify`.
The Stage 3 acceptance bar is p95 < 300 ms and the measured margin is under
4 ms, so a few milliseconds of Postgres on the hot path is the difference
between passing and failing an accepted criterion. The request path therefore
does one `put_nowait` into a bounded queue -- sub-microsecond, never blocking --
and a background thread does the writing.

WHAT THAT COSTS
---------------
A durability window. Records queued but not yet flushed are lost if the process
dies. The window is bounded by `flush_interval` (default 1 s) and `batch_size`.
That is an accepted trade and it is bounded on both sides:

  * Redis still enforces the quota independently, so a lost usage record can
    never cause the service to over-serve a tenant -- it can only under-bill.
  * The queue is bounded. Under sustained database failure it fills and then
    drops, loudly, rather than growing until the process is killed. Dropping
    usage rows is bad; taking the API down with it is worse, and the limiter
    still holds either way.

Batches are aggregated by (tenant, key_id, day) before writing, so a thousand
queued requests become a handful of upserts.
"""

from __future__ import annotations

import queue
import threading
import time
from datetime import date, datetime, timezone

from musicintel.db.pool import DatabaseUnavailable, connection
from musicintel.db.repositories import UsageRepository


class UsageWriter:
    """Background aggregator and writer for the `usage` table."""

    def __init__(self, *, max_queue: int = 10_000, batch_size: int = 500,
                 flush_interval: float = 1.0, retry_backoff: float = 2.0,
                 max_backoff: float = 60.0, logger=None) -> None:
        self._q: queue.Queue = queue.Queue(maxsize=max_queue)
        self._batch_size = batch_size
        self._flush_interval = flush_interval
        self._retry_backoff = retry_backoff
        self._max_backoff = max_backoff
        self._log = logger
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._idle = threading.Event()
        self._idle.set()

        self.enqueued = 0
        self.written = 0
        self.dropped = 0
        self.failures = 0

    # -- producer side (request path) ----------------------------------
    def record(self, tenant: str, key_id: str, *, audio_seconds: float,
               matched: bool, when: datetime | None = None) -> bool:
        """Queue one request's usage. Never blocks, never raises.

        Returns False if the record was dropped because the queue is full --
        the caller does not act on that; it is surfaced through `dropped`.
        """
        day = (when or datetime.now(timezone.utc)).date()
        try:
            self._q.put_nowait((tenant, key_id, day, float(audio_seconds),
                                1, 1 if matched else 0, 0 if matched else 1))
            self.enqueued += 1
            return True
        except queue.Full:
            self.dropped += 1
            if self._log is not None:
                self._log.warning("usage.dropped", tenant=tenant, key_id=key_id,
                                  queued=self._q.qsize())
            return False

    # -- lifecycle -----------------------------------------------------
    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name="usage-writer", daemon=True)
        self._thread.start()

    def stop(self, *, timeout: float = 10.0) -> None:
        """Stop the writer, flushing whatever is queued first."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None
        self._drain_once(final=True)

    def wait_idle(self, timeout: float = 5.0) -> bool:
        """Block until the queue is empty and the last batch is committed."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._q.empty() and self._idle.is_set():
                return True
            time.sleep(0.01)
        return self._q.empty() and self._idle.is_set()

    # -- consumer side -------------------------------------------------
    def _run(self) -> None:
        backoff = 0.0
        while not self._stop.is_set():
            if backoff:
                if self._stop.wait(backoff):
                    break
            try:
                wrote = self._drain_once()
                backoff = 0.0 if wrote is not None else min(
                    max(self._retry_backoff, backoff * 2), self._max_backoff)
            except Exception:                      # pragma: no cover - defensive
                self.failures += 1
                backoff = min(max(self._retry_backoff, backoff * 2),
                              self._max_backoff)
            if not backoff:
                self._stop.wait(self._flush_interval)

    def _collect(self, final: bool) -> dict:
        """Pull up to `batch_size` records, aggregating as we go."""
        agg: dict[tuple[str, str, date], list] = {}
        limit = self._batch_size if not final else self._q.qsize()
        for _ in range(max(limit, 0)):
            try:
                tenant, key_id, day, secs, reqs, m, n = self._q.get_nowait()
            except queue.Empty:
                break
            slot = agg.setdefault((tenant, key_id, day), [0.0, 0, 0, 0])
            slot[0] += secs
            slot[1] += reqs
            slot[2] += m
            slot[3] += n
        return agg

    def _drain_once(self, *, final: bool = False):
        """One flush. Returns rows written, or None when the database is down."""
        agg = self._collect(final)
        if not agg:
            self._idle.set()
            return 0
        self._idle.clear()
        batch = [(t, k, d, v[0], v[1], v[2], v[3]) for (t, k, d), v in agg.items()]
        try:
            with connection() as conn:
                written = UsageRepository(conn).record_many(batch)
        except DatabaseUnavailable as exc:
            # Put the aggregate back so nothing is lost while the database is
            # away. If the queue has since filled, these are dropped and counted
            # -- an unbounded buffer would trade a billing gap for an outage.
            self.failures += 1
            if self._log is not None:
                self._log.error("usage.write_failed", error=str(exc),
                                pending=len(batch))
            for row in batch:
                try:
                    self._q.put_nowait(row)
                except queue.Full:
                    self.dropped += 1
            self._idle.set()
            return None
        self.written += written
        self._idle.set()
        return written


__all__ = ["UsageWriter"]
