"""MusicBrainz web-service client.

SCOPE
-----
Metadata only. This never touches fingerprints, thresholds or matching, and
nothing in this module is reachable from `/v1/identify` -- enrichment runs as an
offline CLI worker, because at one request per second even 500 tracks take
minutes and neither a request nor application start-up can wait for that.

WHAT COMES FROM WHERE
---------------------
Timeouts, bounded attempts and the hard wall-clock cap follow the repository's
existing fetch convention (`scripts/fetch_fixture_corpus.py`: a `Session` with a
User-Agent, `(10, 20)` timeouts, a small number of attempts, and a cap on total
transfer time).

The **one request per second** floor and the requirement that the User-Agent
carry a **real contact** are MusicBrainz's own published policy, not a repository
convention. They are honoured here because calling someone else's free service
without doing so is rude and gets clients blocked.

NO CONTACT, NO CLIENT
---------------------
`contact` is mandatory and unvalidated-but-required: the constructor refuses to
build without one rather than sending a placeholder. The repository's corpus
fetcher uses `"contact via repo"`, which is acceptable for a handful of
archive.org calls and is not acceptable at MusicBrainz's scale. No contact
string is invented here; it is supplied by deployment configuration.
"""

from __future__ import annotations

import os
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import requests

try:                                    # POSIX advisory locking
    import fcntl
    _FLOCK = True
except ImportError:                     # pragma: no cover - non-POSIX
    fcntl = None                        # type: ignore[assignment]
    _FLOCK = False

DEFAULT_BASE_URL = "https://musicbrainz.org/ws/2"

# OUR operating default, not MusicBrainz's stated requirement. Their published
# policy is "on average one request per second"; this is 2.0 s because a
# characterisation run measured what that actually costs in practice:
#
#     interval   lookups   HTTP   200   503   retries   final errors
#       3.0 s      12       12     12     0       0           0
#       2.0 s      12       12     12     0       0           0
#       1.0 s       8       17      6    11       9           2
#
# 2.0 s is the smallest interval *tested* that was 503-free. The true server-side
# threshold lies somewhere in (1.0, 2.0] and is NOT known -- 1.5 s was never
# tested. Raise or lower it per deployment with `min_interval`; the value is
# configurable precisely because it is an empirical operating point rather than a
# published limit.
MIN_REQUEST_INTERVAL = 2.0


class MusicBrainzError(RuntimeError):
    """Base class for lookup failures."""


class ContactRequired(MusicBrainzError):
    """No contact was configured. See MUSICINTEL_MUSICBRAINZ_CONTACT."""


class TransientError(MusicBrainzError):
    """Worth retrying: 503, 429, timeout, connection reset."""


class PermanentError(MusicBrainzError):
    """Not worth retrying: 4xx other than 429, malformed payload."""


@dataclass(frozen=True)
class LookupResult:
    """One lookup's outcome, before normalisation."""

    status: str                 # matched | no_match | ambiguous | error
    query: str
    candidates: list[dict] = field(default_factory=list)
    raw: dict | None = None
    error: str | None = None
    attempts: int = 0
    seconds: float = 0.0


def default_state_path() -> Path:
    """Where the shared last-request timestamp lives.

    The system temporary directory, not the repository: this is transient
    coordination state, not something to version or ship. The uid is in the name
    so two users on one machine cannot collide on a file one of them cannot
    open.
    """
    return Path(tempfile.gettempdir()) / f"musicintel-musicbrainz-ratelimit-{os.getuid()}"


class RateLimiter:
    """A minimum interval between requests, enforced ACROSS processes.

    WHY NOT JUST AN IN-PROCESS TIMESTAMP
    ------------------------------------
    That is what this was, and the live experiment caught it. Each CLI
    invocation builds a fresh client, so the in-memory `_last` reset to zero and
    the first request of a new run never waited. Measured: run 2's first request
    left **16 ms** after run 1's last. MusicBrainz limits per source IP, not per
    process, so two back-to-back invocations broke the limit no single run ever
    broke.

    HOW IT IS FIXED
    ---------------
    The last-request time is kept in a small file, and the file is locked with
    `flock` for the whole of `acquire()` -- **including the sleep**. A second
    process entering `acquire()` blocks on the lock until the first has both
    waited and stamped, then computes its own wait from that fresh stamp. Two
    processes starting simultaneously therefore serialise instead of both
    concluding they may go first.

    Wall clock, not monotonic, because `time.monotonic()` epochs are not
    guaranteed comparable between processes. A clock moving backwards is handled
    by waiting a full interval rather than trusting a negative elapsed time, and
    every computed wait is clamped to `[0, min_interval]`, so no clock anomaly
    can produce either a burst or an unbounded sleep.

    CRASH SAFETY
    ------------
    `flock` is released by the kernel when the file descriptor closes, including
    on abnormal termination -- there is no lock to leave behind. A stale
    timestamp is harmless: it can only make the next caller wait up to one extra
    interval, never less.

    LIMITS OF THIS
    --------------
    It coordinates processes of one user on one machine, which is the shape of
    the deployment: one service account running an offline worker. It does not
    coordinate across hosts sharing an outbound IP; that would need a shared
    store, and nothing in Stage 2 requires it.
    """

    def __init__(self, min_interval: float = MIN_REQUEST_INTERVAL, *,
                 state_path: str | Path | None = None,
                 cross_process: bool = True) -> None:
        self.min_interval = float(min_interval)
        self._lock = threading.Lock()
        self._last = 0.0                     # monotonic; in-process fallback only
        self.waits = 0
        self.total_wait = 0.0
        self.state_path = Path(state_path) if state_path else default_state_path()
        # Degrades rather than fails where flock is unavailable; the in-process
        # guarantee still holds and is never weaker than before this change.
        self.cross_process = bool(cross_process) and _FLOCK

    def _wait_for(self, last: float, now: float) -> float:
        """Seconds to sleep given a previous timestamp. Always in [0, interval]."""
        if last <= 0:
            return 0.0
        elapsed = now - last
        if elapsed < 0:                      # clock moved backwards
            return self.min_interval         # assume no time has passed
        if elapsed >= self.min_interval:
            return 0.0
        return min(self.min_interval, self.min_interval - elapsed)

    def _sleep(self, wait: float) -> None:
        if wait > 0:
            time.sleep(wait)
            self.waits += 1
            self.total_wait += wait

    def _acquire_local(self) -> float:
        now = time.monotonic()
        wait = self._wait_for(self._last, now)
        self._sleep(wait)
        self._last = time.monotonic()
        return wait

    def _acquire_shared(self) -> float:
        try:
            fd = os.open(self.state_path, os.O_RDWR | os.O_CREAT, 0o600)
        except OSError:
            # Unwritable state directory: fall back rather than refuse to run.
            self.cross_process = False
            return self._acquire_local()
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)   # blocks other processes
            try:
                raw = os.read(fd, 64).decode("ascii", "ignore").strip()
                last = float(raw) if raw else 0.0
            except (ValueError, OSError):
                last = 0.0                   # corrupt stamp: treat as absent
            wait = self._wait_for(last, time.time())
            self._sleep(wait)
            os.lseek(fd, 0, os.SEEK_SET)
            os.ftruncate(fd, 0)
            os.write(fd, f"{time.time():.6f}".encode("ascii"))
            return wait
        finally:
            os.close(fd)                     # releases the lock

    def acquire(self) -> float:
        with self._lock:                     # threads first, then processes
            if self.cross_process:
                return self._acquire_shared()
            return self._acquire_local()


class MusicBrainzClient:
    """Bounded, rate-limited recording search."""

    def __init__(self, contact: str | None, *, base_url: str = DEFAULT_BASE_URL,
                 app_name: str = "musicintel", app_version: str = "0.1",
                 timeout: tuple[float, float] = (10.0, 20.0),
                 max_attempts: int = 3, backoff: float = 1.0,
                 max_seconds_per_track: float = 60.0,
                 min_interval: float = MIN_REQUEST_INTERVAL,
                 session: requests.Session | None = None,
                 ambiguity_margin: float = 5.0,
                 min_score: float = 80.0,
                 rate_limit_state_path: str | Path | None = None) -> None:
        contact = (contact or "").strip()
        if not contact:
            raise ContactRequired(
                "MusicBrainz requires a User-Agent naming a real contact. Set "
                "MUSICINTEL_MUSICBRAINZ_CONTACT (an email address or project "
                "URL). No default is supplied because inventing one would "
                "misrepresent who is calling.")
        self.contact = contact
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_attempts = max(1, int(max_attempts))
        self.backoff = float(backoff)
        self.max_seconds_per_track = float(max_seconds_per_track)
        self.ambiguity_margin = float(ambiguity_margin)
        self.min_score = float(min_score)
        self.limiter = RateLimiter(min_interval, state_path=rate_limit_state_path)

        self.user_agent = f"{app_name}/{app_version} ( {contact} )"
        self.session = session or requests.Session()
        self.session.headers.update({"User-Agent": self.user_agent,
                                     "Accept": "application/json"})
        self.requests_made = 0

    # -- query ------------------------------------------------------------
    @staticmethod
    def build_query(artist: str, title: str) -> str:
        """A Lucene query over recording, with values quoted and escaped."""
        def esc(v: str) -> str:
            out = []
            for ch in str(v):
                if ch in '+-&|!(){}[]^"~*?:\\/':
                    out.append("\\" + ch)
                else:
                    out.append(ch)
            return "".join(out).strip()
        return f'recording:"{esc(title)}" AND artist:"{esc(artist)}"'

    def search_recording(self, artist: str, title: str, *, limit: int = 5) -> LookupResult:
        """Search for a recording. Never raises for a normal negative outcome."""
        query = self.build_query(artist, title)
        started = time.monotonic()
        attempts = 0
        last_error: str | None = None

        while attempts < self.max_attempts:
            if time.monotonic() - started > self.max_seconds_per_track:
                return LookupResult("error", query, error="wall-clock cap exceeded",
                                    attempts=attempts,
                                    seconds=time.monotonic() - started)
            attempts += 1
            self.limiter.acquire()
            try:
                payload = self._get("/recording", {"query": query, "fmt": "json",
                                                   "limit": int(limit)})
            except TransientError as exc:
                last_error = str(exc)
                if attempts >= self.max_attempts:
                    break
                time.sleep(self.backoff * attempts)
                continue
            except PermanentError as exc:
                return LookupResult("error", query, error=str(exc), attempts=attempts,
                                    seconds=time.monotonic() - started)

            recordings = payload.get("recordings")
            if not isinstance(recordings, list):
                return LookupResult("error", query,
                                    error="response has no 'recordings' list",
                                    raw=payload, attempts=attempts,
                                    seconds=time.monotonic() - started)
            status = self._classify(recordings)
            return LookupResult(status, query, candidates=recordings, raw=payload,
                                attempts=attempts,
                                seconds=time.monotonic() - started)

        return LookupResult("error", query, error=last_error or "retries exhausted",
                            attempts=attempts, seconds=time.monotonic() - started)

    def _classify(self, recordings: list[dict]) -> str:
        """matched / ambiguous / no_match.

        `ambiguous` is never silently promoted to `matched`: when the top two
        candidates score within `ambiguity_margin` of each other there is no
        basis for choosing, and recording a match would invent certainty.
        """
        scored = [r for r in recordings if isinstance(r, dict)]
        if not scored:
            return "no_match"
        top = float(scored[0].get("score") or 0)
        if top < self.min_score:
            return "no_match"
        if len(scored) > 1:
            second = float(scored[1].get("score") or 0)
            if top - second < self.ambiguity_margin:
                return "ambiguous"
        return "matched"

    # -- transport --------------------------------------------------------
    def _get(self, path: str, params: dict[str, Any]) -> dict:
        url = f"{self.base_url}{path}"
        try:
            resp = self.session.get(url, params=params, timeout=self.timeout)
        except requests.Timeout as exc:
            raise TransientError(f"timeout: {exc}") from exc
        except requests.RequestException as exc:
            raise TransientError(f"connection error: {exc}") from exc
        self.requests_made += 1

        if resp.status_code in (429, 503, 502, 504):
            raise TransientError(f"HTTP {resp.status_code}")
        if resp.status_code >= 400:
            # Every other 4xx is our fault and will not improve on retry.
            raise PermanentError(f"HTTP {resp.status_code}")
        try:
            payload = resp.json()
        except ValueError as exc:
            raise PermanentError(f"response was not JSON: {exc}") from exc
        if not isinstance(payload, dict):
            raise PermanentError("response was not a JSON object")
        return payload


__all__ = [
    "DEFAULT_BASE_URL", "MIN_REQUEST_INTERVAL", "ContactRequired",
    "LookupResult", "MusicBrainzClient", "MusicBrainzError", "PermanentError",
    "RateLimiter", "TransientError", "default_state_path",
]
