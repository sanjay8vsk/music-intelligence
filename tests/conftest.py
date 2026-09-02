"""Shared test guards.

The MusicBrainz rate limiter coordinates processes through a file in the system
temp directory, keyed by uid. That is right for production -- two CLI runs on one
machine must not both decide they may go first -- but it means a test that
constructs a limiter without an explicit `state_path` reads and writes state
shared with every other test, with any concurrently running enrichment job, and
with whatever a previous run left behind.

The symptom was an intermittent failure in
`test_the_limiter_enforces_a_minimum_interval`: a limiter whose first acquire
normally proceeds immediately must wait when the shared stamp is recent, so
`waits` becomes 4 rather than 3. It looked like timing flakiness. It was shared
mutable state.

The guard fills in a per-test `state_path` at construction. It deliberately does
not patch `default_state_path` itself, because that function carries its own
contract -- the stamp must live outside the repository -- which a test asserts
directly and which must keep seeing the real implementation.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def isolate_musicbrainz_rate_limit_state(tmp_path, monkeypatch):
    """No test may read or write the machine-wide rate-limit stamp."""
    from musicintel.enrichment.musicbrainz import RateLimiter

    state = tmp_path / "musicbrainz-ratelimit.state"
    original_init = RateLimiter.__init__

    def guarded_init(self, *args, **kwargs):
        if kwargs.get("state_path") is None:
            kwargs["state_path"] = state
        original_init(self, *args, **kwargs)

    monkeypatch.setattr(RateLimiter, "__init__", guarded_init)
    return state
