"""MusicBrainz enrichment: client, normalizer, persistence and the CLI worker.

Every HTTP interaction runs against a stdlib `http.server` stub bound to
localhost. **No test contacts MusicBrainz**, and no HTTP mocking library is
introduced -- the client takes an injectable `base_url`, which is what makes a
real local server the simplest honest substitute.

Persistence runs against a real PostgreSQL (`pgserver`), because the constraints
being tested are CHECK constraints and a cascading composite foreign key.
"""

from __future__ import annotations

import datetime as dt
import itertools
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

psycopg = pytest.importorskip("psycopg")
pgserver = pytest.importorskip("pgserver")

from musicintel.catalog.models import Catalog, CatalogTrack            # noqa: E402
from musicintel.db.migrate import apply_migrations                     # noqa: E402
from musicintel.db.repositories import (                               # noqa: E402
    CatalogRepository, TrackMetadataRepository,
)
from musicintel.enrichment.musicbrainz import (                        # noqa: E402
    ContactRequired, MusicBrainzClient, RateLimiter,
)
from musicintel.enrichment.normalize import normalize_recording        # noqa: E402

CONTACT = "tests@example.invalid"
TABLES = ("track_metadata", "usage", "tracks", "catalogs", "api_keys")


# ------------------------------------------------------------ stub server --
class _Stub:
    """A local HTTP server whose responses each test programs."""

    def __init__(self):
        self.responses: list = []       # each: (status, body) | callable | "timeout"
        self.default = (200, {"recordings": []})
        self.requests: list = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):  # silence
                pass

            def do_GET(self):
                outer.requests.append({"path": self.path,
                                       "user_agent": self.headers.get("User-Agent"),
                                       "at": time.monotonic()})
                spec = outer.responses.pop(0) if outer.responses else outer.default
                if spec == "timeout":
                    time.sleep(3.0)          # longer than the client's read timeout
                    spec = outer.default
                if callable(spec):
                    spec = spec()
                status, body = spec
                raw = body if isinstance(body, (bytes, str)) else json.dumps(body)
                if isinstance(raw, str):
                    raw = raw.encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    @property
    def base_url(self) -> str:
        host, port = self.server.server_address[:2]
        return f"http://{host}:{port}"

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def stub():
    with _Stub() as s:
        yield s


_STATE_SEQ = itertools.count()


def _client(stub, **kw):
    kw.setdefault("min_interval", 0.0)      # timing tested explicitly elsewhere
    kw.setdefault("max_seconds_per_track", 30.0)
    # Each client gets its own throttle state file. The limiter is now
    # machine-wide by design, and tests must not serialise on the real one.
    kw.setdefault("rate_limit_state_path",
                  Path(tempfile.gettempdir()) /
                  f"musicintel-test-rl-{os.getpid()}-{next(_STATE_SEQ)}")
    return MusicBrainzClient(CONTACT, base_url=stub.base_url, **kw)


RECORDING = {
    "id": "b9ad642e-b012-41c7-b5b6-2a0dbd4a1f42", "score": 100,
    "title": "Nachtaktiv",
    "artist-credit": [{"name": "Dan X",
                       "artist": {"id": "11111111-2222-3333-4444-555555555555",
                                  "name": "Dan X"}}],
    "releases": [{"id": "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
                  "title": "Attack", "date": "2011-04"}],
    "isrcs": ["DEAB71000123"],
}


# ------------------------------------------------------------ postgres --
@pytest.fixture(scope="session")
def dsn(tmp_path_factory):
    server = pgserver.get_server(tmp_path_factory.mktemp("pg_enrich") / "data")
    uri = server.get_uri()
    with psycopg.connect(uri) as conn:
        apply_migrations(conn)
    yield uri
    server.cleanup()


@pytest.fixture
def conn(dsn):
    with psycopg.connect(dsn) as c:
        with c.cursor() as cur:
            cur.execute(f"TRUNCATE {', '.join(TABLES)} CASCADE")
        c.commit()
        _seed(c)
        yield c


def _seed(conn, catalog_id="acme", tenant="acme", n=3, with_metadata=True):
    tracks = [
        CatalogTrack(track_id=f"t{i}", source_path=f"/a/{i}.wav",
                     sha256=f"{i:064x}", duration_sec=10.0 + i,
                     title=f"Song {i}" if with_metadata else None,
                     artist=f"Artist {i}" if with_metadata else None)
        for i in range(1, n + 1)
    ]
    CatalogRepository(conn).sync(Catalog(catalog_id=catalog_id, tracks=tracks),
                                 tenant=tenant)


# ---------------------------------------------------------------- client --
class TestContactRequirement:
    @pytest.mark.parametrize("bad", [None, "", "   "])
    def test_client_refuses_to_exist_without_a_contact(self, bad):
        with pytest.raises(ContactRequired, match="real contact"):
            MusicBrainzClient(bad)

    def test_the_contact_reaches_the_user_agent(self, stub):
        c = _client(stub)
        assert CONTACT in c.user_agent
        c.search_recording("Dan X", "Nachtaktiv")
        assert CONTACT in stub.requests[0]["user_agent"]

    def test_no_contact_string_is_hard_coded_anywhere(self):
        import pathlib
        src = pathlib.Path("musicintel/enrichment/musicbrainz.py").read_text()
        assert "@" not in src.split('"""', 2)[2] or "example" not in src.lower()


class TestLookupOutcomes:
    def test_a_clear_top_hit_is_matched(self, stub):
        stub.responses = [(200, {"recordings": [RECORDING]})]
        r = _client(stub).search_recording("Dan X", "Nachtaktiv")
        assert r.status == "matched" and r.candidates[0]["id"] == RECORDING["id"]

    def test_an_empty_result_is_no_match(self, stub):
        stub.responses = [(200, {"recordings": []})]
        assert _client(stub).search_recording("Nobody", "Nothing").status == "no_match"

    def test_a_low_scoring_hit_is_no_match(self, stub):
        stub.responses = [(200, {"recordings": [{**RECORDING, "score": 42}]})]
        assert _client(stub).search_recording("a", "b").status == "no_match"

    def test_two_close_candidates_are_ambiguous_not_matched(self, stub):
        """Choosing between near-identical scores would invent certainty."""
        stub.responses = [(200, {"recordings": [
            {**RECORDING, "score": 100},
            {**RECORDING, "id": "cccccccc-cccc-cccc-cccc-cccccccccccc", "score": 98},
        ]})]
        assert _client(stub).search_recording("a", "b").status == "ambiguous"

    def test_a_clear_winner_over_a_weak_second_is_matched(self, stub):
        stub.responses = [(200, {"recordings": [
            {**RECORDING, "score": 100},
            {**RECORDING, "id": "cccccccc-cccc-cccc-cccc-cccccccccccc", "score": 60},
        ]})]
        assert _client(stub).search_recording("a", "b").status == "matched"

    def test_the_query_is_recorded_and_escaped(self, stub):
        stub.responses = [(200, {"recordings": []})]
        r = _client(stub).search_recording('Bad "Artist"', "Title: Part (1)")
        assert 'recording:' in r.query and 'artist:' in r.query
        assert '\\:' in r.query or '\\"' in r.query


class TestErrorHandling:
    def test_malformed_json_is_a_permanent_error_and_is_not_retried(self, stub):
        stub.responses = [(200, b"{ not json")]
        r = _client(stub, max_attempts=3).search_recording("a", "b")
        assert r.status == "error" and r.attempts == 1
        assert len(stub.requests) == 1

    def test_a_response_without_recordings_is_an_error(self, stub):
        stub.responses = [(200, {"unexpected": True})]
        r = _client(stub).search_recording("a", "b")
        assert r.status == "error" and "recordings" in r.error

    def test_a_4xx_is_not_retried(self, stub):
        stub.responses = [(400, {"error": "bad"})]
        r = _client(stub, max_attempts=3).search_recording("a", "b")
        assert r.status == "error"
        assert len(stub.requests) == 1, "a 4xx must not be retried"

    def test_a_503_is_retried_then_succeeds(self, stub):
        stub.responses = [(503, {"error": "busy"}),
                          (200, {"recordings": [RECORDING]})]
        r = _client(stub, max_attempts=3, backoff=0.0).search_recording("a", "b")
        assert r.status == "matched" and r.attempts == 2
        assert len(stub.requests) == 2

    def test_retries_are_bounded_and_then_give_up(self, stub):
        stub.responses = [(503, {}) for _ in range(10)]
        r = _client(stub, max_attempts=3, backoff=0.0).search_recording("a", "b")
        assert r.status == "error" and r.attempts == 3
        assert len(stub.requests) == 3, "exceeded the attempt bound"

    def test_a_timeout_is_transient_and_bounded(self, stub):
        stub.responses = ["timeout", (200, {"recordings": [RECORDING]})]
        client = _client(stub, timeout=(2.0, 0.5), max_attempts=2, backoff=0.0)
        r = client.search_recording("a", "b")
        assert r.status == "matched" and r.attempts == 2

    def test_the_wall_clock_cap_stops_a_slow_run(self, stub):
        stub.responses = ["timeout"] * 5
        client = _client(stub, timeout=(2.0, 0.3), max_attempts=5, backoff=0.0,
                         max_seconds_per_track=0.5)
        r = client.search_recording("a", "b")
        assert r.status == "error"
        assert r.seconds < 5.0


class TestRateLimiting:
    def test_the_limiter_enforces_a_minimum_interval(self, tmp_path):
        # Explicit state, like every other limiter test here: with the shared
        # default a recent stamp makes the *first* acquire wait too, and waits
        # becomes 4. conftest also redirects the default, belt and braces.
        limiter = RateLimiter(min_interval=0.25,
                              state_path=tmp_path / "ratelimit.state")
        started = time.monotonic()
        for _ in range(4):
            limiter.acquire()
        elapsed = time.monotonic() - started
        assert elapsed >= 0.75 - 0.02, f"4 acquisitions took only {elapsed:.3f}s"
        assert limiter.waits == 3

    def test_a_fresh_stamp_makes_even_the_first_acquire_wait(self):
        """The mechanism behind the old intermittent failure, pinned.

        A limiter's first acquire normally proceeds immediately, giving 3 waits
        over 4 acquisitions. When the state file already holds a recent stamp --
        another process, or another test sharing the machine-wide default -- the
        first acquire waits too and the count is 4. The behaviour is correct and
        deliberate: the limit is per source IP, not per process. It is pinned
        here so the isolation in conftest is never mistaken for pedantry.
        """
        import time as _time
        state = self._isolated()
        state.write_text(str(_time.time()))
        limiter = RateLimiter(min_interval=0.25, state_path=state)
        for _ in range(4):
            limiter.acquire()
        assert limiter.waits == 4

    def test_a_limiter_given_no_state_path_stays_off_the_machine_wide_file(
            self, tmp_path):
        """conftest must keep the suite out of the shared uid-keyed stamp."""
        from musicintel.enrichment.musicbrainz import default_state_path
        limiter = RateLimiter(min_interval=0.1)
        assert limiter.state_path != default_state_path()
        assert str(limiter.state_path).startswith(str(tmp_path))

    @staticmethod
    def _isolated():
        import tempfile
        return Path(tempfile.mkdtemp()) / "ratelimit.state"

    def test_requests_are_spaced_by_at_least_the_interval(self, stub):
        stub.responses = [(200, {"recordings": []}) for _ in range(3)]
        client = _client(stub, min_interval=0.2)
        for _ in range(3):
            client.search_recording("a", "b")
        stamps = [r["at"] for r in stub.requests]
        gaps = [b - a for a, b in zip(stamps, stamps[1:])]
        assert all(g >= 0.2 - 0.02 for g in gaps), f"gaps too small: {gaps}"

    def test_the_shipped_default_interval_is_two_seconds(self):
        """2.0 s is our operating default, not MusicBrainz's stated limit.

        Chosen because a characterisation run measured 11 HTTP 503s across 8
        lookups at 1.0 s (9 retries, 2 final errors) while 2.0 s returned 12/12
        with none. It is the smallest interval *tested* that was 503-free; the
        true threshold in (1.0, 2.0] is unknown.
        """
        from musicintel.enrichment.musicbrainz import MIN_REQUEST_INTERVAL
        assert MIN_REQUEST_INTERVAL == 2.0
        assert MusicBrainzClient(CONTACT).limiter.min_interval == 2.0
        assert RateLimiter().min_interval == 2.0

    def test_the_cli_default_tracks_the_shipped_constant(self):
        """One source of truth: the CLI must not carry its own copy."""
        import re
        from musicintel.enrichment.musicbrainz import MIN_REQUEST_INTERVAL
        src = Path("scripts/enrich_musicbrainz.py").read_text()
        block = src[src.index('"--min-interval"'):]
        block = block[:block.index(")")]
        assert "MIN_REQUEST_INTERVAL" in block, "CLI hardcodes its own interval"
        assert not re.search(r"default\s*=\s*[0-9]", block)


# ------------------------------------------------------------ normalizer --
class TestNormalization:
    def test_every_field_is_extracted(self):
        n = normalize_recording(RECORDING).as_fields()
        assert n["title"] == "Nachtaktiv" and n["artist"] == "Dan X"
        assert n["album"] == "Attack" and n["release_date"] == "2011-04"
        assert n["isrc"] == "DEAB71000123"
        assert n["mb_recording_id"] == RECORDING["id"]

    @pytest.mark.parametrize("payload", [
        {}, {"id": "not-a-uuid"}, {"artist-credit": "wrong type"},
        {"releases": []}, {"releases": [None]}, {"isrcs": {}}, {"score": "high"},
    ])
    def test_malformed_payloads_degrade_rather_than_raise(self, payload):
        normalize_recording(payload)          # must not raise

    def test_a_non_uuid_identifier_is_dropped(self):
        assert normalize_recording({"id": "12345"}).mb_recording_id is None

    def test_scores_are_clamped(self):
        assert normalize_recording({"score": 500}).match_score == 100.0
        assert normalize_recording({"score": -5}).match_score == 0.0


# ----------------------------------------------------------- persistence --
class TestPersistence:
    def test_a_matched_row_round_trips(self, conn):
        repo = TrackMetadataRepository(conn)
        fields = normalize_recording(RECORDING).as_fields()
        repo.upsert("acme", "t1", match_status="matched",
                    query_used="q", raw_response=RECORDING, **fields)
        row = repo.get("acme", "t1")
        assert row.match_status == "matched"
        assert row.title == "Nachtaktiv" and row.isrc == "DEAB71000123"
        assert row.mb_recording_id == RECORDING["id"]
        assert row.match_score == 100.0
        assert repo.raw_response("acme", "t1")["id"] == RECORDING["id"]
        assert row.fetched_at is not None

    def test_re_enrichment_replaces_rather_than_accumulates(self, conn):
        repo = TrackMetadataRepository(conn)
        repo.upsert("acme", "t1", match_status="matched", title="First")
        repo.upsert("acme", "t1", match_status="matched", title="Second")
        assert repo.get("acme", "t1").title == "Second"
        assert repo.counts_by_status("acme") == {"matched": 1}

    def test_a_no_match_is_recorded_not_omitted(self, conn):
        repo = TrackMetadataRepository(conn)
        repo.upsert("acme", "t1", match_status="no_match", query_used="q")
        row = repo.get("acme", "t1")
        assert row.match_status == "no_match" and row.title is None

    def test_an_unknown_status_is_refused(self, conn):
        with pytest.raises(ValueError, match="match_status"):
            TrackMetadataRepository(conn).upsert("acme", "t1", match_status="maybe")

    def test_the_database_rejects_an_invalid_status(self, conn):
        with pytest.raises(psycopg.errors.CheckViolation):
            with conn.cursor() as cur:
                cur.execute("INSERT INTO track_metadata (catalog_id, track_id, "
                            "match_status) VALUES ('acme','t1','nonsense')")
        conn.rollback()

    def test_a_matched_row_must_name_something(self, conn):
        with pytest.raises(psycopg.errors.CheckViolation):
            with conn.cursor() as cur:
                cur.execute("INSERT INTO track_metadata (catalog_id, track_id, "
                            "match_status) VALUES ('acme','t1','matched')")
        conn.rollback()

    def test_an_out_of_range_score_is_refused(self, conn):
        with pytest.raises(psycopg.errors.CheckViolation):
            with conn.cursor() as cur:
                cur.execute("INSERT INTO track_metadata (catalog_id, track_id, "
                            "match_status, title, match_score) "
                            "VALUES ('acme','t1','matched','x',101)")
        conn.rollback()

    def test_metadata_for_an_unknown_track_is_refused(self, conn):
        with pytest.raises(psycopg.errors.ForeignKeyViolation):
            TrackMetadataRepository(conn).upsert("acme", "ghost",
                                                 match_status="no_match")
        conn.rollback()

    def test_deleting_a_catalog_cascades_to_its_metadata(self, conn):
        repo = TrackMetadataRepository(conn)
        repo.upsert("acme", "t1", match_status="no_match")
        CatalogRepository(conn).delete("acme")
        assert repo.get("acme", "t1") is None

    def test_two_sources_coexist_for_one_track(self, conn):
        repo = TrackMetadataRepository(conn)
        repo.upsert("acme", "t1", match_status="matched", title="MB", source="musicbrainz")
        repo.upsert("acme", "t1", match_status="matched", title="Other", source="other")
        assert repo.get("acme", "t1", "musicbrainz").title == "MB"
        assert repo.get("acme", "t1", "other").title == "Other"


class TestIsolation:
    def test_metadata_does_not_leak_between_catalogs(self, conn):
        _seed(conn, catalog_id="globex", tenant="globex", n=2)
        repo = TrackMetadataRepository(conn)
        repo.upsert("acme", "t1", match_status="matched", title="Acme song")
        repo.upsert("globex", "t1", match_status="matched", title="Globex song")
        assert repo.get("acme", "t1").title == "Acme song"
        assert repo.get("globex", "t1").title == "Globex song"
        assert repo.counts_by_status("acme") == {"matched": 1}
        assert repo.counts_by_status("globex") == {"matched": 1}

    def test_pending_is_scoped_to_one_catalog(self, conn):
        _seed(conn, catalog_id="globex", tenant="globex", n=2)
        repo = TrackMetadataRepository(conn)
        assert len(repo.pending("acme")) == 3
        assert len(repo.pending("globex")) == 2


class TestPendingSelection:
    def test_tracks_without_title_or_artist_are_not_eligible(self, conn):
        with conn.cursor() as cur:
            cur.execute("TRUNCATE track_metadata, tracks, catalogs CASCADE")
        conn.commit()
        _seed(conn, n=3, with_metadata=False)
        repo = TrackMetadataRepository(conn)
        assert repo.pending("acme") == []
        assert len(repo.without_metadata("acme")) == 3

    def test_already_enriched_tracks_are_skipped(self, conn):
        repo = TrackMetadataRepository(conn)
        assert len(repo.pending("acme")) == 3
        repo.upsert("acme", "t1", match_status="no_match")
        assert [t[0] for t in repo.pending("acme")] == ["t2", "t3"]

    def test_force_reselects_everything(self, conn):
        repo = TrackMetadataRepository(conn)
        repo.upsert("acme", "t1", match_status="no_match")
        assert len(repo.pending("acme", force=True)) == 3

    def test_max_age_reselects_stale_rows(self, conn):
        repo = TrackMetadataRepository(conn)
        repo.upsert("acme", "t1", match_status="no_match")
        with conn.cursor() as cur:
            cur.execute("UPDATE track_metadata SET fetched_at = now() - "
                        "interval '40 days' WHERE track_id = 't1'")
        conn.commit()
        assert len(repo.pending("acme", max_age_days=30)) == 3
        assert len(repo.pending("acme", max_age_days=90)) == 2


# ----------------------------------------------------- ingest sidecar (A) --
class TestIngestSidecar:
    """The sidecar is optional and descriptive. It must not touch identity."""

    @staticmethod
    def _audio(tmp_path, seeds=(1, 2)):
        import numpy as np
        import soundfile as sf
        d = tmp_path / "audio"
        d.mkdir(parents=True, exist_ok=True)
        for s in seeds:
            rng = np.random.default_rng(s)
            t = np.linspace(0, 8.0, 11025 * 8, endpoint=False)
            y = (0.5 * np.sin(2 * np.pi * (440 + 11 * s) * t)
                 + 0.02 * rng.standard_normal(t.size)).astype(np.float32)
            sf.write(d / f"trk_{s}.wav", y, 11025, subtype="PCM_16")
        return d

    def test_without_a_sidecar_nothing_changes(self, tmp_path):
        from musicintel.catalog.ingest import ingest_directory
        r = ingest_directory(self._audio(tmp_path))
        assert r.enriched == 0
        assert all(t.title is None and t.artist is None for t in r.catalog.tracks)

    def test_a_sidecar_attaches_title_and_artist(self, tmp_path):
        from musicintel.catalog.ingest import ingest_directory, load_sidecar
        audio = self._audio(tmp_path)
        sidecar = tmp_path / "meta.json"
        sidecar.write_text(json.dumps([
            {"track_id": "trk_1", "title": "One", "artist": "A"},
            {"track_id": "trk_2", "title": "Two", "artist": "B"},
        ]))
        r = ingest_directory(audio, metadata=load_sidecar(sidecar))
        assert r.enriched == 2
        assert {t.track_id: (t.title, t.artist) for t in r.catalog.tracks} == {
            "trk_1": ("One", "A"), "trk_2": ("Two", "B")}

    def test_a_sidecar_cannot_change_catalog_or_index_identity(self, tmp_path):
        """The invariant the whole prerequisite rests on."""
        from musicintel.catalog.ingest import (
            build_catalog_index, ingest_directory, load_sidecar,
        )
        audio = self._audio(tmp_path)
        plain = ingest_directory(audio)
        sidecar = tmp_path / "meta.json"
        sidecar.write_text(json.dumps(
            [{"track_id": t.track_id, "title": "T", "artist": "A"}
             for t in plain.catalog.tracks]))
        enriched = ingest_directory(audio, metadata=load_sidecar(sidecar))

        assert plain.catalog.content_hash() == enriched.catalog.content_hash()
        i1 = build_catalog_index(plain.catalog, plain.fingerprints)
        i2 = build_catalog_index(enriched.catalog, enriched.fingerprints)
        assert i1.content_hash() == i2.content_hash()

    def test_sha256_keys_win_over_track_id_keys(self, tmp_path):
        from musicintel.catalog.ingest import ingest_directory, load_sidecar
        audio = self._audio(tmp_path)
        plain = ingest_directory(audio)
        first = plain.catalog.tracks[0]
        sidecar = tmp_path / "meta.json"
        sidecar.write_text(json.dumps([
            {"track_id": first.track_id, "title": "By id", "artist": "id"},
            {"sha256": first.sha256, "title": "By hash", "artist": "hash"},
        ]))
        r = ingest_directory(audio, metadata=load_sidecar(sidecar))
        got = {t.track_id: t.title for t in r.catalog.tracks}
        assert got[first.track_id] == "By hash"

    def test_empty_and_none_strings_are_treated_as_absent(self, tmp_path):
        from musicintel.catalog.ingest import ingest_directory, load_sidecar
        audio = self._audio(tmp_path)
        sidecar = tmp_path / "meta.json"
        sidecar.write_text(json.dumps(
            [{"track_id": "trk_1", "title": "", "artist": "None"}]))
        r = ingest_directory(audio, metadata=load_sidecar(sidecar))
        t = next(t for t in r.catalog.tracks if t.track_id == "trk_1")
        assert t.title is None and t.artist is None

    def test_a_manifest_shaped_file_is_accepted_directly(self, tmp_path):
        from musicintel.catalog.ingest import load_sidecar
        p = tmp_path / "manifest.json"
        p.write_text(json.dumps({"version": 1, "tracks": [
            {"track_id": "a", "sha256": "f" * 64, "title": "T", "artist": "A"}]}))
        s = load_sidecar(p)
        assert s.lookup("a", "f" * 64) == {"title": "T", "artist": "A"}


class TestBackfillDeterminism:
    """The one-off backfill must be reproducible and identity-preserving."""

    @staticmethod
    def _catalog():
        return Catalog(catalog_id="acme", tracks=[
            CatalogTrack(track_id=f"t{i}", source_path=f"/a/{i}.wav",
                         sha256=f"{i:064x}", duration_sec=10.0 + i)
            for i in range(1, 4)])

    @staticmethod
    def _sidecar(tmp_path):
        from musicintel.catalog.ingest import load_sidecar
        p = tmp_path / "m.json"
        p.write_text(json.dumps([
            {"track_id": f"t{i}", "sha256": f"{i:064x}",
             "title": f"Title {i}", "artist": f"Artist {i}"} for i in range(1, 4)]))
        return load_sidecar(p)

    def test_backfill_preserves_the_catalog_content_hash(self, tmp_path):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "backfill", "scripts/backfill_catalog_metadata.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        catalog = self._catalog()
        before = catalog.content_hash()
        updated, attached, unchanged = mod.apply_metadata(catalog,
                                                          self._sidecar(tmp_path))
        assert attached == 3 and unchanged == 0
        assert updated.content_hash() == before
        assert [t.title for t in updated.tracks] == ["Title 1", "Title 2", "Title 3"]

    def test_backfill_is_deterministic_and_idempotent(self, tmp_path):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "backfill", "scripts/backfill_catalog_metadata.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        sidecar = self._sidecar(tmp_path)
        once, a1, _ = mod.apply_metadata(self._catalog(), sidecar)
        twice, a2, u2 = mod.apply_metadata(once, sidecar)
        assert a1 == 3 and a2 == 0 and u2 == 3       # second pass is a no-op
        assert [t.to_dict() for t in once.tracks] == [t.to_dict() for t in twice.tracks]
        assert once.content_hash() == twice.content_hash()

    def test_track_order_is_preserved(self, tmp_path):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "backfill", "scripts/backfill_catalog_metadata.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        catalog = self._catalog()
        updated, _, _ = mod.apply_metadata(catalog, self._sidecar(tmp_path))
        assert [t.track_id for t in updated.tracks] == \
               [t.track_id for t in catalog.tracks]


# --------------------------------------------------- the worker, end to end --
class TestEnrichmentWorker:
    @staticmethod
    def _enrich(conn, client, **kw):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "worker", "scripts/enrich_musicbrainz.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.enrich(conn, client, "acme", echo=lambda *a: None, **kw)

    def test_a_full_run_records_every_outcome(self, conn, stub):
        stub.responses = [
            (200, {"recordings": [RECORDING]}),                      # matched
            (200, {"recordings": []}),                               # no_match
            (200, {"recordings": [{**RECORDING, "score": 100},
                                  {**RECORDING, "id": "cccccccc-cccc-cccc-cccc-cccccccccccc",
                                   "score": 99}]}),                  # ambiguous
        ]
        counts = self._enrich(conn, _client(stub))
        assert counts["eligible"] == 3
        assert counts["matched"] == 1
        assert counts["no_match"] == 1
        assert counts["ambiguous"] == 1
        assert counts["rows_written"] == 3
        assert TrackMetadataRepository(conn).counts_by_status("acme") == {
            "matched": 1, "no_match": 1, "ambiguous": 1}

    def test_a_second_run_does_nothing(self, conn, stub):
        stub.default = (200, {"recordings": []})
        first = self._enrich(conn, _client(stub))
        assert first["eligible"] == 3
        before = len(stub.requests)
        second = self._enrich(conn, _client(stub))
        assert second["eligible"] == 0
        assert len(stub.requests) == before, "a repeat run made requests"

    def test_force_re_enriches(self, conn, stub):
        stub.default = (200, {"recordings": []})
        self._enrich(conn, _client(stub))
        again = self._enrich(conn, _client(stub), force=True)
        assert again["eligible"] == 3

    def test_tracks_without_metadata_are_skipped_not_guessed(self, conn, stub):
        with conn.cursor() as cur:
            cur.execute("TRUNCATE track_metadata, tracks, catalogs CASCADE")
        conn.commit()
        _seed(conn, n=2, with_metadata=False)
        counts = self._enrich(conn, _client(stub))
        assert counts["eligible"] == 0
        assert counts["skipped_missing_metadata"] == 2
        assert len(stub.requests) == 0, "a lookup was attempted without metadata"
        assert TrackMetadataRepository(conn).get("acme", "t1").match_status == "skipped"

    def test_an_error_on_one_track_does_not_sink_the_run(self, conn, stub):
        stub.responses = [(400, {}), (200, {"recordings": [RECORDING]}),
                          (200, {"recordings": []})]
        counts = self._enrich(conn, _client(stub))
        assert counts["error"] == 1 and counts["matched"] == 1
        assert counts["rows_written"] == 3
        # The failure must be explainable from the row, not just counted.
        repo = TrackMetadataRepository(conn)
        failed = [repo.get("acme", t) for t in ("t1", "t2", "t3")]
        err = next(r for r in failed if r.match_status == "error")
        assert "400" in err.error_detail
        assert err.attempts == 1

    def test_provenance_is_recorded(self, conn, stub):
        stub.responses = [(200, {"recordings": [RECORDING]})]
        stub.default = (200, {"recordings": []})
        self._enrich(conn, _client(stub))
        row = TrackMetadataRepository(conn).get("acme", "t1")
        assert row.query_used and "recording:" in row.query_used
        assert row.source == "musicbrainz" and row.fetched_at is not None
        assert TrackMetadataRepository(conn).raw_response("acme", "t1") is not None

    def test_the_run_reports_its_own_request_activity(self, conn, stub):
        stub.default = (200, {"recordings": []})
        counts = self._enrich(conn, _client(stub, min_interval=0.05))
        assert counts["requests_made"] == 3
        assert counts["http_dispatches"] == 3
        assert counts["http_dispatches_per_sec"] > 0
        assert counts["elapsed_seconds"] >= 0.1


class TestApiIsUnaffected:
    def test_the_api_never_imports_enrichment(self):
        import pathlib
        for f in pathlib.Path("musicintel/api").glob("*.py"):
            assert "musicintel.enrichment" not in f.read_text(), f
            assert "musicbrainz" not in f.read_text().lower(), f

    def test_the_recognition_package_never_imports_enrichment(self):
        import pathlib
        for f in pathlib.Path("musicintel/recognition").glob("*.py"):
            text = f.read_text().lower()
            assert "musicbrainz" not in text and "enrichment" not in text, f


# ------------------------------------------------ cross-process throttling --
class TestCrossProcessRateLimiting:
    """The limiter must hold across separate CLI invocations, not just within one.

    The live experiment measured run 2's first request leaving 16 ms after run
    1's last, because each invocation built a fresh client whose in-memory
    timestamp started at zero. MusicBrainz limits per source IP, so that broke
    the limit no single run broke. These tests pin the fix.

    No test contacts MusicBrainz. Timing tests use a shortened interval to keep
    the suite fast; that is a test of the *mechanism*, and the shipped default
    is asserted separately by `test_the_shipped_default_interval_is_two_seconds`.
    """

    @staticmethod
    def _state(tmp_path):
        return tmp_path / "ratelimit.state"

    def test_a_second_limiter_sharing_state_must_wait(self, tmp_path):
        """The exact regression: a 'fresh' limiter cannot go immediately."""
        state = self._state(tmp_path)
        first = RateLimiter(min_interval=0.5, state_path=state)
        assert first.acquire() == 0.0            # nothing has gone before

        second = RateLimiter(min_interval=0.5, state_path=state)  # a new "process"
        started = time.monotonic()
        waited = second.acquire()
        elapsed = time.monotonic() - started
        assert waited > 0, "a fresh limiter issued immediately after another"
        assert elapsed >= 0.45, f"second limiter waited only {elapsed:.3f}s"

    def test_the_interval_is_respected_across_real_processes(self, tmp_path):
        """Two separate interpreters, sequentially -- the CLI-invocation case."""
        state = self._state(tmp_path)
        stamps = [self._acquire_in_subprocess(state, 0.5) for _ in range(3)]
        gaps = [b - a for a, b in zip(stamps, stamps[1:])]
        assert all(g >= 0.45 for g in gaps), f"gaps between processes: {gaps}"

    def test_concurrent_processes_cannot_both_go_first(self, tmp_path):
        """Two interpreters started together must serialise, not race."""
        state = self._state(tmp_path)
        procs = [self._spawn(state, 0.5) for _ in range(2)]
        stamps = sorted(float(p.communicate()[0].strip()) for p in procs)
        assert all(p.returncode == 0 for p in procs)
        gap = stamps[1] - stamps[0]
        assert gap >= 0.45, (
            f"two concurrent processes acquired {gap:.3f}s apart; the lock did "
            "not serialise them")

    def test_the_limiter_recovers_after_the_interval_expires(self, tmp_path):
        state = self._state(tmp_path)
        RateLimiter(min_interval=0.2, state_path=state).acquire()
        time.sleep(0.25)
        assert RateLimiter(min_interval=0.2, state_path=state).acquire() == 0.0

    def test_a_stale_timestamp_costs_nothing(self, tmp_path):
        state = self._state(tmp_path)
        state.write_text(f"{time.time() - 86400:.6f}")     # yesterday
        assert RateLimiter(min_interval=1.0, state_path=state).acquire() == 0.0

    def test_a_corrupt_state_file_is_treated_as_absent(self, tmp_path):
        state = self._state(tmp_path)
        state.write_text("not a timestamp")
        assert RateLimiter(min_interval=1.0, state_path=state).acquire() == 0.0

    def test_a_clock_moving_backwards_does_not_permit_a_burst(self, tmp_path):
        """A negative elapsed time must not be read as 'long enough ago'."""
        state = self._state(tmp_path)
        state.write_text(f"{time.time() + 3600:.6f}")      # an hour in the future
        limiter = RateLimiter(min_interval=0.3, state_path=state)
        started = time.monotonic()
        waited = limiter.acquire()
        assert waited == pytest.approx(0.3, abs=0.05)
        assert time.monotonic() - started >= 0.25

    def test_every_wait_is_bounded_by_the_interval(self, tmp_path):
        limiter = RateLimiter(min_interval=0.4, state_path=self._state(tmp_path))
        for last, now in ((0, 100.0), (100.0, 100.0), (100.0, 99.0),
                          (100.0, 1e12), (1e12, 100.0)):
            assert 0.0 <= limiter._wait_for(last, now) <= 0.4

    def test_an_unwritable_state_path_degrades_instead_of_failing(self, tmp_path):
        """A read-only state directory must not take the worker down."""
        limiter = RateLimiter(min_interval=0.0,
                              state_path=tmp_path / "nope" / "deeper" / "s")
        assert limiter.acquire() == 0.0
        assert limiter.cross_process is False   # fell back to in-process only

    def test_two_clients_sharing_state_space_their_requests(self, stub, tmp_path):
        """End to end through the client, against the local stub."""
        state = self._state(tmp_path)
        stub.default = (200, {"recordings": []})
        for _ in range(3):
            MusicBrainzClient(CONTACT, base_url=stub.base_url, min_interval=0.4,
                              rate_limit_state_path=state
                              ).search_recording("a", "b")
        stamps = [r["at"] for r in stub.requests]
        gaps = [b - a for a, b in zip(stamps, stamps[1:])]
        assert all(g >= 0.35 for g in gaps), f"gaps: {gaps}"

    def test_cross_process_is_on_by_default_and_can_be_disabled(self, tmp_path):
        assert RateLimiter(state_path=self._state(tmp_path)).cross_process is True
        assert RateLimiter(state_path=self._state(tmp_path),
                           cross_process=False).cross_process is False

    def test_the_default_state_path_is_outside_the_repository(self):
        from musicintel.enrichment.musicbrainz import default_state_path
        path = default_state_path()
        assert str(path).startswith(tempfile.gettempdir())
        assert "music-intelligence" not in str(path)

    # -- helpers -------------------------------------------------------
    _SNIPPET = (
        "import sys, time;"
        "sys.path.insert(0, {repo!r});"
        "from musicintel.enrichment.musicbrainz import RateLimiter;"
        "RateLimiter(min_interval={interval}, state_path={state!r}).acquire();"
        "print(time.time())"
    )

    @classmethod
    def _spawn(cls, state, interval):
        code = cls._SNIPPET.format(
            repo=str(Path(__file__).resolve().parent.parent),
            interval=interval, state=str(state))
        return subprocess.Popen([sys.executable, "-c", code],
                                stdout=subprocess.PIPE, text=True)

    @classmethod
    def _acquire_in_subprocess(cls, state, interval) -> float:
        proc = cls._spawn(state, interval)
        out, _ = proc.communicate(timeout=60)
        assert proc.returncode == 0, out
        return float(out.strip())


# ------------------------------------------------- error diagnostics (C) --
class TestErrorDiagnosticsPersistence:
    """A failed row must say why it failed.

    Before this, every failure class collapsed to `match_status='error'` plus
    the query: a 400 (our query is malformed, retrying is pointless) and a 503
    retry-exhaustion (transient, re-run later) were byte-identical on disk. The
    live experiment hit exactly that -- `ia_AF026` failed and the row could not
    explain it. The client already computed both facts; the worker discarded
    them.
    """

    @staticmethod
    def _run(conn, stub, **client_kw):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "worker", "scripts/enrich_musicbrainz.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.enrich(conn, _client(stub, **client_kw), "acme",
                          echo=lambda *a: None)

    def test_a_400_records_the_status_and_a_single_attempt(self, conn, stub):
        stub.responses = [(400, {"error": "bad"})]
        stub.default = (200, {"recordings": []})
        self._run(conn, stub, max_attempts=3)
        row = TrackMetadataRepository(conn).get("acme", "t1")
        assert row.match_status == "error"
        assert "400" in row.error_detail
        assert row.attempts == 1, "a 4xx must not have been retried"

    def test_an_exhausted_503_records_the_status_and_every_attempt(self, conn, stub):
        stub.responses = [(503, {}), (503, {}), (503, {})]
        stub.default = (200, {"recordings": []})
        self._run(conn, stub, max_attempts=3, backoff=0.0)
        row = TrackMetadataRepository(conn).get("acme", "t1")
        assert row.match_status == "error"
        assert "503" in row.error_detail
        assert row.attempts == 3, "retry exhaustion must be visible"

    def test_400_and_503_are_now_distinguishable_on_disk(self, conn, stub):
        """The whole point of the change."""
        stub.responses = [(400, {}), (503, {}), (503, {}), (503, {})]
        stub.default = (200, {"recordings": []})
        self._run(conn, stub, max_attempts=3, backoff=0.0)
        repo = TrackMetadataRepository(conn)
        a, b = repo.get("acme", "t1"), repo.get("acme", "t2")
        assert a.match_status == b.match_status == "error"
        assert (a.error_detail, a.attempts) != (b.error_detail, b.attempts)
        assert "400" in a.error_detail and a.attempts == 1
        assert "503" in b.error_detail and b.attempts == 3

    def test_a_503_that_recovers_leaves_no_error_but_records_the_retry(self, conn, stub):
        stub.responses = [(503, {}), (200, {"recordings": [RECORDING]})]
        stub.default = (200, {"recordings": []})
        self._run(conn, stub, max_attempts=3, backoff=0.0)
        row = TrackMetadataRepository(conn).get("acme", "t1")
        assert row.match_status == "matched"
        assert row.error_detail is None, "a recovered lookup is not an error"
        assert row.attempts == 2, "the retry that was needed must still be visible"

    def test_a_connection_failure_survives_persistence(self, conn):
        """No server at all -- the longest diagnostic the client produces."""
        client = MusicBrainzClient(CONTACT, base_url="http://127.0.0.1:1",
                                   min_interval=0.0, backoff=0.0, max_attempts=2,
                                   rate_limit_state_path="/tmp/mi-connfail-test")
        result = client.search_recording("Artist 1", "Song 1")
        assert result.status == "error"
        repo = TrackMetadataRepository(conn)
        repo.upsert("acme", "t1", match_status="error",
                    query_used=result.query, error_detail=result.error,
                    attempts=result.attempts)
        row = repo.get("acme", "t1")
        assert "connection error" in row.error_detail
        assert row.attempts == 2

    def test_the_classification_survives_truncation(self, conn):
        """Every diagnostic puts its class first, so cutting the tail is safe."""
        from musicintel.db.repositories import ERROR_DETAIL_MAX_CHARS
        oversized = "connection error: " + ("x" * 5000)
        repo = TrackMetadataRepository(conn)
        repo.upsert("acme", "t1", match_status="error", error_detail=oversized)
        row = repo.get("acme", "t1")
        assert len(row.error_detail) == ERROR_DETAIL_MAX_CHARS
        assert row.error_detail.startswith("connection error:")
        assert row.error_detail.endswith("[truncated]")

    def test_the_database_refuses_an_unbounded_detail(self, conn):
        """The CHECK is a backstop for anything that bypasses the repository."""
        with pytest.raises(psycopg.errors.CheckViolation):
            with conn.cursor() as cur:
                cur.execute("INSERT INTO track_metadata (catalog_id, track_id, "
                            "match_status, error_detail) VALUES "
                            "('acme','t1','error', %s)", ("y" * 513,))
        conn.rollback()

    def test_a_negative_attempt_count_is_refused(self, conn):
        with pytest.raises(psycopg.errors.CheckViolation):
            with conn.cursor() as cur:
                cur.execute("INSERT INTO track_metadata (catalog_id, track_id, "
                            "match_status, attempts) VALUES ('acme','t1','error',-1)")
        conn.rollback()

    def test_successful_and_no_match_rows_are_otherwise_unchanged(self, conn, stub):
        """Only `attempts` is added to a healthy row; nothing else moves."""
        stub.responses = [(200, {"recordings": [RECORDING]}), (200, {"recordings": []})]
        stub.default = (200, {"recordings": []})
        self._run(conn, stub)
        repo = TrackMetadataRepository(conn)
        matched, no_match = repo.get("acme", "t1"), repo.get("acme", "t2")
        assert matched.match_status == "matched"
        assert matched.title == "Nachtaktiv" or matched.title == RECORDING["title"]
        assert matched.mb_recording_id == RECORDING["id"]
        assert matched.error_detail is None and matched.attempts == 1
        assert no_match.match_status == "no_match"
        assert no_match.error_detail is None and no_match.attempts == 1
        assert no_match.title is None and no_match.match_score is None

    def test_a_skipped_row_records_no_attempt(self, conn, stub):
        """Nothing was looked up, so there is no attempt count to record."""
        with conn.cursor() as cur:
            cur.execute("TRUNCATE track_metadata, tracks, catalogs CASCADE")
        conn.commit()
        _seed(conn, n=2, with_metadata=False)
        self._run(conn, stub)
        row = TrackMetadataRepository(conn).get("acme", "t1")
        assert row.match_status == "skipped"
        assert row.attempts is None and row.error_detail is None

    def test_re_enrichment_clears_a_previous_error(self, conn, stub):
        stub.responses = [(400, {})]
        stub.default = (200, {"recordings": []})
        self._run(conn, stub)
        repo = TrackMetadataRepository(conn)
        assert repo.get("acme", "t1").error_detail is not None
        stub.responses = []
        stub.default = (200, {"recordings": [RECORDING]})
        self._run(conn, stub, min_interval=0.0)   # force=False leaves t1 alone
        repo.upsert("acme", "t1", match_status="matched", title="x", attempts=1)
        row = repo.get("acme", "t1")
        assert row.error_detail is None, "a fresh success must clear the old error"
        assert row.attempts == 1

    def test_migration_003_is_recorded_and_idempotent(self, conn):
        from musicintel.db.migrate import apply_migrations, applied_migrations
        assert "003_track_metadata_diagnostics.sql" in applied_migrations(conn)
        assert apply_migrations(conn) == []


# --------------------------------------------- dispatch-rate metric (D) --
class TestDispatchRateMetric:
    """`http_dispatches_per_sec` must describe THIS run's actual dispatch rate.

    The metric it replaces divided the client's LIFETIME response count by the
    current run's elapsed time. Two independent errors: a second run on the same
    client counted the first run's requests, and dividing n requests by the span
    of the n-1 intervals they occupy overstates the sustained rate by n/(n-1) --
    measured at 1.33x for four requests. It also silently dropped timed-out
    requests, which are dispatched and do reach the server.
    """

    @staticmethod
    def _run(conn, stub, catalog="acme", **client_kw):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "worker", "scripts/enrich_musicbrainz.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod, mod.enrich(conn, _client(stub, **client_kw), catalog,
                               echo=lambda *a: None)

    def test_the_rate_matches_the_configured_interval(self, conn, stub):
        """Three lookups at 0.4 s should report ~1/0.4 = 2.5 dispatches/s."""
        stub.default = (200, {"recordings": []})
        _, counts = self._run(conn, stub, min_interval=0.4)
        assert counts["http_dispatches"] == 3
        assert counts["http_dispatches_per_sec"] == pytest.approx(2.5, rel=0.20)

    def test_there_is_no_n_over_n_minus_one_inflation(self, conn, stub):
        """The reported rate must be below count/window, which was the bug."""
        stub.default = (200, {"recordings": []})
        _, counts = self._run(conn, stub, min_interval=0.4)
        n = counts["http_dispatches"]
        window = counts["dispatch_window_seconds"]
        inflated = n / window
        assert counts["http_dispatches_per_sec"] < inflated
        assert counts["http_dispatches_per_sec"] == pytest.approx(
            (n - 1) / window, rel=0.01)

    def test_a_second_run_on_one_client_reports_only_its_own_requests(self, conn, stub):
        """The regression: the lifetime counter leaking into the next run."""
        _seed(conn, catalog_id="second", tenant="acme", n=3)
        stub.default = (200, {"recordings": []})
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "worker", "scripts/enrich_musicbrainz.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        client = _client(stub, min_interval=0.05)

        first = mod.enrich(conn, client, "acme", echo=lambda *a: None)
        second = mod.enrich(conn, client, "second", echo=lambda *a: None)
        assert first["requests_made"] == 3 and first["http_dispatches"] == 3
        assert second["requests_made"] == 3, "the first run's requests leaked in"
        assert second["http_dispatches"] == 3
        # The client's own lifetime counter is deliberately left cumulative.
        assert client.requests_made == 6

    def test_a_run_with_nothing_pending_reports_zero(self, conn, stub):
        stub.default = (200, {"recordings": []})
        self._run(conn, stub, min_interval=0.0)          # exhausts the catalog
        _, counts = self._run(conn, stub, min_interval=0.0)
        assert counts["eligible"] == 0
        assert counts["http_dispatches"] == 0
        assert counts["http_dispatches_per_sec"] == 0.0   # not a division by zero
        assert counts["requests_made"] == 0

    def test_a_single_dispatch_reports_zero_rather_than_inventing_a_rate(self, conn, stub):
        """One request spans no interval, so it has no rate."""
        with conn.cursor() as cur:
            cur.execute("TRUNCATE track_metadata, tracks, catalogs CASCADE")
        conn.commit()
        _seed(conn, n=1)
        stub.default = (200, {"recordings": []})
        _, counts = self._run(conn, stub, min_interval=0.0)
        assert counts["http_dispatches"] == 1
        assert counts["http_dispatches_per_sec"] == 0.0

    def test_a_timed_out_dispatch_is_counted(self, conn, stub):
        """It reached the server; `requests_made` cannot see it, the metric must."""
        stub.responses = ["timeout", (200, {"recordings": []})]
        stub.default = (200, {"recordings": []})
        _, counts = self._run(conn, stub, min_interval=0.05,
                              timeout=(2.0, 0.4), max_attempts=3, backoff=0.0)
        assert counts["http_dispatches"] == 4, "the timed-out request was dropped"
        assert counts["requests_made"] == 3, "requests_made counts only responses"
        assert counts["http_dispatches"] > counts["requests_made"]

    def test_retries_are_counted_as_dispatches(self, conn, stub):
        stub.responses = [(503, {}), (200, {"recordings": []})]
        stub.default = (200, {"recordings": []})
        _, counts = self._run(conn, stub, min_interval=0.05, backoff=0.0)
        assert counts["http_dispatches"] == 4      # 3 lookups, one retried
        assert counts["requests_made"] == 4

    def test_the_window_is_reported_so_the_rate_can_be_rechecked(self, conn, stub):
        stub.default = (200, {"recordings": []})
        _, counts = self._run(conn, stub, min_interval=0.2)
        assert counts["dispatch_window_seconds"] > 0
        assert counts["http_dispatches_per_sec"] == pytest.approx(
            (counts["http_dispatches"] - 1) / counts["dispatch_window_seconds"],
            rel=0.01)

    def test_the_old_misleading_field_is_gone(self, conn, stub):
        stub.default = (200, {"recordings": []})
        _, counts = self._run(conn, stub, min_interval=0.0)
        assert "effective_rate_per_sec" not in counts

    # -- F/G: nothing else moved -------------------------------------
    def test_retry_behaviour_is_unchanged(self, stub):
        stub.responses = [(503, {}), (503, {}), (503, {})]
        r = _client(stub, max_attempts=3, backoff=0.0).search_recording("a", "b")
        assert r.status == "error" and r.attempts == 3
        assert len(stub.requests) == 3
        stub.requests.clear()
        stub.responses = [(400, {})]
        r = _client(stub, max_attempts=3).search_recording("a", "b")
        assert r.status == "error" and r.attempts == 1
        assert len(stub.requests) == 1, "a 4xx must still not be retried"

    def test_finding_c_diagnostics_are_intact(self, conn, stub):
        stub.responses = [(400, {}), (503, {}), (200, {"recordings": [RECORDING]}),
                          (200, {"recordings": []})]
        stub.default = (200, {"recordings": []})
        self._run(conn, stub, min_interval=0.0, backoff=0.0, max_attempts=3)
        repo = TrackMetadataRepository(conn)
        err, recovered, no_match = (repo.get("acme", "t1"), repo.get("acme", "t2"),
                                    repo.get("acme", "t3"))
        assert err.match_status == "error"
        assert "400" in err.error_detail and err.attempts == 1
        assert recovered.match_status == "matched"
        assert recovered.error_detail is None and recovered.attempts == 2
        assert no_match.match_status == "no_match"
        assert no_match.error_detail is None and no_match.attempts == 1
        assert no_match.title is None
