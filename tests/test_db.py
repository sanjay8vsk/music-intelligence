"""Stage 2 PostgreSQL persistence.

These run against a real PostgreSQL server (`pgserver` ships one), not a mock
and not SQLite. The schema leans on CHECK constraints, array columns, partial
indexes and `ON CONFLICT ... DO UPDATE` arithmetic; none of that is meaningfully
exercised by a substitute engine, and the constraints are the point — they are
what stops a bad row rather than a code path that might be skipped.

What is deliberately NOT tested here: recognition. Nothing in this package
touches fingerprints, and the tables hold no postings.
"""

from __future__ import annotations

import datetime as dt

import pytest

psycopg = pytest.importorskip("psycopg")
pgserver = pytest.importorskip("pgserver")

from musicintel.catalog.models import Catalog, CatalogTrack       # noqa: E402
from musicintel.db.migrate import apply_migrations, applied_migrations  # noqa: E402
from musicintel.db.pool import (                                   # noqa: E402
    DatabaseUnavailable, close_pool, connection, open_pool,
)
from musicintel.db.repositories import (                           # noqa: E402
    ApiKeyRepository, CatalogRepository, UsageRepository,
)
from musicintel.db.usage_writer import UsageWriter                 # noqa: E402

DAY = dt.date(2026, 8, 28)
TABLES = ("usage", "tracks", "catalogs", "api_keys")


@pytest.fixture(scope="session")
def dsn(tmp_path_factory):
    server = pgserver.get_server(tmp_path_factory.mktemp("pg") / "data")
    uri = server.get_uri()
    with psycopg.connect(uri) as conn:
        apply_migrations(conn)
    yield uri
    server.cleanup()


@pytest.fixture
def conn(dsn):
    """A connection onto an empty schema."""
    with psycopg.connect(dsn) as c:
        with c.cursor() as cur:
            cur.execute(f"TRUNCATE {', '.join(TABLES)} CASCADE")
        c.commit()
        yield c


def _catalog(catalog_id="acme", n=2, offset=0):
    return Catalog(catalog_id=catalog_id, tracks=[
        CatalogTrack(track_id=f"t{i}", source_path=f"/audio/{catalog_id}/{i}.wav",
                     sha256=f"{i + offset:064x}", duration_sec=10.0 + i,
                     bytes=1000 * (i + 1), fingerprint_count=100 * (i + 1),
                     title=f"Track {i}", artist="Someone")
        for i in range(1, n + 1)
    ])


ARTIFACT = {"artifact_version": 1, "fingerprint_count": 4242,
            "index_content_hash": "f" * 64, "fingerprint_format_version": 1,
            "index_format_version": 1}


# ------------------------------------------------------------ migrations --
class TestMigrations:
    def test_schema_is_recorded_and_reapplying_is_a_no_op(self, conn):
        recorded = applied_migrations(conn)
        assert "001_stage2_identity_and_usage.sql" in recorded
        assert apply_migrations(conn) == []

    def test_all_four_roadmap_tables_exist(self, conn):
        with conn.cursor() as cur:
            cur.execute("SELECT table_name FROM information_schema.tables "
                        "WHERE table_schema = 'public'")
            names = {r[0] for r in cur.fetchall()}
        assert {"tracks", "catalogs", "api_keys", "usage"} <= names

    def test_editing_an_applied_migration_is_refused(self, conn, tmp_path):
        """Silent drift between environments is how schemas diverge."""
        (tmp_path / "001_stage2_identity_and_usage.sql").write_text("SELECT 1")
        with pytest.raises(RuntimeError, match="changed after it was applied"):
            apply_migrations(conn, directory=tmp_path)


# -------------------------------------------------------------- catalogs --
class TestCatalogPersistence:
    def test_catalog_and_tracks_round_trip(self, conn):
        repo = CatalogRepository(conn)
        catalog = _catalog(n=3)
        repo.sync(catalog, tenant="acme", artifact=ARTIFACT)

        row = repo.get("acme")
        assert row["tenant"] == "acme"
        assert row["track_count"] == 3
        assert row["content_hash"] == catalog.content_hash()
        assert row["fingerprint_count"] == 4242

        tracks = repo.tracks("acme")
        assert [t.track_id for t in tracks] == ["t1", "t2", "t3"]
        assert tracks[0].title == "Track 1" and tracks[0].artist == "Someone"
        assert tracks[0].duration_sec == 11.0

    def test_resync_updates_and_removes_departed_tracks(self, conn):
        repo = CatalogRepository(conn)
        repo.sync(_catalog(n=3), tenant="acme", artifact=ARTIFACT)
        repo.sync(_catalog(n=1), tenant="acme", artifact=ARTIFACT)
        assert [t.track_id for t in repo.tracks("acme")] == ["t1"]
        assert repo.get("acme")["track_count"] == 1

    def test_content_hash_matches_the_catalog_object(self, conn):
        """The stored hash must be the catalog's own identity, not a new one."""
        catalog = _catalog(n=2)
        CatalogRepository(conn).sync(catalog, tenant="acme", artifact=ARTIFACT)
        assert CatalogRepository(conn).get("acme")["content_hash"] == \
            catalog.content_hash()

    def test_duplicate_audio_in_one_catalog_is_refused_by_the_database(self, conn):
        """Content-hash dedup as an invariant, not just an ingest convention."""
        dupe = Catalog(catalog_id="acme", tracks=[
            CatalogTrack("a", "/1.wav", "1" * 64, 10.0),
            CatalogTrack("b", "/2.wav", "1" * 64, 10.0),   # same audio, new id
        ])
        with pytest.raises(psycopg.errors.UniqueViolation):
            CatalogRepository(conn).sync(dupe, tenant="acme")
        conn.rollback()

    @pytest.mark.parametrize("bad", ["../etc", "has space", ".hidden", "a" * 65])
    def test_catalog_ids_the_store_would_refuse_are_refused_here_too(self, conn, bad):
        with pytest.raises(psycopg.errors.CheckViolation):
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO catalogs (catalog_id, tenant, content_hash) "
                    "VALUES (%s, 'acme', %s)", (bad, "0" * 64))
        conn.rollback()

    def test_deleting_a_catalog_removes_its_tracks(self, conn):
        repo = CatalogRepository(conn)
        repo.sync(_catalog(n=2), tenant="acme", artifact=ARTIFACT)
        assert repo.delete("acme") is True
        assert repo.tracks("acme") == []


# ---------------------------------------------------------------- tenancy --
class TestIsolation:
    def test_catalogs_are_listed_only_for_their_owning_tenant(self, conn):
        repo = CatalogRepository(conn)
        repo.sync(_catalog("acme", n=2), tenant="acme", artifact=ARTIFACT)
        repo.sync(_catalog("globex", n=3, offset=100), tenant="globex",
                  artifact=ARTIFACT)
        assert repo.list_for_tenant("acme") == ["acme"]
        assert repo.list_for_tenant("globex") == ["globex"]
        assert repo.list_for_tenant("nobody") == []

    def test_tracks_never_leak_across_catalogs(self, conn):
        repo = CatalogRepository(conn)
        repo.sync(_catalog("acme", n=2), tenant="acme", artifact=ARTIFACT)
        repo.sync(_catalog("globex", n=3, offset=100), tenant="globex",
                  artifact=ARTIFACT)
        assert len(repo.tracks("acme")) == 2
        assert len(repo.tracks("globex")) == 3
        acme_hashes = {t.sha256 for t in repo.tracks("acme")}
        globex_hashes = {t.sha256 for t in repo.tracks("globex")}
        assert acme_hashes.isdisjoint(globex_hashes)

    def test_the_same_recording_may_belong_to_two_tenants(self, conn):
        """Licensing the same track twice is legitimate; the unique constraint
        is per catalog, not global."""
        repo = CatalogRepository(conn)
        shared = CatalogTrack("t1", "/s.wav", "9" * 64, 10.0)
        repo.sync(Catalog(catalog_id="acme", tracks=[shared]), tenant="acme")
        repo.sync(Catalog(catalog_id="globex", tracks=[shared]), tenant="globex")
        with conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM tracks WHERE sha256 = %s", ("9" * 64,))
            assert cur.fetchone()[0] == 2

    def test_usage_is_isolated_between_tenants(self, conn):
        repo = UsageRepository(conn)
        repo.record("acme", "k1", DAY, audio_seconds=10.0)
        repo.record("globex", "k2", DAY, audio_seconds=99.0)
        assert repo.tenant_total("acme", DAY) == 10.0
        assert repo.tenant_total("globex", DAY) == 99.0


# --------------------------------------------------------------- api_keys --
class TestApiKeyPersistence:
    RECORD = {"key_id": "k_acme", "tenant": "acme", "key_sha256": "a" * 64,
              "catalogs": ["acme"], "scopes": ["identify"],
              "rate_limit_per_minute": 120, "rate_limit_burst": 40,
              "audio_seconds_per_day": 7200}

    def test_record_shape_matches_what_the_registry_consumes(self, conn):
        """The whole point: a repository swap, not a redesign."""
        from musicintel.api.auth import ApiKeyRegistry

        ApiKeyRepository(conn).upsert(self.RECORD)
        records = ApiKeyRepository(conn).load_records()
        registry = ApiKeyRegistry(records)          # must accept it unchanged
        assert len(registry) == 1

    def test_authorization_semantics_survive_the_round_trip(self, conn):
        from musicintel.api.auth import ApiKeyRegistry, hash_key

        raw = "sk_test_roundtrip_0000000000000"
        ApiKeyRepository(conn).upsert({**self.RECORD, "key_sha256": hash_key(raw)})
        principal = ApiKeyRegistry(ApiKeyRepository(conn).load_records()).resolve(raw)
        assert principal is not None
        assert principal.tenant == "acme"
        assert principal.may_access("acme") and not principal.may_access("globex")
        assert principal.has_scope("identify")
        assert not principal.has_scope("catalogs:read")
        assert principal.rate_limit_per_minute == 120
        assert principal.burst == 40
        assert principal.audio_seconds_per_day == 7200

    def test_empty_catalog_list_means_every_catalog(self, conn):
        from musicintel.api.auth import ApiKeyRegistry, hash_key

        raw = "sk_test_unrestricted_00000000"
        ApiKeyRepository(conn).upsert(
            {**self.RECORD, "key_id": "k_all", "catalogs": [],
             "key_sha256": hash_key(raw)})
        p = ApiKeyRegistry(ApiKeyRepository(conn).load_records()).resolve(raw)
        assert p.may_access("anything")

    def test_revocation_hides_the_key(self, conn):
        repo = ApiKeyRepository(conn)
        repo.upsert(self.RECORD)
        assert repo.revoke("k_acme") is True
        assert repo.load_records() == []
        assert len(repo.load_records(include_inactive=True)) == 1

    def test_no_raw_key_column_exists(self, conn):
        with conn.cursor() as cur:
            cur.execute("SELECT column_name FROM information_schema.columns "
                        "WHERE table_name = 'api_keys'")
            columns = {r[0] for r in cur.fetchall()}
        assert "key" not in columns and "secret" not in columns
        assert "key_sha256" in columns

    def test_a_record_carrying_a_raw_key_is_refused(self, conn):
        with pytest.raises(ValueError, match="raw key"):
            ApiKeyRepository(conn).upsert({**self.RECORD, "key": "sk_live_oops"})

    def test_duplicate_digest_is_refused(self, conn):
        repo = ApiKeyRepository(conn)
        repo.upsert(self.RECORD)
        with pytest.raises(psycopg.errors.UniqueViolation):
            repo.upsert({**self.RECORD, "key_id": "k_other"})
        conn.rollback()

    def test_a_non_digest_is_refused_by_the_database(self, conn):
        with pytest.raises(psycopg.errors.CheckViolation):
            with conn.cursor() as cur:
                cur.execute("INSERT INTO api_keys (key_id, tenant, key_sha256) "
                            "VALUES ('k', 'acme', %s)", ("NOT-A-DIGEST" + "0" * 52,))
        conn.rollback()


# ------------------------------------------------------------------ usage --
class TestUsagePersistence:
    def test_a_single_record_is_stored(self, conn):
        UsageRepository(conn).record("acme", "k1", DAY, audio_seconds=5.0,
                                     matches=1)
        row = UsageRepository(conn).get("acme", "k1", DAY)
        assert row.audio_seconds == 5.0
        assert row.request_count == 1 and row.match_count == 1

    def test_repeated_updates_accumulate(self, conn):
        repo = UsageRepository(conn)
        for _ in range(10):
            repo.record("acme", "k1", DAY, audio_seconds=5.0, matches=1)
        row = repo.get("acme", "k1", DAY)
        assert row.audio_seconds == 50.0
        assert row.request_count == 10 and row.match_count == 10

    def test_fractional_seconds_do_not_drift(self, conn):
        """numeric, not float: this is money-adjacent and accumulates."""
        repo = UsageRepository(conn)
        for _ in range(300):
            repo.record("acme", "k1", DAY, audio_seconds=0.1)
        assert repo.get("acme", "k1", DAY).audio_seconds == 30.0

    def test_days_are_separate_rows(self, conn):
        repo = UsageRepository(conn)
        repo.record("acme", "k1", DAY, audio_seconds=5.0)
        repo.record("acme", "k1", DAY + dt.timedelta(days=1), audio_seconds=7.0)
        assert repo.tenant_total("acme", DAY) == 5.0
        assert len(repo.history("acme")) == 2

    def test_keys_of_one_tenant_are_summed(self, conn):
        repo = UsageRepository(conn)
        repo.record("acme", "k1", DAY, audio_seconds=5.0)
        repo.record("acme", "k2", DAY, audio_seconds=6.0)
        assert repo.tenant_total("acme", DAY) == 11.0

    def test_batch_write_aggregates_in_one_transaction(self, conn):
        repo = UsageRepository(conn)
        assert repo.record_many([
            ("acme", "k1", DAY, 1.5, 1, 1, 0),
            ("acme", "k1", DAY, 2.5, 1, 0, 1),
        ]) == 2
        row = repo.get("acme", "k1", DAY)
        assert row.audio_seconds == 4.0
        assert row.match_count == 1 and row.no_match_count == 1

    def test_negative_usage_is_refused(self, conn):
        with pytest.raises(psycopg.errors.CheckViolation):
            with conn.cursor() as cur:
                cur.execute("INSERT INTO usage (tenant, key_id, usage_day, "
                            "audio_seconds) VALUES ('a','k',%s,-1)", (DAY,))
        conn.rollback()

    def test_history_survives_key_deletion(self, conn):
        """Billing history outlives credentials -- deliberately no cascade."""
        ApiKeyRepository(conn).upsert(
            {"key_id": "k1", "tenant": "acme", "key_sha256": "b" * 64})
        UsageRepository(conn).record("acme", "k1", DAY, audio_seconds=12.0)
        with conn.cursor() as cur:
            cur.execute("DELETE FROM api_keys WHERE key_id = 'k1'")
        conn.commit()
        assert UsageRepository(conn).get("acme", "k1", DAY).audio_seconds == 12.0


# ------------------------------------------------------- redis vs postgres --
class TestDurabilityAcrossRedis:
    def test_usage_history_survives_a_full_redis_flush(self, conn):
        """The reason this table exists.

        Redis is the operational limiter and its counters carry a two-day TTL.
        Flushing it -- a restart, an eviction, a failover to a cold replica --
        resets what the limiter knows. It must not reset what was consumed.
        """
        fakeredis = pytest.importorskip("fakeredis")
        import asyncio

        from musicintel.api.ratelimit import RateLimiter

        redis = fakeredis.aioredis.FakeRedis(decode_responses=False)
        limiter = RateLimiter(redis)
        usage = UsageRepository(conn)

        async def consume(seconds: float):
            await limiter.consume_audio_seconds("acme", seconds=seconds,
                                                daily_limit=1000)
            usage.record("acme", "k1", DAY, audio_seconds=seconds)

        async def redis_view() -> float:
            return (await limiter.peek_audio_seconds("acme", daily_limit=1000)).used

        async def scenario():
            for _ in range(4):
                await consume(5.0)
            before_redis = await redis_view()
            await redis.flushall()                      # the outage
            after_redis = await redis_view()
            return before_redis, after_redis

        before_redis, after_redis = asyncio.run(scenario())

        assert before_redis == 20.0
        assert after_redis == 0.0, "the limiter should have forgotten"
        assert usage.tenant_total("acme", DAY) == 20.0, \
            "the durable record must not have forgotten"

    def test_limiter_and_ledger_track_the_same_quantity(self, conn):
        """Not a claim that they are the same store -- that they agree."""
        fakeredis = pytest.importorskip("fakeredis")
        import asyncio

        from musicintel.api.ratelimit import RateLimiter

        limiter = RateLimiter(fakeredis.aioredis.FakeRedis(decode_responses=False))
        usage = UsageRepository(conn)

        async def scenario():
            for seconds in (1.5, 2.25, 3.0):
                await limiter.consume_audio_seconds("acme", seconds=seconds,
                                                    daily_limit=1000)
                usage.record("acme", "k1", DAY, audio_seconds=seconds)
            return (await limiter.peek_audio_seconds("acme", daily_limit=1000)).used

        assert asyncio.run(scenario()) == pytest.approx(6.75)
        assert usage.tenant_total("acme", DAY) == pytest.approx(6.75)


# ----------------------------------------------------------- write-behind --
class TestUsageWriter:
    def test_records_reach_postgres(self, dsn, conn):
        open_pool(dsn)
        try:
            writer = UsageWriter(flush_interval=0.05)
            writer.start()
            for _ in range(5):
                writer.record("acme", "k1", audio_seconds=2.0, matched=True,
                              when=dt.datetime(2026, 8, 28, 12, tzinfo=dt.timezone.utc))
            writer.stop()
        finally:
            close_pool()
        row = UsageRepository(conn).get("acme", "k1", DAY)
        assert row.audio_seconds == 10.0
        assert row.request_count == 5 and row.match_count == 5

    def test_a_batch_is_aggregated_before_writing(self, dsn, conn):
        """A thousand requests must not become a thousand round trips."""
        open_pool(dsn)
        try:
            writer = UsageWriter(flush_interval=5.0, batch_size=1000)
            when = dt.datetime(2026, 8, 28, 12, tzinfo=dt.timezone.utc)
            for _ in range(200):
                writer.record("acme", "k1", audio_seconds=0.5, matched=False,
                              when=when)
            assert writer._drain_once(final=True) == 1     # one aggregated row
        finally:
            close_pool()
        row = UsageRepository(conn).get("acme", "k1", DAY)
        assert row.audio_seconds == 100.0 and row.request_count == 200

    def test_record_never_blocks_or_raises_when_the_queue_is_full(self):
        writer = UsageWriter(max_queue=3)
        results = [writer.record("acme", "k1", audio_seconds=1.0, matched=True)
                   for _ in range(6)]
        assert results == [True, True, True, False, False, False]
        assert writer.dropped == 3          # counted, never raised

    def test_records_are_retained_while_postgres_is_unavailable(self, conn):
        """A database outage must not silently discard billing history."""
        close_pool()                        # no pool at all
        writer = UsageWriter(flush_interval=0.05)
        writer.record("acme", "k1", audio_seconds=4.0, matched=True,
                      when=dt.datetime(2026, 8, 28, 12, tzinfo=dt.timezone.utc))
        assert writer._drain_once() is None          # reported as a failure
        assert writer.failures == 1
        assert writer.dropped == 0                   # and nothing thrown away
        assert writer._q.qsize() == 1                # still queued for retry

    def test_queued_records_flush_once_postgres_returns(self, dsn, conn):
        close_pool()
        writer = UsageWriter(flush_interval=0.05)
        writer.record("acme", "k1", audio_seconds=8.0, matched=True,
                      when=dt.datetime(2026, 8, 28, 12, tzinfo=dt.timezone.utc))
        assert writer._drain_once() is None
        open_pool(dsn)                               # the database comes back
        try:
            assert writer._drain_once() == 1
        finally:
            close_pool()
        assert UsageRepository(conn).get("acme", "k1", DAY).audio_seconds == 8.0


# ---------------------------------------------------- failure when down --
class TestDatabaseUnavailable:
    def test_connecting_to_a_dead_server_raises_one_known_exception(self):
        with pytest.raises(DatabaseUnavailable):
            with connection("postgresql://127.0.0.1:1/nonexistent?connect_timeout=1"):
                pass

    def test_using_the_pool_before_it_is_open_raises(self):
        close_pool()
        with pytest.raises(DatabaseUnavailable, match="pool is not open"):
            with connection():
                pass

    def test_key_loading_fails_loudly_rather_than_authenticating_nobody(self):
        """Silently returning zero keys would look like mass revocation."""
        from musicintel.api.app import _load_key_records
        from musicintel.api.config import Settings
        from musicintel.api.logging import get_logger

        close_pool()
        settings = Settings(database_url="postgresql://127.0.0.1:1/x",
                            api_keys_source="database", log_json=False)
        with pytest.raises(RuntimeError, match="unreachable"):
            _load_key_records(settings, get_logger())


# ------------------------------------------------- the service, end to end --
class TestApiWithPersistence:
    """The whole path: keys read from Postgres, usage written back to it.

    Proves the two halves actually meet — a repository that round-trips in
    isolation but is never reached by the service would pass every test above.
    """

    @staticmethod
    def _world(tmp_path):
        import numpy as np
        import soundfile as sf

        from musicintel.catalog.ingest import build_catalog_index, ingest_directory
        from musicintel.catalog.store import CatalogStore

        sr = 11025
        audio = tmp_path / "audio"
        audio.mkdir(parents=True, exist_ok=True)
        clips = {}
        for seed in (1, 2):
            rng = np.random.default_rng(seed)
            t = np.linspace(0, 12.0, sr * 12, endpoint=False)
            y = (0.5 * np.sin(2 * np.pi * (440 + 7 * seed) * t)
                 + 0.3 * np.sin(2 * np.pi * 900 * t)
                 + 0.02 * rng.standard_normal(t.size)).astype(np.float32)
            sf.write(audio / f"acme_{seed}.wav", y, sr, subtype="PCM_16")
            clips[seed] = y
        store = CatalogStore(tmp_path / "store")
        r = ingest_directory(audio)
        store.save(r.catalog, build_catalog_index(r.catalog, r.fingerprints),
                   catalog_id="acme")
        return store, clips, sr

    def test_keys_come_from_postgres_and_usage_lands_in_it(self, dsn, conn, tmp_path):
        import io

        import fakeredis.aioredis
        import soundfile as sf
        from fastapi.testclient import TestClient

        from musicintel.api.app import create_app
        from musicintel.api.auth import hash_key
        from musicintel.api.config import Settings

        store, clips, sr = self._world(tmp_path)
        raw = "sk_test_persisted_00000000000"
        ApiKeyRepository(conn).upsert({
            "key_id": "k_pg", "tenant": "acme", "key_sha256": hash_key(raw),
            "catalogs": ["acme"], "rate_limit_per_minute": 600,
            "rate_limit_burst": 200, "audio_seconds_per_day": 100000,
        })

        settings = Settings(catalog_root=store.root, database_url=dsn,
                            environment="test", log_json=False,
                            usage_flush_seconds=0.05)
        assert settings.resolved_api_keys_source == "database"

        clip = clips[2][sr * 3: sr * 8]          # 5 s from a catalogued track
        buf = io.BytesIO()
        sf.write(buf, clip, sr, format="WAV", subtype="PCM_16")
        payload = buf.getvalue()

        app = create_app(settings)
        app.state.redis = fakeredis.aioredis.FakeRedis(decode_responses=False)
        try:
            with TestClient(app) as client:
                # No JSON key file was configured at all: this can only work if
                # the key was read from Postgres.
                assert client.get("/v1/catalogs", headers={
                    "Authorization": "Bearer nope"}).status_code == 401
                r = client.post("/v1/identify",
                                files={"file": ("q.wav", payload, "audio/wav")},
                                data={"catalog_id": "acme"},
                                headers={"Authorization": f"Bearer {raw}"})
                assert r.status_code == 200, r.text
                body = r.json()
                assert body["decision"] == "match"
                assert body["match"]["track_id"] == "acme_2"
                app.state.usage_writer.wait_idle()
        finally:
            close_pool()

        today = dt.datetime.now(dt.timezone.utc).date()
        row = UsageRepository(conn).get("acme", "k_pg", today)
        assert row is not None, "no durable usage row was written"
        assert row.request_count == 1
        assert row.match_count == 1
        assert row.audio_seconds == pytest.approx(5.0, abs=0.05)

    def test_without_a_database_the_service_behaves_exactly_as_stage_3(self, tmp_path):
        """No DSN -> configuration keys, no writer, no persistence."""
        import json

        import fakeredis.aioredis
        from fastapi.testclient import TestClient

        from musicintel.api.app import create_app
        from musicintel.api.auth import hash_key
        from musicintel.api.config import Settings

        store, _clips, _sr = self._world(tmp_path)
        raw = "sk_test_configonly_0000000000"
        keys = tmp_path / "keys.json"
        keys.write_text(json.dumps([{
            "key_id": "k_cfg", "tenant": "acme", "key_sha256": hash_key(raw),
            "catalogs": ["acme"],
        }]))

        settings = Settings(catalog_root=store.root, api_keys_file=keys,
                            environment="test", log_json=False)
        assert settings.resolved_api_keys_source == "config"
        assert settings.persistence_enabled is False

        app = create_app(settings)
        app.state.redis = fakeredis.aioredis.FakeRedis(decode_responses=False)
        with TestClient(app) as client:
            assert app.state.usage_writer is None
            r = client.get("/v1/catalogs", headers={"Authorization": f"Bearer {raw}"})
            assert r.status_code == 200
            assert [c["catalog_id"] for c in r.json()["catalogs"]] == ["acme"]
