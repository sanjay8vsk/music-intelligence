"""Repositories over the Stage 2 tables.

Each one speaks the vocabulary the rest of the codebase already uses:

  * `CatalogRepository` takes a `Catalog` and the artifact descriptor the store
    already writes, so syncing is a call, not a translation layer.
  * `ApiKeyRepository.load_records()` returns dicts in **exactly** the shape
    `musicintel.api.auth.ApiKeyRegistry` already consumes. Moving keys from a
    JSON file to Postgres is therefore a swap of where the list comes from, with
    no change to authorisation semantics.
  * `UsageRepository.record()` is an accumulating upsert, so applying the same
    batch twice after a retry adds the amount twice but never corrupts a row --
    and applying a batch partially is impossible, because each batch is one
    transaction.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import Decimal

from musicintel.catalog.models import Catalog, CatalogTrack


# ------------------------------------------------------------- catalogs --
class CatalogRepository:
    """Catalog and track identity. Never fingerprints."""

    def __init__(self, conn) -> None:
        self._conn = conn

    def sync(self, catalog: Catalog, *, tenant: str,
             artifact: dict | None = None, catalog_id: str | None = None) -> str:
        """Write a catalog and all of its tracks, atomically.

        Tracks absent from `catalog` are deleted, so the table reflects the
        catalog as it now is rather than accumulating removed recordings. The
        whole thing is one transaction: a half-written catalog would claim a
        `content_hash` that does not describe its own rows.
        """
        cid = catalog_id or catalog.catalog_id
        art = artifact or {}
        with self._conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO catalogs (
                    catalog_id, tenant, content_hash, track_count,
                    fingerprint_count, artifact_version, index_content_hash,
                    fingerprint_format_version, index_format_version, updated_at
                ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s, now())
                ON CONFLICT (catalog_id) DO UPDATE SET
                    tenant                     = EXCLUDED.tenant,
                    content_hash               = EXCLUDED.content_hash,
                    track_count                = EXCLUDED.track_count,
                    fingerprint_count          = EXCLUDED.fingerprint_count,
                    artifact_version           = EXCLUDED.artifact_version,
                    index_content_hash         = EXCLUDED.index_content_hash,
                    fingerprint_format_version = EXCLUDED.fingerprint_format_version,
                    index_format_version       = EXCLUDED.index_format_version,
                    updated_at                 = now()
                """,
                (cid, tenant, catalog.content_hash(), len(catalog),
                 int(art.get("fingerprint_count", catalog.total_fingerprints)),
                 art.get("artifact_version"), art.get("index_content_hash"),
                 art.get("fingerprint_format_version"),
                 art.get("index_format_version")),
            )
            for t in catalog.tracks:
                cur.execute(
                    """
                    INSERT INTO tracks (
                        catalog_id, track_id, sha256, duration_sec, bytes,
                        fingerprint_count, title, artist, source_path, updated_at
                    ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s, now())
                    ON CONFLICT (catalog_id, track_id) DO UPDATE SET
                        sha256            = EXCLUDED.sha256,
                        duration_sec      = EXCLUDED.duration_sec,
                        bytes             = EXCLUDED.bytes,
                        fingerprint_count = EXCLUDED.fingerprint_count,
                        title             = EXCLUDED.title,
                        artist            = EXCLUDED.artist,
                        source_path       = EXCLUDED.source_path,
                        updated_at        = now()
                    """,
                    (cid, t.track_id, t.sha256, float(t.duration_sec),
                     int(t.bytes), int(t.fingerprint_count), t.title, t.artist,
                     t.source_path),
                )
            keep = list(catalog.track_ids)
            cur.execute(
                "DELETE FROM tracks WHERE catalog_id = %s AND NOT (track_id = ANY(%s))",
                (cid, keep),
            )
        self._conn.commit()
        return cid

    def get(self, catalog_id: str) -> dict | None:
        with self._conn.cursor() as cur:
            cur.execute(
                """SELECT catalog_id, tenant, content_hash, track_count,
                          fingerprint_count, artifact_version, index_content_hash
                   FROM catalogs WHERE catalog_id = %s""", (catalog_id,))
            row = cur.fetchone()
        if row is None:
            return None
        return dict(zip(
            ("catalog_id", "tenant", "content_hash", "track_count",
             "fingerprint_count", "artifact_version", "index_content_hash"), row))

    def list_for_tenant(self, tenant: str) -> list[str]:
        with self._conn.cursor() as cur:
            cur.execute(
                "SELECT catalog_id FROM catalogs WHERE tenant = %s ORDER BY catalog_id",
                (tenant,))
            return [r[0] for r in cur.fetchall()]

    def current_versions(self) -> dict[str, str]:
        """catalog_id -> index_content_hash, for artifact version resolution.

        The Stage 2 schema already records which index a catalog is at, so boot
        synchronisation can answer "which version" without a mutable `latest`
        pointer in object storage.
        """
        with self._conn.cursor() as cur:
            cur.execute("SELECT catalog_id, index_content_hash FROM catalogs "
                        "WHERE index_content_hash IS NOT NULL")
            return {r[0]: r[1] for r in cur.fetchall()}

    def tracks(self, catalog_id: str) -> list[CatalogTrack]:
        with self._conn.cursor() as cur:
            cur.execute(
                """SELECT track_id, source_path, sha256, duration_sec, bytes,
                          fingerprint_count, title, artist
                   FROM tracks WHERE catalog_id = %s ORDER BY track_id""",
                (catalog_id,))
            return [
                CatalogTrack(track_id=r[0], source_path=r[1] or "", sha256=r[2],
                             duration_sec=float(r[3]), bytes=int(r[4]),
                             fingerprint_count=int(r[5]), title=r[6], artist=r[7])
                for r in cur.fetchall()
            ]

    def delete(self, catalog_id: str) -> bool:
        with self._conn.cursor() as cur:
            cur.execute("DELETE FROM catalogs WHERE catalog_id = %s", (catalog_id,))
            deleted = cur.rowcount > 0
        self._conn.commit()
        return deleted


# ------------------------------------------------------------- api_keys --
class ApiKeyRepository:
    """Keys, in the exact record shape `ApiKeyRegistry` already consumes."""

    def __init__(self, conn) -> None:
        self._conn = conn

    def load_records(self, *, include_inactive: bool = False) -> list[dict]:
        sql = """SELECT key_id, tenant, key_sha256, catalogs, scopes,
                        rate_limit_per_minute, rate_limit_burst,
                        audio_seconds_per_day, active
                 FROM api_keys"""
        if not include_inactive:
            sql += " WHERE active"
        sql += " ORDER BY key_id"
        with self._conn.cursor() as cur:
            cur.execute(sql)
            rows = cur.fetchall()
        return [
            {"key_id": r[0], "tenant": r[1], "key_sha256": r[2],
             "catalogs": list(r[3] or []), "scopes": list(r[4] or []),
             "rate_limit_per_minute": int(r[5]), "rate_limit_burst": int(r[6]),
             "audio_seconds_per_day": int(r[7]), "active": bool(r[8])}
            for r in rows
        ]

    def upsert(self, record: dict) -> str:
        """Insert or update one key record. Never accepts a raw key."""
        if "key" in record or "secret" in record:
            raise ValueError(
                "refusing a record containing a raw key; store key_sha256 only")
        with self._conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO api_keys (
                    key_id, tenant, key_sha256, catalogs, scopes,
                    rate_limit_per_minute, rate_limit_burst,
                    audio_seconds_per_day, active, updated_at
                ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s, now())
                ON CONFLICT (key_id) DO UPDATE SET
                    tenant                = EXCLUDED.tenant,
                    key_sha256            = EXCLUDED.key_sha256,
                    catalogs              = EXCLUDED.catalogs,
                    scopes                = EXCLUDED.scopes,
                    rate_limit_per_minute = EXCLUDED.rate_limit_per_minute,
                    rate_limit_burst      = EXCLUDED.rate_limit_burst,
                    audio_seconds_per_day = EXCLUDED.audio_seconds_per_day,
                    active                = EXCLUDED.active,
                    updated_at            = now()
                """,
                (record["key_id"], record["tenant"],
                 record["key_sha256"].strip().lower(),
                 list(record.get("catalogs") or []),
                 list(record.get("scopes") or ["identify", "catalogs:read"]),
                 int(record.get("rate_limit_per_minute", 60)),
                 int(record.get("rate_limit_burst", 0)),
                 int(record.get("audio_seconds_per_day", 3600)),
                 bool(record.get("active", True))),
            )
        self._conn.commit()
        return record["key_id"]

    def revoke(self, key_id: str) -> bool:
        with self._conn.cursor() as cur:
            cur.execute(
                "UPDATE api_keys SET active = false, updated_at = now() "
                "WHERE key_id = %s", (key_id,))
            changed = cur.rowcount > 0
        self._conn.commit()
        return changed


# ---------------------------------------------------------------- usage --
@dataclass(frozen=True)
class UsageRow:
    tenant: str
    key_id: str
    usage_day: date
    audio_seconds: float
    request_count: int
    match_count: int
    no_match_count: int


class UsageRepository:
    """Durable consumption history. Redis enforces; this records."""

    def __init__(self, conn) -> None:
        self._conn = conn

    def record(self, tenant: str, key_id: str, usage_day: date, *,
               audio_seconds: float, requests: int = 1,
               matches: int = 0, no_matches: int = 0) -> None:
        self.record_many([
            (tenant, key_id, usage_day, audio_seconds, requests, matches, no_matches)
        ])

    def record_many(self, batch) -> int:
        """Accumulate a batch of usage in one transaction.

        Additive upsert, so ordering does not matter and a partially applied
        batch is impossible. Retrying a batch double-counts it; the writer
        therefore only retries batches it knows were never committed.
        """
        rows = list(batch)
        if not rows:
            return 0
        with self._conn.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO usage (tenant, key_id, usage_day, audio_seconds,
                                   request_count, match_count, no_match_count)
                VALUES (%s,%s,%s,%s,%s,%s,%s)
                ON CONFLICT (tenant, key_id, usage_day) DO UPDATE SET
                    audio_seconds  = usage.audio_seconds  + EXCLUDED.audio_seconds,
                    request_count  = usage.request_count  + EXCLUDED.request_count,
                    match_count    = usage.match_count    + EXCLUDED.match_count,
                    no_match_count = usage.no_match_count + EXCLUDED.no_match_count,
                    updated_at     = now()
                """,
                [(t, k, d, Decimal(str(round(float(s), 3))), int(r), int(m), int(n))
                 for t, k, d, s, r, m, n in rows],
            )
        self._conn.commit()
        return len(rows)

    def get(self, tenant: str, key_id: str, usage_day: date) -> UsageRow | None:
        with self._conn.cursor() as cur:
            cur.execute(
                """SELECT tenant, key_id, usage_day, audio_seconds, request_count,
                          match_count, no_match_count
                   FROM usage WHERE tenant=%s AND key_id=%s AND usage_day=%s""",
                (tenant, key_id, usage_day))
            row = cur.fetchone()
        if row is None:
            return None
        return UsageRow(row[0], row[1], row[2], float(row[3]),
                        int(row[4]), int(row[5]), int(row[6]))

    def tenant_total(self, tenant: str, usage_day: date) -> float:
        """Audio seconds for a tenant on one day, across all of its keys."""
        with self._conn.cursor() as cur:
            cur.execute(
                "SELECT COALESCE(SUM(audio_seconds), 0) FROM usage "
                "WHERE tenant = %s AND usage_day = %s", (tenant, usage_day))
            return float(cur.fetchone()[0])

    def history(self, tenant: str, *, since: date | None = None) -> list[UsageRow]:
        sql = ("SELECT tenant, key_id, usage_day, audio_seconds, request_count, "
               "match_count, no_match_count FROM usage WHERE tenant = %s")
        params: list = [tenant]
        if since is not None:
            sql += " AND usage_day >= %s"
            params.append(since)
        sql += " ORDER BY usage_day, key_id"
        with self._conn.cursor() as cur:
            cur.execute(sql, params)
            return [UsageRow(r[0], r[1], r[2], float(r[3]), int(r[4]), int(r[5]),
                             int(r[6])) for r in cur.fetchall()]


# ------------------------------------------------------- track_metadata --
@dataclass(frozen=True)
class TrackMetadataRow:
    catalog_id: str
    track_id: str
    source: str
    match_status: str
    title: str | None = None
    artist: str | None = None
    album: str | None = None
    release_date: str | None = None
    isrc: str | None = None
    mb_recording_id: str | None = None
    mb_release_id: str | None = None
    mb_artist_id: str | None = None
    match_score: float | None = None
    query_used: str | None = None
    error_detail: str | None = None
    attempts: int | None = None
    fetched_at: object | None = None


_METADATA_COLUMNS = (
    "catalog_id", "track_id", "source", "title", "artist", "album",
    "release_date", "isrc", "mb_recording_id", "mb_release_id", "mb_artist_id",
    "match_status", "match_score", "query_used", "error_detail", "attempts",
)

# Matches the CHECK in migration 003. Truncation happens here so the constraint
# is a backstop that never fires. Safe because every diagnostic the client
# produces puts its classification first -- cutting the tail loses URL detail,
# never the failure class.
ERROR_DETAIL_MAX_CHARS = 512
_TRUNCATION_MARKER = "…[truncated]"


def _bound_error_detail(value) -> str | None:
    if value is None:
        return None
    text = str(value)
    if len(text) <= ERROR_DETAIL_MAX_CHARS:
        return text
    keep = ERROR_DETAIL_MAX_CHARS - len(_TRUNCATION_MARKER)
    return text[:keep] + _TRUNCATION_MARKER


class TrackMetadataRepository:
    """Third-party metadata. Never owned identity, never fingerprints.

    `upsert` replaces one provider's answer for one track: re-enrichment is a
    new answer, not an accumulation, which is the opposite of `usage` and is why
    this uses assignment rather than addition.
    """

    def __init__(self, conn) -> None:
        self._conn = conn

    def upsert(self, catalog_id: str, track_id: str, *, match_status: str,
               source: str = "musicbrainz", raw_response: dict | None = None,
               **fields) -> None:
        import json as _json

        if match_status not in ("matched", "no_match", "ambiguous", "error",
                                "skipped"):
            raise ValueError(f"unknown match_status {match_status!r}")
        payload = {c: None for c in _METADATA_COLUMNS}
        payload.update(catalog_id=catalog_id, track_id=track_id, source=source,
                       match_status=match_status)
        for key, value in fields.items():
            if key not in payload:
                raise ValueError(f"unknown track_metadata field {key!r}")
            payload[key] = value
        payload["error_detail"] = _bound_error_detail(payload["error_detail"])

        with self._conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO track_metadata (
                    catalog_id, track_id, source, title, artist, album,
                    release_date, isrc, mb_recording_id, mb_release_id,
                    mb_artist_id, match_status, match_score, query_used,
                    error_detail, attempts,
                    raw_response, fetched_at, updated_at
                ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,
                          now(), now())
                ON CONFLICT (catalog_id, track_id, source) DO UPDATE SET
                    title           = EXCLUDED.title,
                    artist          = EXCLUDED.artist,
                    album           = EXCLUDED.album,
                    release_date    = EXCLUDED.release_date,
                    isrc            = EXCLUDED.isrc,
                    mb_recording_id = EXCLUDED.mb_recording_id,
                    mb_release_id   = EXCLUDED.mb_release_id,
                    mb_artist_id    = EXCLUDED.mb_artist_id,
                    match_status    = EXCLUDED.match_status,
                    match_score     = EXCLUDED.match_score,
                    query_used      = EXCLUDED.query_used,
                    error_detail    = EXCLUDED.error_detail,
                    attempts        = EXCLUDED.attempts,
                    raw_response    = EXCLUDED.raw_response,
                    fetched_at      = now(),
                    updated_at      = now()
                """,
                (*[payload[c] for c in _METADATA_COLUMNS],
                 _json.dumps(raw_response) if raw_response is not None else None),
            )
        self._conn.commit()

    def get(self, catalog_id: str, track_id: str,
            source: str = "musicbrainz") -> TrackMetadataRow | None:
        with self._conn.cursor() as cur:
            cur.execute(
                """SELECT catalog_id, track_id, source, match_status, title,
                          artist, album, release_date, isrc, mb_recording_id,
                          mb_release_id, mb_artist_id, match_score, query_used,
                          error_detail, attempts, fetched_at
                   FROM track_metadata
                   WHERE catalog_id=%s AND track_id=%s AND source=%s""",
                (catalog_id, track_id, source))
            row = cur.fetchone()
        if row is None:
            return None
        return TrackMetadataRow(
            *row[:4],
            title=row[4], artist=row[5], album=row[6], release_date=row[7],
            isrc=row[8],
            mb_recording_id=str(row[9]) if row[9] else None,
            mb_release_id=str(row[10]) if row[10] else None,
            mb_artist_id=str(row[11]) if row[11] else None,
            match_score=float(row[12]) if row[12] is not None else None,
            query_used=row[13], error_detail=row[14],
            attempts=int(row[15]) if row[15] is not None else None,
            fetched_at=row[16])

    def raw_response(self, catalog_id: str, track_id: str,
                     source: str = "musicbrainz") -> dict | None:
        with self._conn.cursor() as cur:
            cur.execute("SELECT raw_response FROM track_metadata "
                        "WHERE catalog_id=%s AND track_id=%s AND source=%s",
                        (catalog_id, track_id, source))
            row = cur.fetchone()
        return row[0] if row and row[0] is not None else None

    def counts_by_status(self, catalog_id: str | None = None,
                         source: str = "musicbrainz") -> dict[str, int]:
        sql = ("SELECT match_status, count(*) FROM track_metadata "
               "WHERE source = %s")
        params: list = [source]
        if catalog_id is not None:
            sql += " AND catalog_id = %s"
            params.append(catalog_id)
        sql += " GROUP BY match_status"
        with self._conn.cursor() as cur:
            cur.execute(sql, params)
            return {r[0]: int(r[1]) for r in cur.fetchall()}

    def pending(self, catalog_id: str, *, source: str = "musicbrainz",
                max_age_days: int | None = None, force: bool = False,
                limit: int | None = None) -> list[tuple[str, str, str]]:
        """(track_id, title, artist) for tracks still needing enrichment.

        Only tracks that actually carry title and artist are returned: without
        them there is nothing to search MusicBrainz with, and guessing from a
        filename-derived track_id would manufacture wrong matches.
        """
        sql = ["""SELECT t.track_id, t.title, t.artist
                  FROM tracks t
                  LEFT JOIN track_metadata m
                    ON m.catalog_id = t.catalog_id
                   AND m.track_id  = t.track_id
                   AND m.source    = %s
                  WHERE t.catalog_id = %s
                    AND t.title IS NOT NULL AND t.artist IS NOT NULL"""]
        params: list = [source, catalog_id]
        if not force:
            if max_age_days is None:
                sql.append(" AND m.track_id IS NULL")
            else:
                sql.append(" AND (m.track_id IS NULL OR m.fetched_at < now() - "
                           "make_interval(days => %s))")
                params.append(int(max_age_days))
        sql.append(" ORDER BY t.track_id")
        if limit is not None:
            sql.append(" LIMIT %s")
            params.append(int(limit))
        with self._conn.cursor() as cur:
            cur.execute("".join(sql), params)
            return [(r[0], r[1], r[2]) for r in cur.fetchall()]

    def without_metadata(self, catalog_id: str) -> list[str]:
        """Tracks that cannot be enriched because they carry no title/artist."""
        with self._conn.cursor() as cur:
            cur.execute("SELECT track_id FROM tracks WHERE catalog_id = %s "
                        "AND (title IS NULL OR artist IS NULL) ORDER BY track_id",
                        (catalog_id,))
            return [r[0] for r in cur.fetchall()]


__all__ = [
    "ERROR_DETAIL_MAX_CHARS", "ApiKeyRepository", "CatalogRepository",
    "TrackMetadataRepository", "TrackMetadataRow", "UsageRepository", "UsageRow",
]
