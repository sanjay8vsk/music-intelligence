-- Stage 2: durable identity, configuration and usage history.
--
-- WHAT THIS IS NOT
-- ----------------
-- This schema does not hold fingerprints, postings or any part of the
-- recognition index. Catalog isolation stays STRUCTURAL: each catalog is its
-- own on-disk index artifact, so a query against one catalog physically cannot
-- reach another's postings. These tables record *what exists and who owns it*,
-- which is the part that needs to survive a restart and be queried by something
-- other than the recogniser.

-- ------------------------------------------------------------- catalogs --
CREATE TABLE catalogs (
    catalog_id                  text PRIMARY KEY,
    tenant                      text        NOT NULL,
    -- Identity of WHICH audio the catalog holds: Catalog.content_hash(), over
    -- (track_id, sha256) pairs only.
    content_hash                char(64)    NOT NULL,
    track_count                 integer     NOT NULL DEFAULT 0,
    fingerprint_count           bigint      NOT NULL DEFAULT 0,
    -- Mirrors artifact.json so drift between the database and the artifact on
    -- disk is detectable rather than silent.
    artifact_version            integer,
    index_content_hash          char(64),
    fingerprint_format_version  integer,
    index_format_version        integer,
    created_at                  timestamptz NOT NULL DEFAULT now(),
    updated_at                  timestamptz NOT NULL DEFAULT now(),

    -- The same rule validate_catalog_id() enforces in Python, enforced again
    -- here so a direct SQL writer cannot introduce an id the store would refuse
    -- to open.
    CONSTRAINT catalogs_id_shape
        CHECK (catalog_id ~ '^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$'),
    CONSTRAINT catalogs_content_hash_shape CHECK (content_hash ~ '^[0-9a-f]{64}$'),
    CONSTRAINT catalogs_counts_sane
        CHECK (track_count >= 0 AND fingerprint_count >= 0)
);

CREATE INDEX catalogs_tenant_idx ON catalogs (tenant);

-- --------------------------------------------------------------- tracks --
CREATE TABLE tracks (
    catalog_id        text        NOT NULL
                                  REFERENCES catalogs (catalog_id) ON DELETE CASCADE,
    -- The identity the recogniser returns.
    track_id          text        NOT NULL,
    sha256            char(64)    NOT NULL,
    duration_sec      double precision NOT NULL,
    bytes             bigint      NOT NULL DEFAULT 0,
    fingerprint_count integer     NOT NULL DEFAULT 0,
    title             text,
    artist            text,
    -- Provenance, never identity: a rename is not a new track. Never returned
    -- by the API.
    source_path       text,
    created_at        timestamptz NOT NULL DEFAULT now(),
    updated_at        timestamptz NOT NULL DEFAULT now(),

    PRIMARY KEY (catalog_id, track_id),
    -- Content-hash deduplication as a database invariant, not just an ingest
    -- convention: the same audio cannot enter one catalog twice under two ids.
    CONSTRAINT tracks_content_unique UNIQUE (catalog_id, sha256),
    CONSTRAINT tracks_sha256_shape CHECK (sha256 ~ '^[0-9a-f]{64}$'),
    CONSTRAINT tracks_duration_sane CHECK (duration_sec >= 0),
    CONSTRAINT tracks_counts_sane CHECK (bytes >= 0 AND fingerprint_count >= 0)
);

-- Cross-catalog duplicate discovery: the same recording licensed to two
-- tenants is two rows, and this makes them findable without a scan.
CREATE INDEX tracks_sha256_idx ON tracks (sha256);

-- ------------------------------------------------------------- api_keys --
-- Exactly the record shape musicintel/api/auth.py already consumes, so moving
-- keys here is a repository swap and not a redesign.
--
-- No raw key is stored anywhere in this schema, only its SHA-256 digest. There
-- is no column a leak could expose a usable credential from.
CREATE TABLE api_keys (
    key_id                text        PRIMARY KEY,
    tenant                text        NOT NULL,
    key_sha256            char(64)    NOT NULL UNIQUE,
    -- Empty array means every catalog, matching Principal.may_access().
    catalogs              text[]      NOT NULL DEFAULT '{}',
    scopes                text[]      NOT NULL DEFAULT '{identify,catalogs:read}',
    rate_limit_per_minute integer     NOT NULL DEFAULT 60,
    -- 0 means "same as the per-minute rate", matching Principal.burst.
    rate_limit_burst      integer     NOT NULL DEFAULT 0,
    audio_seconds_per_day integer     NOT NULL DEFAULT 3600,
    -- Revocation. Effective on the next registry load.
    active                boolean     NOT NULL DEFAULT true,
    created_at            timestamptz NOT NULL DEFAULT now(),
    updated_at            timestamptz NOT NULL DEFAULT now(),

    CONSTRAINT api_keys_digest_shape CHECK (key_sha256 ~ '^[0-9a-f]{64}$'),
    CONSTRAINT api_keys_limits_sane CHECK (
        rate_limit_per_minute > 0
        AND rate_limit_burst >= 0
        AND audio_seconds_per_day >= 0
    )
);

CREATE INDEX api_keys_tenant_idx ON api_keys (tenant);
CREATE INDEX api_keys_active_idx ON api_keys (active) WHERE active;

-- ---------------------------------------------------------------- usage --
-- The durable billing record. Redis remains the OPERATIONAL limiter -- it
-- answers "may this request proceed" in one atomic round trip -- but its
-- counters carry a two-day TTL and do not survive a flush. This table is what
-- an invoice is built from.
--
-- Deliberately NOT foreign-keyed to api_keys: deleting a key must not delete
-- the history of what it consumed. Billing history outlives credentials.
CREATE TABLE usage (
    tenant         text        NOT NULL,
    key_id         text        NOT NULL,
    usage_day      date        NOT NULL,
    -- numeric, not double precision: this accumulates across millions of small
    -- additions and is money-adjacent. Float drift is not acceptable here.
    audio_seconds  numeric(14, 3) NOT NULL DEFAULT 0,
    request_count  bigint      NOT NULL DEFAULT 0,
    match_count    bigint      NOT NULL DEFAULT 0,
    no_match_count bigint      NOT NULL DEFAULT 0,
    first_seen_at  timestamptz NOT NULL DEFAULT now(),
    updated_at     timestamptz NOT NULL DEFAULT now(),

    PRIMARY KEY (tenant, key_id, usage_day),
    CONSTRAINT usage_amounts_sane CHECK (
        audio_seconds >= 0 AND request_count >= 0
        AND match_count >= 0 AND no_match_count >= 0
    )
);

CREATE INDEX usage_day_idx ON usage (usage_day);
CREATE INDEX usage_tenant_day_idx ON usage (tenant, usage_day);
