-- Stage 2: third-party metadata, kept separate from owned track identity.
--
-- WHY A SEPARATE TABLE AND NOT COLUMNS ON `tracks`
-- ------------------------------------------------
-- Two concrete reasons, both observed rather than stylistic:
--
--   1. `CatalogRepository.sync()` writes `title = EXCLUDED.title` from
--      catalog.json. Anything enriched into `tracks.title` would be erased by
--      the next catalog re-sync. Separation makes re-syncing a catalog and
--      re-enriching it independent operations that cannot destroy each other.
--
--   2. Enrichment written back into catalog.json changes its bytes without
--      changing `catalog_content_hash` or `index_content_hash` -- the same
--      content-addressed artifact key with different content, which the storage
--      layer correctly refuses. Enrichment must not live in the artifact.
--
-- Beyond that: a customer's own claims about their audio and a third party's
-- claims have different trust, refresh cadence and licensing, and should not
-- share a row.
--
-- This table holds NO fingerprints and takes no part in recognition.

CREATE TABLE track_metadata (
    catalog_id      text NOT NULL,
    track_id        text NOT NULL,
    -- In the primary key so a second provider can be added later without a
    -- migration, and so one provider's answer never overwrites another's.
    source          text NOT NULL DEFAULT 'musicbrainz',

    -- ---- the third party's claims -------------------------------------
    title           text,
    artist          text,
    album           text,
    release_date    text,          -- MusicBrainz dates may be YYYY or YYYY-MM
    isrc            text,
    mb_recording_id uuid,
    mb_release_id   uuid,
    mb_artist_id    uuid,

    -- ---- provenance ----------------------------------------------------
    -- `ambiguous` is a first-class outcome: several plausible recordings and no
    -- basis for choosing. Recording it as a match would manufacture certainty
    -- the lookup did not have.
    match_status    text NOT NULL,
    match_score     numeric(5, 2),
    -- The exact query sent, so a result can be explained months later.
    query_used      text,
    -- The provider's answer as received, for re-normalisation without refetching.
    raw_response    jsonb,
    fetched_at      timestamptz NOT NULL DEFAULT now(),
    created_at      timestamptz NOT NULL DEFAULT now(),
    updated_at      timestamptz NOT NULL DEFAULT now(),

    PRIMARY KEY (catalog_id, track_id, source),

    -- Cascade, unlike `usage`: a third party's claims about a deleted track
    -- have no independent value, whereas billing history does.
    FOREIGN KEY (catalog_id, track_id)
        REFERENCES tracks (catalog_id, track_id) ON DELETE CASCADE,

    CONSTRAINT track_metadata_status CHECK (
        match_status IN ('matched', 'no_match', 'ambiguous', 'error', 'skipped')
    ),
    CONSTRAINT track_metadata_score CHECK (
        match_score IS NULL OR (match_score >= 0 AND match_score <= 100)
    ),
    -- A match must actually name something; otherwise it is `no_match`.
    CONSTRAINT track_metadata_matched_has_content CHECK (
        match_status <> 'matched'
        OR title IS NOT NULL OR mb_recording_id IS NOT NULL
    )
);

CREATE INDEX track_metadata_isrc_idx ON track_metadata (isrc)
    WHERE isrc IS NOT NULL;
CREATE INDEX track_metadata_recording_idx ON track_metadata (mb_recording_id)
    WHERE mb_recording_id IS NOT NULL;
CREATE INDEX track_metadata_status_idx ON track_metadata (source, match_status);
-- Drives "what still needs enriching / re-enriching past its max age".
CREATE INDEX track_metadata_fetched_idx ON track_metadata (source, fetched_at);
