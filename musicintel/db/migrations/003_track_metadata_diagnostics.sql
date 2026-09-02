-- Stage 2: keep enough on a failed enrichment row to explain it afterwards.
--
-- WHAT WAS LOST BEFORE THIS
-- -------------------------
-- Every failure collapsed to `match_status='error'` plus the query. A 400 (our
-- query is malformed -- retrying is pointless) and a 503 retry-exhaustion
-- (transient -- re-run later) were byte-identical on disk. The live experiment
-- hit exactly that: `ia_AF026` failed and the row could not say why.
--
-- The client already computed both facts and threw them away: `LookupResult`
-- carries `.error` and `.attempts`, and the worker passed neither.
--
-- WHAT IS DELIBERATELY NOT HERE
-- -----------------------------
-- No `http_status` column: half the failure classes (timeout, connection
-- refused, wall-clock cap) have no HTTP status at all, and where one exists it
-- is already the first token of `error_detail`. No exception-type column, for
-- the same reason. No response bodies -- `_get` never reads a body on an error
-- status, and storing attacker-influenced payloads to explain a failure is a
-- cost with no diagnostic return.

ALTER TABLE track_metadata
    -- The client's own bounded diagnostic string, e.g. 'HTTP 503',
    -- 'wall-clock cap exceeded', 'connection error: ...'. This is OPERATIONAL
    -- data produced by us, not content returned by MusicBrainz: it carries none
    -- of the CC0 / CC BY-NC-SA question that attaches to `raw_response` and
    -- `match_score`.
    --
    -- 512 characters, chosen from measurement rather than taste. Every
    -- classifying diagnostic is short -- 'HTTP 503' is 8 characters, the longest
    -- fixed message 33 -- but a connection failure embeds the full request URL
    -- and measured 326 characters with a short title and 404 with the longest
    -- real corpus title. 512 clears the largest observed case with headroom
    -- while keeping a pathological message from bloating the table. Truncation
    -- is safe by construction: every diagnostic puts its classification FIRST
    -- ('HTTP 503', 'timeout:', 'connection error:', 'response was not JSON:'),
    -- so cutting the tail costs URL detail and never the failure class.
    --
    -- Truncation happens in the repository; this CHECK is the backstop that
    -- should never fire.
    ADD COLUMN error_detail text,

    -- How many lookup attempts the client made. Separates "failed immediately"
    -- (permanent) from "failed after exhausting retries" (transient), and on a
    -- SUCCESSFUL row records that a retry was needed -- transient trouble that
    -- resolved itself and is otherwise invisible, since `requests_made` does
    -- not even count a timed-out request.
    ADD COLUMN attempts smallint;

ALTER TABLE track_metadata
    ADD CONSTRAINT track_metadata_error_detail_bounded
        CHECK (error_detail IS NULL OR length(error_detail) <= 512),
    ADD CONSTRAINT track_metadata_attempts_sane
        CHECK (attempts IS NULL OR attempts >= 0);

-- Finding rows that failed for one reason, without scanning the table.
CREATE INDEX track_metadata_error_idx ON track_metadata (source, match_status)
    WHERE error_detail IS NOT NULL;
