"""Turn a MusicBrainz recording into the columns `track_metadata` stores.

Defensive by construction: every field is optional in the response, so each is
extracted with an explicit fallback and a missing field is simply absent rather
than an exception. A provider changing its payload shape should degrade what we
record, not crash a batch halfway through.

Nothing here decides *whether* something matched -- `MusicBrainzClient._classify`
does that, and an `ambiguous` result is normalised for its evidence but never
promoted to a match.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass

_UUID = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-"
                   r"[0-9a-f]{4}-[0-9a-f]{12}$", re.I)


@dataclass(frozen=True)
class NormalizedMetadata:
    title: str | None = None
    artist: str | None = None
    album: str | None = None
    release_date: str | None = None
    isrc: str | None = None
    mb_recording_id: str | None = None
    mb_release_id: str | None = None
    mb_artist_id: str | None = None
    match_score: float | None = None

    def as_fields(self) -> dict:
        return {k: v for k, v in asdict(self).items() if v is not None}


def _text(value) -> str | None:
    if value is None:
        return None
    s = str(value).strip()
    return s or None


def _uuid(value) -> str | None:
    s = _text(value)
    return s.lower() if s and _UUID.match(s) else None


def _artist_credit(recording: dict) -> tuple[str | None, str | None]:
    """Joined artist name and the first artist's MBID.

    MusicBrainz models collaborations as an ordered credit list with join
    phrases ("A feat. B"); reconstructing it preserves what the provider
    actually says rather than silently keeping only the first name.
    """
    credits = recording.get("artist-credit")
    if not isinstance(credits, list) or not credits:
        return None, None
    parts: list[str] = []
    first_id: str | None = None
    for entry in credits:
        if isinstance(entry, str):
            parts.append(entry)
            continue
        if not isinstance(entry, dict):
            continue
        artist = entry.get("artist") if isinstance(entry.get("artist"), dict) else {}
        name = _text(entry.get("name")) or _text(artist.get("name"))
        if name:
            parts.append(name)
        if first_id is None:
            first_id = _uuid(artist.get("id"))
        join = entry.get("joinphrase")
        if isinstance(join, str) and join:
            parts.append(join)
    joined = "".join(
        p if p.startswith((" ", ",")) or not parts else p for p in parts).strip()
    return (_text(joined), first_id)


def _release(recording: dict) -> tuple[str | None, str | None, str | None]:
    """(album title, release MBID, date) from the first release, if any."""
    releases = recording.get("releases")
    if not isinstance(releases, list) or not releases:
        return None, None, None
    first = releases[0]
    if not isinstance(first, dict):
        return None, None, None
    return (_text(first.get("title")), _uuid(first.get("id")),
            _text(first.get("date")))


def _isrc(recording: dict) -> str | None:
    isrcs = recording.get("isrcs")
    if isinstance(isrcs, list):
        for value in isrcs:
            code = _text(value)
            if code:
                return code.upper()
    return None


def normalize_recording(recording: dict) -> NormalizedMetadata:
    """One MusicBrainz recording -> storable fields. Never raises."""
    if not isinstance(recording, dict):
        return NormalizedMetadata()
    artist, artist_id = _artist_credit(recording)
    album, release_id, date = _release(recording)
    try:
        score = float(recording.get("score")) if recording.get("score") is not None else None
    except (TypeError, ValueError):
        score = None
    if score is not None:
        score = max(0.0, min(100.0, score))
    return NormalizedMetadata(
        title=_text(recording.get("title")),
        artist=artist,
        album=album,
        release_date=date,
        isrc=_isrc(recording),
        mb_recording_id=_uuid(recording.get("id")),
        mb_release_id=release_id,
        mb_artist_id=artist_id,
        match_score=score,
    )


__all__ = ["NormalizedMetadata", "normalize_recording"]
