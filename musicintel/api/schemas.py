"""Request and response models.

These are the API's contract. Two rules hold throughout:

  * **No confidence field.** `evidence_score` is a rate -- aligned landmarks
    over query landmarks -- exactly as in the decision layer. Presenting it as a
    probability would be a lie the recognition core deliberately refuses to
    tell, and a test asserts no such field appears.
  * **No internal identifiers.** File paths, ordinals and index internals stay
    server-side.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class HealthResponse(BaseModel):
    model_config = ConfigDict(json_schema_extra={
        "example": {"status": "ok", "service": "music-intelligence",
                    "catalogs_loaded": 1, "recognition_ready": True}
    })

    status: Literal["ok", "degraded"] = Field(description="Overall service state.")
    service: str
    catalogs_loaded: int = Field(ge=0, description="Catalogs currently in memory.")
    recognition_ready: bool = Field(
        description="True once the matcher's compiled kernels are warm."
    )


class VersionResponse(BaseModel):
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "service": "music-intelligence", "version": "0.1.0",
            "api_version": "v1", "recognizer": "landmark-gated",
            "fingerprint_format_version": 1, "index_format_version": 1,
            "git_commit": "6ce1285", "environment": "production",
        }
    })

    service: str
    version: str
    api_version: Literal["v1"]
    recognizer: str = Field(description="Recognition pipeline identifier.")
    fingerprint_format_version: int
    index_format_version: int
    git_commit: str | None = None
    environment: str


class CatalogSummary(BaseModel):
    model_config = ConfigDict(json_schema_extra={
        "example": {"catalog_id": "acme", "track_count": 500, "loaded": True}
    })

    catalog_id: str
    track_count: int = Field(ge=0)
    loaded: bool = Field(description="Whether the index is resident in memory.")


class CatalogListResponse(BaseModel):
    catalogs: list[CatalogSummary]


class TrackSummary(BaseModel):
    track_id: str
    title: str | None = None
    artist: str | None = None
    duration_seconds: float | None = Field(default=None, ge=0)


class CatalogDetail(BaseModel):
    catalog_id: str
    track_count: int = Field(ge=0)
    loaded: bool
    content_hash: str = Field(description="Hash over the catalog's track set.")
    tracks: list[TrackSummary] = Field(
        description="Truncated to the requested page."
    )
    returned: int = Field(ge=0)
    offset: int = Field(ge=0)


class IdentifyMatch(BaseModel):
    """The identified recording, present only when `decision` is `match`."""

    track_id: str
    title: str | None = None
    artist: str | None = None
    offset_seconds: float | None = Field(
        default=None,
        description="Where the query sits inside the reference recording.",
    )
    rate_percent: float | None = Field(
        default=None,
        description="Speed correction applied by the cascade, in percent. "
                    "Null when the query matched at native rate.",
    )


class IdentifyResponse(BaseModel):
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "decision": "match", "catalog_id": "acme",
            "match": {"track_id": "t-0042", "title": "Example",
                      "artist": "Someone", "offset_seconds": 61.3,
                      "rate_percent": None},
            "evidence_score": 0.0871, "threshold": 0.026316,
            "aligned_landmarks": 64, "query_landmarks": 735,
            "stage": 1, "escalated": False,
            "query_duration_seconds": 5.0, "latency_ms": 84.2,
        }
    })

    decision: Literal["match", "no_match"]
    catalog_id: str
    match: IdentifyMatch | None = None

    # Evidence, reported as a rate. NOT a probability -- see the module note.
    evidence_score: float = Field(
        ge=0.0,
        description="Aligned landmarks divided by query landmarks. A rate, "
                    "not a probability and not a calibrated confidence.",
    )
    threshold: float = Field(
        ge=0.0, description="The evidence rate this query had to clear."
    )
    aligned_landmarks: int = Field(ge=0)
    query_landmarks: int = Field(ge=0)

    stage: int | None = Field(
        default=None, ge=0,
        description="Which cascade stage produced the verdict: 1 = native rate, "
                    "2 = speed-corrected. Null when nothing matched, because no "
                    "stage accepted the query.",
    )
    escalated: bool = Field(
        description="Whether the speed-tolerant second stage ran."
    )

    query_duration_seconds: float = Field(ge=0)
    latency_ms: float = Field(
        ge=0, description="Server-side recognition time, excluding decode."
    )


class ProblemResponse(BaseModel):
    """RFC 9457 problem document. Declared so it appears in the OpenAPI spec."""

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "type": "https://docs.musicintel.dev/problems/unauthorized",
            "title": "Unauthorized", "status": 401,
            "detail": "A valid API key is required.", "instance": "/v1/identify",
        }
    })

    type: str
    title: str
    status: int
    detail: str | None = None
    instance: str | None = None


__all__ = [
    "CatalogDetail", "CatalogListResponse", "CatalogSummary", "HealthResponse",
    "IdentifyMatch", "IdentifyResponse", "ProblemResponse", "TrackSummary",
    "VersionResponse",
]
