"""POST /v1/identify -- the product.

Order of operations is a security property, not a style choice. Every cheap
refusal happens before every expensive one:

    auth -> scope -> catalog authorisation -> request rate -> quota peek
         -> size -> magic bytes -> sandboxed decode -> quota charge
         -> recognition

An unauthenticated caller never reaches the decoder. A caller over quota never
reaches the decoder either. Only a caller who is authenticated, authorised, in
rate and in budget can make this service spend CPU on parsing bytes it did not
choose.

Both blocking steps -- decode and recognition -- run in a worker thread.
Blocking the event loop would serialise every concurrent request behind one
CPU-bound identify, which is a self-inflicted denial of service.
"""

from __future__ import annotations

import time

from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, Request, Response
from starlette.concurrency import run_in_threadpool

from musicintel.api import errors, metrics
from musicintel.api.auth import Principal, require_principal
from musicintel.api.decode import DecodeError, decode_audio
from musicintel.api.logging import get_logger
from musicintel.api.ratelimit import RateLimiterUnavailable
from musicintel.api.routes_catalogs import _require_scope, _resolve
from musicintel.api.schemas import IdentifyMatch, IdentifyResponse
from musicintel.api.upload import SUPPORTED_FORMATS
from musicintel.recognition.decision import Decision

router = APIRouter(tags=["recognition"])
log = get_logger("musicintel.api.identify")

# Decode failures the client can act on, mapped to the right status. Anything
# not listed is a 422 -- never a 500, because a bad upload is not a server bug.
_DECODE_STATUS = {
    "unsupported_format": "unsupported_media",
    "empty": "unprocessable",
    "empty_audio": "unprocessable",
    "too_many_channels": "unprocessable",
    "decode_timeout": "unprocessable",
    "decoded_too_long": "payload_too_large",
    "decode_crashed": "unprocessable",
    "decode_failed": "unprocessable",
    "decoder_unavailable": "service_unavailable",
}


@router.post(
    "/identify",
    response_model=IdentifyResponse,
    summary="Identify a recording",
    description=(
        "Upload a short audio excerpt and identify it against one catalog.\n\n"
        "The response carries `evidence_score`, the fraction of query landmarks "
        "that aligned. It is a **rate, not a probability** -- there is no "
        "calibrated confidence, and this API does not invent one."
    ),
    responses={
        400: {"description": "The multipart body could not be parsed."},
        401: {"description": "Missing or invalid API key."},
        403: {"description": "The key lacks the 'identify' scope."},
        404: {"description": "No such catalog, or the key cannot access it."},
        413: {"description": "Upload exceeds the size limit."},
        415: {"description": "Not a recognised audio container."},
        422: {"description": "Audio could not be decoded."},
        429: {"description": "Request rate or audio-second quota exceeded."},
        500: {"description": "Unexpected server error."},
        503: {"description": "A dependency is unavailable."},
    },
)
async def identify(
    request: Request,
    response: Response,
    # `bytes` rather than `UploadFile`: a multipart part with no `filename=`
    # is parsed as an ordinary form field, and an `UploadFile` annotation then
    # rejects it with a confusing "file: Field required" even though the client
    # did send `file`. Accepting bytes takes both spellings. Nothing is lost --
    # the body is already capped upstream and read into memory either way.
    # min_length=1 is not decoration: a zero-byte upload cannot be identified,
    # so the schema should say so. Without it the spec advertises that an empty
    # body is acceptable, and the caller gets a misleading "file: Field
    # required" for a field they did send.
    file: Annotated[bytes, File(min_length=1,
                                description="Audio excerpt to identify.")],
    catalog_id: Annotated[str, Form(min_length=1, max_length=64,
                                    description="Catalog to search.")],
    principal: Principal = Depends(require_principal),
) -> IdentifyResponse:
    settings = request.app.state.settings
    limiter = request.app.state.limiter

    _require_scope(principal, "identify")
    catalog_id, loaded = _resolve(request, principal, catalog_id)

    # -- request rate -----------------------------------------------------
    try:
        rate = await limiter.check_request_rate(
            principal.key_id,
            per_minute=principal.rate_limit_per_minute,
            burst=principal.burst,
        )
    except RateLimiterUnavailable as exc:
        log.error("ratelimit.unavailable", error=str(exc))
        metrics.REJECTIONS.labels(reason="limiter_unavailable").inc()
        raise errors.service_unavailable(
            "Rate limiting is temporarily unavailable."
        )
    if not rate.allowed:
        metrics.REJECTIONS.labels(reason="rate_limited").inc()
        raise errors.rate_limited(
            "Request rate exceeded for this API key.",
            retry_after=rate.retry_after,
            limit=int(rate.limit),
        )

    # -- quota, before spending anything on decode ------------------------
    try:
        budget = await limiter.peek_audio_seconds(
            principal.tenant, daily_limit=principal.audio_seconds_per_day
        )
    except RateLimiterUnavailable as exc:
        log.error("quota.unavailable", error=str(exc))
        metrics.REJECTIONS.labels(reason="limiter_unavailable").inc()
        raise errors.service_unavailable("Quota accounting is temporarily unavailable.")
    if not budget.allowed:
        metrics.REJECTIONS.labels(reason="quota_exceeded").inc()
        raise errors.quota_exceeded(
            "Daily audio-second quota exhausted for this tenant.",
            retry_after=budget.retry_after,
            limit=budget.limit,
            used=round(budget.used, 3),
        )

    # -- read the upload --------------------------------------------------
    payload = file
    if len(payload) > settings.max_upload_bytes:
        metrics.REJECTIONS.labels(reason="too_large").inc()
        raise errors.payload_too_large(
            "Upload exceeds the size limit.", max_bytes=settings.max_upload_bytes
        )
    if not payload:
        metrics.REJECTIONS.labels(reason="empty_upload").inc()
        raise errors.unprocessable("The uploaded file is empty.")

    # -- sandboxed decode -------------------------------------------------
    try:
        decoded = await run_in_threadpool(
            decode_audio,
            payload,
            target_sample_rate=request.app.state.service.fingerprint_config.sample_rate,
            max_seconds=settings.max_decode_seconds,
            timeout_seconds=settings.decode_timeout_seconds,
            memory_limit_bytes=settings.decode_memory_limit_bytes,
            cpu_seconds=settings.decode_cpu_seconds,
        )
    except DecodeError as exc:
        metrics.REJECTIONS.labels(reason=exc.reason).inc()
        log.info("decode.rejected", reason=exc.reason,
                 api_key_id=principal.key_id, bytes=len(payload))
        raise _decode_problem(exc, settings)

    metrics.DECODE_DURATION.observe(decoded.decode_seconds)

    # A decompression bomb is bounded by the decoder's frame cap -- it never
    # allocated more than `max_decode_seconds` of PCM. Having survived it
    # safely, refuse it explicitly rather than silently identifying the first
    # 30 seconds of a file the caller believes was processed whole.
    if decoded.truncated:
        metrics.REJECTIONS.labels(reason="audio_too_long").inc()
        log.info("decode.too_long", api_key_id=principal.key_id,
                 bytes=len(payload), cap_seconds=settings.max_decode_seconds)
        raise errors.payload_too_large(
            f"Audio is longer than the {settings.max_decode_seconds:g} second "
            f"limit for identification. Upload a shorter excerpt.",
            max_bytes=settings.max_upload_bytes,
        )

    # -- charge the quota for what was actually decoded -------------------
    try:
        charged = await limiter.consume_audio_seconds(
            principal.tenant,
            seconds=decoded.duration_seconds,
            daily_limit=principal.audio_seconds_per_day,
        )
    except RateLimiterUnavailable as exc:
        log.error("quota.unavailable", error=str(exc))
        raise errors.service_unavailable("Quota accounting is temporarily unavailable.")
    if not charged.allowed:
        metrics.REJECTIONS.labels(reason="quota_exceeded").inc()
        raise errors.quota_exceeded(
            "Daily audio-second quota exhausted for this tenant.",
            retry_after=charged.retry_after,
            limit=charged.limit,
            used=round(charged.used, 3),
        )
    metrics.AUDIO_SECONDS.labels(tenant=principal.tenant).inc(decoded.duration_seconds)

    # -- recognition ------------------------------------------------------
    started = time.perf_counter()
    result = await run_in_threadpool(
        request.app.state.service.identify,
        decoded.samples, decoded.sample_rate, catalog_id,
    )
    metrics.RECOGNITION_DURATION.observe(time.perf_counter() - started)

    matched = result.decision is Decision.MATCH
    metrics.IDENTIFICATIONS.labels(
        catalog_id=catalog_id,
        decision="match" if matched else "no_match",
        escalated=str(bool(result.escalated)).lower(),
    ).inc()

    # Durable usage, for billing. Deliberately NOT a database write on this
    # path: `record` is a bounded-queue put, and a background thread does the
    # writing. Redis already enforced the quota above, so a record lost to a
    # database outage can only under-bill -- it can never let a tenant exceed
    # its limit. See musicintel/db/usage_writer.py.
    writer = request.app.state.usage_writer
    if writer is not None:
        queued = writer.record(
            principal.tenant, principal.key_id,
            audio_seconds=decoded.duration_seconds, matched=matched,
        )
        metrics.USAGE_RECORDS.labels(
            outcome="enqueued" if queued else "dropped").inc()

    response.headers["X-RateLimit-Limit"] = str(int(rate.limit))
    response.headers["X-RateLimit-Remaining"] = str(max(0, int(rate.remaining)))
    response.headers["X-Quota-Audio-Seconds-Remaining"] = str(
        max(0, int(charged.remaining))
    )

    match = None
    if matched:
        track = result.track
        match = IdentifyMatch(
            track_id=result.track_id,
            title=getattr(track, "title", None),
            artist=getattr(track, "artist", None),
            offset_seconds=(round(result.offset_seconds, 3)
                            if result.offset_seconds is not None else None),
            rate_percent=result.rate_percent,
        )

    return IdentifyResponse(
        decision="match" if matched else "no_match",
        catalog_id=catalog_id,
        match=match,
        evidence_score=round(float(result.evidence_score), 6),
        threshold=float(result.threshold),
        aligned_landmarks=int(result.aligned_landmarks),
        query_landmarks=int(result.query_landmarks),
        stage=int(result.stage) if result.stage is not None else None,
        escalated=bool(result.escalated),
        query_duration_seconds=round(decoded.duration_seconds, 3),
        latency_ms=round(float(result.latency_ms), 3),
    )


def _decode_problem(exc: DecodeError, settings) -> errors.ProblemDetail:
    kind = _DECODE_STATUS.get(exc.reason, "unprocessable")
    if kind == "unsupported_media":
        return errors.unsupported_media(exc.message, supported=SUPPORTED_FORMATS)
    if kind == "payload_too_large":
        return errors.payload_too_large(
            exc.message, max_bytes=settings.max_upload_bytes
        )
    if kind == "service_unavailable":
        return errors.service_unavailable(exc.message)
    return errors.unprocessable(exc.message, reason=exc.reason)


__all__ = ["router"]
