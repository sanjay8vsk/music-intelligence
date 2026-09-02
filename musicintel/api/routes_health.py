"""Liveness, version and metrics.

These three are unauthenticated on purpose: a health check that needs a secret
cannot be used by the thing that restarts the process. They therefore expose
nothing an attacker could not learn by reading the public docs -- no catalog
names, no key counts, no paths.
"""

from __future__ import annotations

from fastapi import APIRouter, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from musicintel.api import metrics
from musicintel.api.schemas import HealthResponse, VersionResponse
from musicintel.recognition.fingerprint import FORMAT_VERSION
from musicintel.recognition.index import INDEX_FORMAT_VERSION

router = APIRouter(tags=["service"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Liveness and readiness",
    description="Unauthenticated. Reports whether recognition is warm and how "
                "many catalog indexes are resident.",
)
async def health(request: Request) -> HealthResponse:
    state = request.app.state
    loaded = len(getattr(state.service, "_cache", {}) or {})
    ready = bool(getattr(state, "recognition_ready", False))
    return HealthResponse(
        status="ok" if ready else "degraded",
        service=state.settings.service_name,
        catalogs_loaded=loaded,
        recognition_ready=ready,
    )


@router.get(
    "/version",
    response_model=VersionResponse,
    summary="Build and format versions",
    description="Unauthenticated. Format versions identify which fingerprint "
                "and index layouts this build can read.",
)
async def version(request: Request) -> VersionResponse:
    state = request.app.state
    return VersionResponse(
        service=state.settings.service_name,
        version=state.build_version,
        api_version="v1",
        recognizer="landmark-gated",
        fingerprint_format_version=FORMAT_VERSION,
        index_format_version=INDEX_FORMAT_VERSION,
        git_commit=state.git_commit,
        environment=state.settings.environment,
    )


@router.get(
    "/metrics",
    include_in_schema=False,
    summary="Prometheus exposition",
)
async def prometheus(request: Request) -> Response:
    if not request.app.state.settings.metrics_enabled:
        return Response(status_code=404)
    return Response(
        content=generate_latest(metrics.REGISTRY),
        media_type=CONTENT_TYPE_LATEST,
    )


__all__ = ["router"]
