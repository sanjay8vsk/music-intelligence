"""Catalog discovery.

**Isolation is enforced twice, on purpose.** The architectural guarantee is
structural -- each catalog is its own index artifact, so a query against catalog
A physically cannot reach catalog B's postings, because they are not in the
array being searched. This module adds the authorisation layer on top: a
principal only sees catalogs its key grants.

A catalog the caller may not access returns **404, not 403**. 403 would confirm
that the catalog exists, which is exactly the fact a competitor should not be
able to probe for. Unknown and forbidden are indistinguishable from outside.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Path, Query, Request

from musicintel.api import errors, metrics
from musicintel.api.auth import Principal, require_principal
from musicintel.api.schemas import (
    CatalogDetail, CatalogListResponse, CatalogSummary, TrackSummary,
)
from musicintel.catalog.store import CatalogStoreError, validate_catalog_id

router = APIRouter(tags=["catalogs"])

_PROBLEM = {
    401: {"description": "Missing or invalid API key."},
    403: {"description": "The key lacks the required scope."},
    404: {"description": "No such catalog, or the key cannot access it."},
}


def _require_scope(principal: Principal, scope: str) -> None:
    if not principal.has_scope(scope):
        raise errors.forbidden(f"This API key lacks the '{scope}' scope.")


def _resolve(request: Request, principal: Principal, catalog_id: str):
    """Validate, authorise and load. Every failure looks like 'not found'."""
    try:
        catalog_id = validate_catalog_id(catalog_id)
    except (CatalogStoreError, ValueError):
        raise errors.not_found("No such catalog.")

    if not principal.may_access(catalog_id):
        metrics.REJECTIONS.labels(reason="catalog_forbidden").inc()
        raise errors.not_found("No such catalog.")

    service = request.app.state.service
    try:
        return catalog_id, service.get(catalog_id)
    except (CatalogStoreError, FileNotFoundError, KeyError):
        raise errors.not_found("No such catalog.")


@router.get(
    "/catalogs",
    response_model=CatalogListResponse,
    responses=_PROBLEM,
    summary="List accessible catalogs",
    description="Returns only catalogs this API key may access.",
)
async def list_catalogs(
    request: Request, principal: Principal = Depends(require_principal)
) -> CatalogListResponse:
    _require_scope(principal, "catalogs:read")
    service = request.app.state.service
    cached = getattr(service, "_cache", {}) or {}

    summaries: list[CatalogSummary] = []
    for catalog_id in sorted(service.catalogs()):
        if not principal.may_access(catalog_id):
            continue
        loaded = catalog_id in cached
        if loaded:
            count = cached[catalog_id].track_count
        else:
            # Read the artifact descriptor rather than loading the index: a
            # listing must not pull hundreds of megabytes into memory.
            try:
                count = int(service.store.describe(catalog_id).get("track_count", 0))
            except Exception:
                continue
        summaries.append(
            CatalogSummary(catalog_id=catalog_id, track_count=count, loaded=loaded)
        )
    return CatalogListResponse(catalogs=summaries)


@router.get(
    "/catalogs/{catalog_id}",
    response_model=CatalogDetail,
    responses=_PROBLEM,
    summary="Describe one catalog",
)
async def get_catalog(
    request: Request,
    catalog_id: str = Path(
        ...,
        min_length=1,
        max_length=64,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$",
        description="Catalog identifier.",
    ),
    limit: int = Query(50, ge=1, le=500, description="Tracks to return."),
    offset: int = Query(0, ge=0, description="Tracks to skip."),
    principal: Principal = Depends(require_principal),
) -> CatalogDetail:
    _require_scope(principal, "catalogs:read")
    catalog_id, loaded = _resolve(request, principal, catalog_id)

    tracks = loaded.catalog.tracks[offset: offset + limit]
    return CatalogDetail(
        catalog_id=catalog_id,
        track_count=loaded.track_count,
        loaded=True,
        content_hash=loaded.catalog.content_hash(),
        tracks=[
            TrackSummary(
                track_id=t.track_id,
                title=t.title,
                artist=t.artist,
                duration_seconds=round(t.duration_sec, 3),
            )
            for t in tracks
        ],
        returned=len(tracks),
        offset=offset,
    )


__all__ = ["router"]
