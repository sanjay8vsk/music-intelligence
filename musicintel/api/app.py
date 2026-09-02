"""Application factory.

Start-up order matters and is deliberate:

  1. logging, so anything that fails later is recorded structurally;
  2. API keys, so a misconfigured key file fails the process rather than
     silently authenticating nobody;
  3. the recognition service, whose constructor warms the matcher's compiled
     kernels (~1.2 s) -- paying that here means it never lands inside a request;
  4. Redis, last, because the limiter degrades to a clear 503 rather than
     preventing the process from starting.
"""

from __future__ import annotations

import os
import subprocess
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI

from musicintel.api import errors, metrics
from musicintel.api.auth import ApiKeyRegistry
from musicintel.api.config import Settings, get_settings
from musicintel.api.logging import RequestLogMiddleware, configure_logging, get_logger
from musicintel.api.metrics_middleware import MetricsMiddleware
from musicintel.api.openapi import (
    apply_problem_responses,
    reject_unknown_query_parameters,
)
from musicintel.api.ratelimit import RateLimiter, build_redis
from musicintel.api.routes_catalogs import router as catalogs_router
from musicintel.api.routes_health import router as health_router
from musicintel.api.routes_identify import router as identify_router
from musicintel.api.upload import BodySizeLimitMiddleware
from musicintel.catalog.store import CatalogStore
from musicintel.db.pool import DatabaseUnavailable, close_pool, connection, open_pool
from musicintel.db.repositories import ApiKeyRepository, CatalogRepository
from musicintel.db.usage_writer import UsageWriter
from musicintel.storage.local import storage_from_url
from musicintel.storage.sync import parse_pins, sync_all
from musicintel.service.recognition import RecognitionService

API_PREFIX = "/v1"

DESCRIPTION = """\
Acoustic recognition over per-tenant catalogs.

**Recognition is landmark fingerprinting with offset-histogram scoring.** A
result carries an `evidence_score` -- aligned landmarks divided by query
landmarks. That is a **rate, not a probability**: it is not calibrated, and this
API deliberately exposes no `confidence` field.

**Catalogs are isolated structurally.** Each catalog is a separate index
artifact, so a query against one catalog cannot reach another tenant's
recordings -- the postings simply are not in the array being searched.

Errors follow **RFC 9457**: every failure is `application/problem+json`.
"""


def _git_commit() -> str | None:
    """Build commit, from the environment if set, else git. Never fatal."""
    env = os.environ.get("MUSICINTEL_GIT_COMMIT")
    if env:
        return env.strip()[:40]
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return out.stdout.strip() or None
    except Exception:
        return None


def _load_key_records(settings: Settings, log) -> list[dict]:
    """API key records from whichever source is configured.

    Authorisation semantics are untouched either way: both paths produce the
    same record shape and the same `ApiKeyRegistry` consumes it.
    """
    source = settings.resolved_api_keys_source
    if source != "database":
        return settings.load_api_key_records()
    try:
        with connection() as conn:
            records = ApiKeyRepository(conn).load_records()
    except DatabaseUnavailable as exc:
        raise RuntimeError(
            f"API keys are configured to come from the database, which is "
            f"unreachable: {exc}"
        ) from exc
    log.info("auth.keys_loaded", source="database", count=len(records))
    return records


def _build_version() -> str:
    try:
        from importlib.metadata import version
        return version("musicintel")
    except Exception:
        return "0.0.0+unknown"


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings: Settings = app.state.settings
    configure_logging(level=settings.log_level, json_output=settings.log_json)
    log = get_logger()

    # Persistence, before anything that depends on it. A service told to read
    # its keys from a database it cannot reach must not start and then serve
    # every request with a 401 -- that looks like mass credential revocation.
    app.state.usage_writer = None
    app.state.persistence = settings.persistence_enabled
    if settings.persistence_enabled:
        open_pool(settings.database_url,
                  min_size=settings.db_pool_min_size,
                  max_size=settings.db_pool_max_size,
                  timeout=settings.db_connect_timeout)
        if settings.db_auto_migrate:
            from musicintel.db.migrate import apply_migrations
            with connection() as conn:
                applied = apply_migrations(conn)
            if applied:
                log.info("db.migrated", migrations=applied)

    app.state.api_keys = ApiKeyRegistry(_load_key_records(settings, log))
    app.state.build_version = _build_version()
    app.state.git_commit = _git_commit()

    # Artifacts are pulled BEFORE the service exists, so nothing can serve a
    # catalog that is absent, partial or unverified -- and so no request ever
    # triggers a remote fetch. Failure here fails start-up on purpose.
    app.state.artifact_sync = None
    if settings.artifact_storage_enabled:
        storage = storage_from_url(settings.artifact_storage_url)
        db_versions = None
        if settings.persistence_enabled:
            with connection() as conn:
                db_versions = CatalogRepository(conn).current_versions()
        results = sync_all(
            storage, settings.catalog_root,
            pins=parse_pins(settings.artifact_pins),
            only=settings.sync_catalog_list or None,
            db_versions=db_versions, logger=log,
        )
        app.state.artifact_sync = results
        log.info("artifact.sync_complete", catalogs=len(results),
                 storage=storage.describe(),
                 fetched=sum(1 for r in results if r.action == "fetched"))

    store = CatalogStore(settings.catalog_root)
    # The constructor warms the JIT; see docs/recognition-service.md.
    app.state.service = RecognitionService(store)
    app.state.recognition_ready = True
    metrics.RECOGNITION_READY.set(1)

    # Pre-injected by tests; built from settings otherwise.
    if getattr(app.state, "redis", None) is None:
        app.state.redis = build_redis(settings)
    app.state.limiter = RateLimiter(app.state.redis, enabled=settings.rate_limit_enabled)

    if settings.persistence_enabled:
        writer = UsageWriter(max_queue=settings.usage_queue_size,
                             batch_size=settings.usage_batch_size,
                             flush_interval=settings.usage_flush_seconds,
                             logger=log)
        writer.start()
        app.state.usage_writer = writer

    log.info(
        "service.started",
        environment=settings.environment,
        api_keys=len(app.state.api_keys),
        catalog_root=str(settings.catalog_root),
        warm_up_seconds=round(app.state.service.warm_up_seconds, 3),
        rate_limiting=settings.rate_limit_enabled,
        persistence=settings.persistence_enabled,
        api_keys_source=settings.resolved_api_keys_source,
        artifact_storage=settings.artifact_storage_url or "local volume",
    )
    try:
        yield
    finally:
        writer = getattr(app.state, "usage_writer", None)
        if writer is not None:
            # Flush before the pool closes: queued records are billing history.
            writer.stop()
            log.info("usage.writer_stopped", written=writer.written,
                     dropped=writer.dropped, failures=writer.failures)
        if settings.persistence_enabled:
            close_pool()
        redis = getattr(app.state, "redis", None)
        if redis is not None:
            try:
                await redis.aclose()
            except Exception:
                pass
        metrics.RECOGNITION_READY.set(0)
        log.info("service.stopped")


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()

    app = FastAPI(
        title="Music Intelligence API",
        description=DESCRIPTION,
        version=_build_version(),
        openapi_url=f"{API_PREFIX}/openapi.json",
        docs_url=f"{API_PREFIX}/docs",
        redoc_url=f"{API_PREFIX}/redoc",
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.recognition_ready = False
    app.state.redis = None

    # Outermost first: the body cap must reject before anything reads the body.
    app.add_middleware(RequestLogMiddleware)
    app.add_middleware(MetricsMiddleware)
    app.add_middleware(BodySizeLimitMiddleware, max_bytes=settings.max_upload_bytes)

    errors.install_error_handlers(app)

    strict = [Depends(reject_unknown_query_parameters)]
    app.include_router(health_router, prefix=API_PREFIX, dependencies=strict)
    app.include_router(catalogs_router, prefix=API_PREFIX, dependencies=strict)
    app.include_router(identify_router, prefix=API_PREFIX, dependencies=strict)

    # The published contract must describe the errors that are actually sent.
    def custom_openapi() -> dict:
        if app.openapi_schema is None:
            app.openapi_schema = apply_problem_responses(_base_openapi(app))
        return app.openapi_schema

    _base_openapi = _make_base_openapi()
    app.openapi = custom_openapi
    return app


def _make_base_openapi():
    from fastapi.openapi.utils import get_openapi

    def base(app: FastAPI) -> dict:
        return get_openapi(
            title=app.title, version=app.version, description=app.description,
            routes=app.routes,
        )
    return base


__all__ = ["API_PREFIX", "create_app", "lifespan"]
