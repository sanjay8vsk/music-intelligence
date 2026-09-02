"""Structured logging.

One event per request, emitted after the response is known, carrying the fields
you actually page on: route, status, duration, tenant, api_key_id, request_id.

What is deliberately NOT logged: the API key itself (S5 -- only `api_key_id`
ever appears), the uploaded bytes, and any filesystem path from the catalog
store. A log line should be safe to ship to a third-party aggregator.
"""

from __future__ import annotations

import logging
import sys
import time
import uuid
from contextvars import ContextVar

import structlog
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.types import ASGIApp

request_id_var: ContextVar[str] = ContextVar("request_id", default="")


def configure_logging(*, level: str = "INFO", json_output: bool = True) -> None:
    """Idempotent structlog setup."""
    logging.basicConfig(
        format="%(message)s", stream=sys.stdout,
        level=getattr(logging, level.upper(), logging.INFO),
    )
    processors: list = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]
    processors.append(
        structlog.processors.JSONRenderer() if json_output
        else structlog.dev.ConsoleRenderer()
    )
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, level.upper(), logging.INFO)
        ),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str = "musicintel.api"):
    return structlog.get_logger(name)


class RequestLogMiddleware(BaseHTTPMiddleware):
    """Assign a request id, time the request, emit exactly one event."""

    def __init__(self, app: ASGIApp) -> None:
        super().__init__(app)
        self._log = get_logger()

    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex
        request_id_var.set(request_id)
        structlog.contextvars.bind_contextvars(request_id=request_id)
        request.state.request_id = request_id

        started = time.perf_counter()
        status_code = 500
        try:
            response = await call_next(request)
            status_code = response.status_code
            response.headers["x-request-id"] = request_id
            return response
        except Exception:
            # The error handler renders the response; this records the cause.
            self._log.exception(
                "request.failed",
                method=request.method,
                path=request.url.path,
                duration_ms=round((time.perf_counter() - started) * 1000, 3),
            )
            raise
        finally:
            duration_ms = round((time.perf_counter() - started) * 1000, 3)
            principal = getattr(request.state, "principal", None)
            self._log.info(
                "request.completed",
                method=request.method,
                path=request.url.path,
                route=getattr(
                    getattr(request.scope.get("route"), "path", None),
                    "__str__", lambda: None
                )() if request.scope.get("route") else None,
                status=status_code,
                duration_ms=duration_ms,
                api_key_id=getattr(principal, "key_id", None),
                tenant=getattr(principal, "tenant", None),
            )
            structlog.contextvars.clear_contextvars()


__all__ = [
    "RequestLogMiddleware", "configure_logging", "get_logger", "request_id_var",
]
