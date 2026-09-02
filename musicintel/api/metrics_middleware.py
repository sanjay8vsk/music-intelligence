"""Per-request Prometheus instrumentation.

Separate from the logging middleware so the metric label set stays small and
explicit. `route` is the *template* (`/v1/catalogs/{catalog_id}`), never the
resolved path -- using the path would let a caller mint unbounded label values
and blow up the time series database.
"""

from __future__ import annotations

import time

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

from musicintel.api import metrics


class MetricsMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        started = time.perf_counter()
        status_code = 500
        try:
            response = await call_next(request)
            status_code = response.status_code
            return response
        finally:
            route = request.scope.get("route")
            template = getattr(route, "path", None) or "unmatched"
            elapsed = time.perf_counter() - started
            metrics.REQUESTS.labels(
                method=request.method, route=template, status=str(status_code)
            ).inc()
            metrics.REQUEST_DURATION.labels(route=template).observe(elapsed)


__all__ = ["MetricsMiddleware"]
