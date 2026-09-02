"""RFC 9457 Problem Details.

Every error this API returns is a `application/problem+json` document. One shape
for every failure means a client writes one error handler, and it means an
unexpected exception cannot leak a stack trace or an internal path -- the
handler below turns anything it does not recognise into a generic 500 with no
detail, and logs the real cause server-side.

RFC 9457 members used: `type` (a URI that identifies the problem kind), `title`
(a short human-readable summary, stable for a given type), `status`, `detail`
(specific to this occurrence), `instance` (the request path). Extension members
are allowed and are used for machine-actionable context -- `retry_after`,
`limit`, `max_bytes` -- never for anything a client should have to parse out of
prose.
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

PROBLEM_MEDIA_TYPE = "application/problem+json"


class ProblemDetail(Exception):
    """An error that already knows how it should be rendered."""

    def __init__(
        self,
        *,
        type_suffix: str,
        title: str,
        status_code: int,
        detail: str | None = None,
        headers: dict[str, str] | None = None,
        **extensions: Any,
    ) -> None:
        super().__init__(detail or title)
        self.type_suffix = type_suffix
        self.title = title
        self.status_code = status_code
        self.detail = detail
        self.headers = headers or {}
        self.extensions = extensions

    def to_dict(self, *, base_uri: str, instance: str) -> dict[str, Any]:
        body: dict[str, Any] = {
            "type": f"{base_uri}/{self.type_suffix}",
            "title": self.title,
            "status": self.status_code,
            "instance": instance,
        }
        if self.detail:
            body["detail"] = self.detail
        body.update({k: v for k, v in self.extensions.items() if v is not None})
        return body


# -- the problem kinds this service can produce ---------------------------
# Each is a function rather than a subclass: the call site reads as a sentence,
# and the `type` URI stays in one place per kind so it cannot drift.


def unauthorized(detail: str = "A valid API key is required.") -> ProblemDetail:
    return ProblemDetail(
        type_suffix="unauthorized",
        title="Unauthorized",
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail=detail,
        headers={"WWW-Authenticate": 'Bearer realm="music-intelligence"'},
    )


def forbidden(detail: str) -> ProblemDetail:
    return ProblemDetail(
        type_suffix="forbidden",
        title="Forbidden",
        status_code=status.HTTP_403_FORBIDDEN,
        detail=detail,
    )


def not_found(detail: str) -> ProblemDetail:
    return ProblemDetail(
        type_suffix="not-found",
        title="Not Found",
        status_code=status.HTTP_404_NOT_FOUND,
        detail=detail,
    )


def payload_too_large(detail: str, *, max_bytes: int) -> ProblemDetail:
    return ProblemDetail(
        type_suffix="payload-too-large",
        title="Payload Too Large",
        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
        detail=detail,
        max_bytes=max_bytes,
    )


def unsupported_media(detail: str, *, supported: list[str] | None = None) -> ProblemDetail:
    return ProblemDetail(
        type_suffix="unsupported-media-type",
        title="Unsupported Media Type",
        status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
        detail=detail,
        supported_formats=supported,
    )


def unprocessable(detail: str, **ext: Any) -> ProblemDetail:
    return ProblemDetail(
        type_suffix="unprocessable-audio",
        title="Unprocessable Audio",
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        detail=detail,
        **ext,
    )


def validation_failed(detail: str, *, errors: list[dict] | None = None) -> ProblemDetail:
    return ProblemDetail(
        type_suffix="validation-failed",
        title="Validation Failed",
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        detail=detail,
        errors=errors,
    )


def rate_limited(detail: str, *, retry_after: int, **ext: Any) -> ProblemDetail:
    return ProblemDetail(
        type_suffix="rate-limit-exceeded",
        title="Too Many Requests",
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        detail=detail,
        headers={"Retry-After": str(retry_after)},
        retry_after=retry_after,
        **ext,
    )


def quota_exceeded(detail: str, *, retry_after: int, **ext: Any) -> ProblemDetail:
    return ProblemDetail(
        type_suffix="quota-exceeded",
        title="Quota Exceeded",
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        detail=detail,
        headers={"Retry-After": str(retry_after)},
        retry_after=retry_after,
        **ext,
    )


def service_unavailable(detail: str) -> ProblemDetail:
    return ProblemDetail(
        type_suffix="service-unavailable",
        title="Service Unavailable",
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail=detail,
    )


def internal_error() -> ProblemDetail:
    # Deliberately detail-free: the cause is logged, never returned.
    return ProblemDetail(
        type_suffix="internal-error",
        title="Internal Server Error",
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
    )


# -- wiring ---------------------------------------------------------------
_STATUS_TITLES = {
    400: ("bad-request", "Bad Request"),
    401: ("unauthorized", "Unauthorized"),
    403: ("forbidden", "Forbidden"),
    404: ("not-found", "Not Found"),
    405: ("method-not-allowed", "Method Not Allowed"),
    413: ("payload-too-large", "Payload Too Large"),
    415: ("unsupported-media-type", "Unsupported Media Type"),
    422: ("validation-failed", "Validation Failed"),
    429: ("rate-limit-exceeded", "Too Many Requests"),
    500: ("internal-error", "Internal Server Error"),
    503: ("service-unavailable", "Service Unavailable"),
}


def _render(request: Request, problem: ProblemDetail) -> JSONResponse:
    base = request.app.state.settings.problem_base_uri
    return JSONResponse(
        status_code=problem.status_code,
        content=problem.to_dict(base_uri=base, instance=request.url.path),
        media_type=PROBLEM_MEDIA_TYPE,
        headers=problem.headers or None,
    )


def install_error_handlers(app: FastAPI) -> None:
    """Route every failure through the problem+json renderer."""

    @app.exception_handler(ProblemDetail)
    async def _problem(request: Request, exc: ProblemDetail) -> JSONResponse:
        return _render(request, exc)

    @app.exception_handler(StarletteHTTPException)
    async def _http(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        suffix, title = _STATUS_TITLES.get(
            exc.status_code, ("error", "Error")
        )
        detail = exc.detail if isinstance(exc.detail, str) else None
        headers = dict(exc.headers or {})
        return _render(
            request,
            ProblemDetail(
                type_suffix=suffix,
                title=title,
                status_code=exc.status_code,
                detail=detail,
                headers=headers,
            ),
        )

    @app.exception_handler(RequestValidationError)
    async def _validation(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        # Pydantic's errors carry the offending input; strip it so an oversized
        # or binary body cannot be echoed back.
        errors = [
            {
                "location": list(e.get("loc", [])),
                "message": e.get("msg", ""),
                "kind": e.get("type", ""),
            }
            for e in exc.errors()
        ]
        return _render(
            request,
            validation_failed("The request did not match the expected schema.",
                              errors=errors),
        )

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
        # The cause is logged by the logging middleware; the client gets nothing.
        return _render(request, internal_error())


__all__ = [
    "PROBLEM_MEDIA_TYPE",
    "ProblemDetail",
    "forbidden",
    "install_error_handlers",
    "internal_error",
    "not_found",
    "payload_too_large",
    "quota_exceeded",
    "rate_limited",
    "service_unavailable",
    "unauthorized",
    "unprocessable",
    "unsupported_media",
    "validation_failed",
]
