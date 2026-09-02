"""Make the published spec tell the truth about errors.

FastAPI documents every validation failure as `application/json` carrying
`HTTPValidationError`. This service does not return that: `install_error_handlers`
renders **every** failure as RFC 9457 `application/problem+json`. Left alone, the
spec would lie about the shape of every error response, and a client generated
from it would break the first time anything went wrong -- which schemathesis
catches immediately, and rightly.

`apply_problem_responses` rewrites the generated document so that every response
with a 4xx or 5xx status advertises the media type and schema that is actually
sent. It runs over the finished schema rather than being declared per route, so
a new endpoint cannot forget to do it.

`reject_unknown_query_parameters` closes the other gap. FastAPI ignores query
parameters it does not recognise, so `?limt=5` silently returns page one instead
of failing. Rejecting them turns a silent wrong answer into a loud error.
"""

from __future__ import annotations

from typing import Any

from fastapi import Request

from musicintel.api import errors

PROBLEM_MEDIA_TYPE = "application/problem+json"
_PROBLEM_REF = {"$ref": "#/components/schemas/ProblemResponse"}

_DEFAULT_DESCRIPTIONS = {
    "400": "Malformed request.",
    "401": "Missing or invalid API key.",
    "403": "The key lacks the required scope.",
    "404": "Not found, or not accessible to this key.",
    "413": "Payload exceeds the configured limit.",
    "415": "Unsupported media type.",
    "422": "The request did not match the schema, or the audio was unusable.",
    "429": "Rate limit or quota exceeded.",
    "500": "Internal error.",
    "503": "A dependency is unavailable.",
}


def apply_problem_responses(schema: dict[str, Any]) -> dict[str, Any]:
    """Rewrite every error response to RFC 9457. Idempotent."""
    components = schema.setdefault("components", {}).setdefault("schemas", {})
    if "ProblemResponse" not in components:
        from musicintel.api.schemas import ProblemResponse
        components["ProblemResponse"] = ProblemResponse.model_json_schema()

    for operations in schema.get("paths", {}).values():
        for operation in operations.values():
            if not isinstance(operation, dict):
                continue
            responses = operation.get("responses")
            if not isinstance(responses, dict):
                continue
            for status, response in responses.items():
                if not status[:1].isdigit() or int(status[0]) < 4:
                    continue
                response["content"] = {PROBLEM_MEDIA_TYPE: {"schema": dict(_PROBLEM_REF)}}
                if not response.get("description"):
                    response["description"] = _DEFAULT_DESCRIPTIONS.get(status, "Error.")

    # FastAPI only emits HTTPValidationError/ValidationError for the 422s it
    # generated; with those rewritten the definitions are dead weight in a
    # published contract.
    for dead in ("HTTPValidationError", "ValidationError"):
        components.pop(dead, None)
    return schema


async def reject_unknown_query_parameters(request: Request) -> None:
    """Refuse query parameters the operation does not declare."""
    route = request.scope.get("route")
    dependant = getattr(route, "dependant", None)
    if dependant is None:
        return
    allowed = {p.alias for p in getattr(dependant, "query_params", [])}
    for sub in getattr(dependant, "dependencies", []):
        allowed |= {p.alias for p in getattr(sub, "query_params", [])}
    unknown = sorted(set(request.query_params) - allowed)
    if unknown:
        raise errors.validation_failed(
            "The request included query parameters this endpoint does not accept.",
            errors=[
                {"location": ["query", name], "message": "Unknown query parameter.",
                 "kind": "unexpected_parameter"}
                for name in unknown
            ],
        )


__all__ = [
    "PROBLEM_MEDIA_TYPE", "apply_problem_responses",
    "reject_unknown_query_parameters",
]
