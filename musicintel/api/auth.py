"""API-key authentication (S5).

Design constraints, each from the security assessment:

  * **Hashes only.** The service never holds a usable key. Configuration carries
    SHA-256 digests; a presented key is hashed and looked up. There is no code
    path that can print a key, because no key is ever stored.
  * **Prefixed keys.** `sk_live_` / `sk_test_` so secret scanners and humans
    recognise a leak on sight.
  * **`api_key_id` in logs, never the key.** Every record has a stable id; that
    is what appears in logs and metrics.
  * **Instant revocation.** `active: false` takes effect on the next request.
  * **Constant-time lookup.** The presented key is hashed and used as a dict
    key. Nothing compares secrets byte-by-byte, so there is no timing signal in
    how far a comparison got.

Keys live in configuration, not a database: Stage 3 predates Postgres by design.
The record shape is the same one a `api_keys` table will hold, so moving to a
database later is a repository swap rather than a redesign.
"""

from __future__ import annotations

import hashlib
import hmac
from dataclasses import dataclass, field

from fastapi import Request

from musicintel.api import errors

KEY_PREFIXES = ("sk_live_", "sk_test_")


def hash_key(raw_key: str) -> str:
    """The digest stored in configuration. SHA-256 of the exact key string."""
    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Principal:
    """Who is calling, and what they are allowed to do."""

    key_id: str
    tenant: str
    catalogs: tuple[str, ...] = ()          # empty tuple == every catalog
    rate_limit_per_minute: int = 60
    rate_limit_burst: int = 0               # 0 == same as the per-minute rate
    audio_seconds_per_day: int = 3600
    active: bool = True
    scopes: frozenset[str] = field(default_factory=lambda: frozenset({"identify", "catalogs:read"}))

    def may_access(self, catalog_id: str) -> bool:
        """Catalog visibility. An empty allow-list means unrestricted."""
        return not self.catalogs or catalog_id in self.catalogs

    def has_scope(self, scope: str) -> bool:
        return scope in self.scopes

    @property
    def burst(self) -> int:
        return self.rate_limit_burst or self.rate_limit_per_minute


class ApiKeyRegistry:
    """Digest -> Principal. Built once at start-up."""

    def __init__(self, records: list[dict]) -> None:
        self._by_digest: dict[str, Principal] = {}
        for record in records:
            digest = record.get("key_sha256")
            key_id = record.get("key_id")
            tenant = record.get("tenant")
            if not digest or not key_id or not tenant:
                raise ValueError(
                    "each API key record needs key_id, tenant and key_sha256"
                )
            digest = digest.strip().lower()
            if len(digest) != 64 or not all(c in "0123456789abcdef" for c in digest):
                raise ValueError(f"key_sha256 for {key_id} is not a SHA-256 digest")
            if digest in self._by_digest:
                raise ValueError(f"duplicate key digest for {key_id}")
            scopes = record.get("scopes")
            self._by_digest[digest] = Principal(
                key_id=key_id,
                tenant=tenant,
                catalogs=tuple(record.get("catalogs") or ()),
                rate_limit_per_minute=int(record.get("rate_limit_per_minute", 60)),
                rate_limit_burst=int(record.get("rate_limit_burst", 0)),
                audio_seconds_per_day=int(record.get("audio_seconds_per_day", 3600)),
                active=bool(record.get("active", True)),
                scopes=frozenset(scopes) if scopes else Principal.__dataclass_fields__[
                    "scopes"].default_factory(),
            )

    def __len__(self) -> int:
        return len(self._by_digest)

    def resolve(self, raw_key: str) -> Principal | None:
        """Principal for a presented key, or None. Revoked keys resolve to None."""
        principal = self._by_digest.get(hash_key(raw_key))
        if principal is None or not principal.active:
            return None
        return principal


def extract_key(request: Request) -> str | None:
    """Pull the key from `Authorization: Bearer` or `X-API-Key`."""
    header = request.headers.get("authorization")
    if header:
        scheme, _, value = header.partition(" ")
        if scheme.lower() == "bearer" and value.strip():
            return value.strip()
        return None
    api_key = request.headers.get("x-api-key")
    return api_key.strip() if api_key else None


def looks_like_key(candidate: str) -> bool:
    return any(candidate.startswith(p) for p in KEY_PREFIXES)


async def require_principal(request: Request) -> Principal:
    """FastAPI dependency: authenticate, or raise a 401 problem document."""
    registry: ApiKeyRegistry = request.app.state.api_keys
    raw = extract_key(request)
    if not raw:
        raise errors.unauthorized(
            "Provide an API key as 'Authorization: Bearer <key>' or 'X-API-Key'."
        )
    principal = registry.resolve(raw)
    if principal is None:
        # One message for unknown, malformed and revoked alike: distinguishing
        # them would confirm which keys exist.
        raise errors.unauthorized("The provided API key is not valid.")
    request.state.principal = principal
    return principal


def constant_time_equals(a: str, b: str) -> bool:
    return hmac.compare_digest(a.encode("utf-8"), b.encode("utf-8"))


__all__ = [
    "ApiKeyRegistry", "KEY_PREFIXES", "Principal", "constant_time_equals",
    "extract_key", "hash_key", "looks_like_key", "require_principal",
]
