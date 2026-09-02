"""Runtime configuration, from environment.

Every setting has a default that is safe to run locally, and every limit that
protects the service has a default that is safe to run in public. Nothing here
reads a secret from the repository -- API keys arrive as a JSON blob or a file
path, both supplied by the environment (S9).
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="MUSICINTEL_", env_file=".env", extra="ignore"
    )

    # -- identity ---------------------------------------------------------
    service_name: str = "music-intelligence"
    environment: str = "development"

    # -- catalogs ---------------------------------------------------------
    catalog_root: Path = Path("data/catalogs")

    # -- artifact storage (Stage 2) ---------------------------------------
    # Unset means catalogs are served from `catalog_root` exactly as before.
    # Only `file://` is implemented; no provider has been selected.
    artifact_storage_url: str | None = None
    # `catalog_id=<index_content_hash>` pairs, comma separated. Pins the exact
    # immutable version this instance serves.
    artifact_pins: str = ""
    # Catalogs to synchronise. Empty means every catalog found in storage.
    sync_catalogs: str = ""

    # -- auth -------------------------------------------------------------
    # JSON list of key records; see auth.py for the shape. Either inline JSON
    # (api_keys) or a path to a JSON file (api_keys_file). A file is preferred
    # in production so the value never appears in `ps` or a process listing.
    api_keys: str = ""
    api_keys_file: Path | None = None

    # -- durable persistence (Stage 2) ------------------------------------
    # Unset means the service behaves exactly as it did before Postgres
    # existed: keys from configuration, no durable usage history.
    database_url: str | None = None
    # Where API keys are read from. "auto" uses the database when
    # `database_url` is set and configuration otherwise.
    api_keys_source: Literal["auto", "database", "config"] = "auto"
    # Migrations are a release step, not a start-up side effect: several
    # workers racing to CREATE TABLE is not a race worth having. Enable only
    # for local development.
    db_auto_migrate: bool = False
    db_pool_min_size: int = 1
    db_pool_max_size: int = 8
    db_connect_timeout: float = 5.0
    # Write-behind usage buffer. See musicintel/db/usage_writer.py for why the
    # request path does not write synchronously.
    usage_queue_size: int = 10_000
    usage_batch_size: int = 500
    usage_flush_seconds: float = 1.0

    # -- rate limiting ----------------------------------------------------
    redis_url: str = "redis://localhost:6379/0"
    # When Redis is unreachable the limiter fails CLOSED (503) rather than
    # silently allowing unlimited traffic. See ratelimit.py for why.
    rate_limit_enabled: bool = True

    # -- upload and decode safety (S1, S2, S3) ----------------------------
    max_upload_bytes: int = 10 * 1024 * 1024        # 10 MiB on the wire
    max_decode_seconds: float = 30.0                # decoded audio, hard cap
    decode_timeout_seconds: float = 10.0            # wall clock for the sandbox
    decode_memory_limit_bytes: int = 512 * 1024 * 1024
    decode_cpu_seconds: int = 10                    # RLIMIT_CPU for the sandbox

    # -- observability ----------------------------------------------------
    log_level: str = "INFO"
    log_json: bool = True
    metrics_enabled: bool = True

    # -- docs -------------------------------------------------------------
    problem_base_uri: str = "https://docs.musicintel.dev/problems"

    @field_validator("max_upload_bytes")
    @classmethod
    def _positive_upload(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("max_upload_bytes must be > 0")
        return v

    @field_validator("max_decode_seconds")
    @classmethod
    def _positive_decode(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("max_decode_seconds must be > 0")
        return v

    @property
    def resolved_api_keys_source(self) -> str:
        """"auto" made concrete: database when one is configured."""
        if self.api_keys_source != "auto":
            return self.api_keys_source
        return "database" if self.database_url else "config"

    @property
    def artifact_storage_enabled(self) -> bool:
        return bool(self.artifact_storage_url)

    @property
    def sync_catalog_list(self) -> list[str]:
        return [c.strip() for c in (self.sync_catalogs or "").split(",") if c.strip()]

    @property
    def persistence_enabled(self) -> bool:
        return bool(self.database_url)

    def load_api_key_records(self) -> list[dict]:
        """Raw key records from whichever source is configured."""
        if self.api_keys_file is not None:
            text = Path(self.api_keys_file).read_text()
        elif self.api_keys:
            text = self.api_keys
        else:
            return []
        data = json.loads(text)
        if not isinstance(data, list):
            raise ValueError("API key configuration must be a JSON list")
        return data


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


__all__ = ["Settings", "get_settings"]
