"""Redis rate limiting and audio-second quotas (S6).

Two independent limits, because they price two different things:

  * **Request rate** -- a token bucket per API key. Smooths bursts without
    punishing a client that sends ten requests in one second and then idles.
  * **Audio seconds per day** -- a counter per tenant. This is the quantity that
    actually costs money to serve, and it is the one a billing plan is written
    against. A caller sending 30-second clips is thirty times more expensive
    than one sending 1-second clips at the same request rate.

Both are evaluated in Lua so the check and the update are one atomic operation.
Doing it in two round trips would let concurrent workers each read "under the
limit" and each allow a request.

**On Redis failure the limiter fails CLOSED** -- 503, not "allow". An
availability incident on Redis must not silently become an unmetered-traffic
incident on the API. That choice is deliberate and is the opposite of what a
cache would do, because this is not a cache.

Clocks: `now` is supplied by the caller rather than read inside Lua, which keeps
the scripts deterministic and portable across Redis and the in-process fake used
by tests. Multiple API workers therefore need roughly synchronised clocks --
true of any container platform running NTP, and the failure mode is a slightly
generous or slightly strict bucket, not a broken one.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import redis.asyncio as aioredis
from redis.exceptions import RedisError

from musicintel.api.config import Settings

# -- token bucket ---------------------------------------------------------
_BUCKET_LUA = """
local key   = KEYS[1]
local rate  = tonumber(ARGV[1])
local burst = tonumber(ARGV[2])
local now   = tonumber(ARGV[3])
local cost  = tonumber(ARGV[4])

local state  = redis.call('HMGET', key, 'tokens', 'ts')
local tokens = tonumber(state[1])
local ts     = tonumber(state[2])
if tokens == nil or ts == nil then
  tokens = burst
  ts = now
end

local elapsed = now - ts
if elapsed < 0 then elapsed = 0 end
tokens = math.min(burst, tokens + elapsed * rate)

local allowed = 0
local retry = 0
if tokens >= cost then
  tokens = tokens - cost
  allowed = 1
else
  retry = math.ceil((cost - tokens) / rate)
  if retry < 1 then retry = 1 end
end

redis.call('HSET', key, 'tokens', tokens, 'ts', now)
redis.call('EXPIRE', key, math.ceil(burst / rate) + 60)
return {allowed, retry, tostring(tokens)}
"""

# -- daily audio-second quota --------------------------------------------
# Checks the *current* total against the limit and, when under, adds `cost`.
# Returning the pre-increment total lets the caller report remaining budget.
_QUOTA_LUA = """
local key   = KEYS[1]
local limit = tonumber(ARGV[1])
local cost  = tonumber(ARGV[2])
local ttl   = tonumber(ARGV[3])

local used = tonumber(redis.call('GET', key))
if used == nil then used = 0 end

if used >= limit then
  return {0, used, limit}
end

local total = redis.call('INCRBYFLOAT', key, cost)
redis.call('EXPIRE', key, ttl)
return {1, tonumber(total), limit}
"""


@dataclass(frozen=True)
class LimitDecision:
    allowed: bool
    retry_after: int = 0
    remaining: float = 0.0
    limit: float = 0.0
    used: float = 0.0


class RateLimiterUnavailable(Exception):
    """Redis could not be reached. Callers turn this into a 503."""


def build_redis(settings: Settings):
    """Async Redis client, or None when limiting is switched off."""
    if not settings.rate_limit_enabled:
        return None
    return aioredis.from_url(
        settings.redis_url,
        encoding="utf-8",
        decode_responses=False,
        socket_connect_timeout=2,
        socket_timeout=2,
        health_check_interval=30,
    )


class RateLimiter:
    def __init__(self, redis, *, enabled: bool = True) -> None:
        self._redis = redis
        self._enabled = enabled and redis is not None
        self._bucket = None
        self._quota = None

    @property
    def enabled(self) -> bool:
        return self._enabled

    def _scripts(self):
        if self._bucket is None:
            self._bucket = self._redis.register_script(_BUCKET_LUA)
            self._quota = self._redis.register_script(_QUOTA_LUA)
        return self._bucket, self._quota

    async def check_request_rate(
        self, key_id: str, *, per_minute: int, burst: int, cost: float = 1.0
    ) -> LimitDecision:
        if not self._enabled:
            return LimitDecision(allowed=True, limit=per_minute)
        bucket, _ = self._scripts()
        rate = per_minute / 60.0
        try:
            allowed, retry, tokens = await bucket(
                keys=[f"musicintel:rl:{key_id}"],
                args=[rate, max(burst, 1), time.time(), cost],
            )
        except RedisError as exc:
            raise RateLimiterUnavailable(str(exc)) from exc
        return LimitDecision(
            allowed=bool(int(allowed)),
            retry_after=int(retry),
            remaining=float(tokens),
            limit=float(per_minute),
        )

    async def consume_audio_seconds(
        self, tenant: str, *, seconds: float, daily_limit: int, now: float | None = None
    ) -> LimitDecision:
        """Charge `seconds` against today's budget, atomically."""
        if not self._enabled:
            return LimitDecision(allowed=True, limit=daily_limit)
        _, quota = self._scripts()
        stamp = time.gmtime(now if now is not None else time.time())
        day = f"{stamp.tm_year:04d}{stamp.tm_mon:02d}{stamp.tm_mday:02d}"
        # Two days of TTL so a request near midnight cannot lose its counter.
        try:
            allowed, used, limit = await quota(
                keys=[f"musicintel:quota:{tenant}:{day}"],
                args=[daily_limit, seconds, 172800],
            )
        except RedisError as exc:
            raise RateLimiterUnavailable(str(exc)) from exc
        used_f, limit_f = float(used), float(limit)
        return LimitDecision(
            allowed=bool(int(allowed)),
            retry_after=_seconds_until_utc_midnight(now),
            remaining=max(0.0, limit_f - used_f),
            limit=limit_f,
            used=used_f,
        )

    async def peek_audio_seconds(self, tenant: str, *, daily_limit: int) -> LimitDecision:
        """Current usage without charging. Used to refuse before decoding."""
        if not self._enabled:
            return LimitDecision(allowed=True, limit=daily_limit)
        stamp = time.gmtime()
        day = f"{stamp.tm_year:04d}{stamp.tm_mon:02d}{stamp.tm_mday:02d}"
        try:
            raw = await self._redis.get(f"musicintel:quota:{tenant}:{day}")
        except RedisError as exc:
            raise RateLimiterUnavailable(str(exc)) from exc
        used = float(raw) if raw else 0.0
        return LimitDecision(
            allowed=used < daily_limit,
            retry_after=_seconds_until_utc_midnight(None),
            remaining=max(0.0, daily_limit - used),
            limit=float(daily_limit),
            used=used,
        )

    async def ping(self) -> bool:
        if not self._enabled:
            return True
        try:
            await self._redis.ping()
            return True
        except RedisError:
            return False


def _seconds_until_utc_midnight(now: float | None) -> int:
    now = now if now is not None else time.time()
    stamp = time.gmtime(now)
    elapsed = stamp.tm_hour * 3600 + stamp.tm_min * 60 + stamp.tm_sec
    return max(1, 86400 - elapsed)


__all__ = [
    "LimitDecision", "RateLimiter", "RateLimiterUnavailable", "build_redis",
]
