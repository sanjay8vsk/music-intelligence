"""Prometheus metrics.

Deliberately small. Four families cover the questions that matter operationally:
is it up, is it fast, is it rejecting things, and is recognition still finding
matches. Cardinality is bounded -- `catalog_id` is the only label sourced from
input, and it is validated against the store before it ever reaches a metric.
"""

from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

# A private registry rather than the global default: two app instances in one
# test process would otherwise collide on metric names.
REGISTRY = CollectorRegistry()

REQUESTS = Counter(
    "musicintel_http_requests_total",
    "HTTP requests by route and status.",
    ["method", "route", "status"],
    registry=REGISTRY,
)

REQUEST_DURATION = Histogram(
    "musicintel_http_request_duration_seconds",
    "Wall-clock time per request, including decode and recognition.",
    ["route"],
    # Bucketed around the 300 ms Stage 3 acceptance bar so p95 is readable
    # straight off the histogram without interpolation guesswork.
    buckets=(0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.3, 0.5, 1.0,
             2.5, 5.0, 10.0),
    registry=REGISTRY,
)

DECODE_DURATION = Histogram(
    "musicintel_decode_duration_seconds",
    "Sandboxed decode time.",
    buckets=(0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0, 5.0, 10.0),
    registry=REGISTRY,
)

RECOGNITION_DURATION = Histogram(
    "musicintel_recognition_duration_seconds",
    "Recognition time, excluding decode.",
    buckets=(0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0),
    registry=REGISTRY,
)

IDENTIFICATIONS = Counter(
    "musicintel_identifications_total",
    "Identify outcomes by decision and whether the speed cascade escalated.",
    ["catalog_id", "decision", "escalated"],
    registry=REGISTRY,
)

REJECTIONS = Counter(
    "musicintel_rejections_total",
    "Requests refused before recognition, by reason.",
    ["reason"],
    registry=REGISTRY,
)

AUDIO_SECONDS = Counter(
    "musicintel_audio_seconds_total",
    "Decoded audio seconds accepted, by tenant. The billing quantity.",
    ["tenant"],
    registry=REGISTRY,
)

USAGE_RECORDS = Counter(
    "musicintel_usage_records_total",
    "Durable usage records, by what happened to them. `dropped` being non-zero "
    "means billing history was lost and the database needs attention.",
    ["outcome"],                       # enqueued | written | dropped
    registry=REGISTRY,
)

CATALOGS_LOADED = Gauge(
    "musicintel_catalogs_loaded",
    "Catalog indexes currently resident in memory.",
    registry=REGISTRY,
)

RECOGNITION_READY = Gauge(
    "musicintel_recognition_ready",
    "1 once the matcher's compiled kernels are warm.",
    registry=REGISTRY,
)


def reset_for_tests() -> None:
    """Clear counters between tests without rebuilding the registry."""
    for metric in (REQUESTS, REQUEST_DURATION, DECODE_DURATION,
                   RECOGNITION_DURATION, IDENTIFICATIONS, REJECTIONS,
                   AUDIO_SECONDS, USAGE_RECORDS):
        metric.clear()


__all__ = [
    "AUDIO_SECONDS", "CATALOGS_LOADED", "DECODE_DURATION", "IDENTIFICATIONS",
    "RECOGNITION_DURATION", "REGISTRY", "REJECTIONS", "REQUESTS",
    "REQUEST_DURATION", "RECOGNITION_READY", "USAGE_RECORDS", "reset_for_tests",
]
