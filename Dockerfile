# syntax=docker/dockerfile:1
#
# Two stages so the runtime image carries no compiler and no build cache.
#
# NOT installed on purpose: ffmpeg. The decode sandbox uses libsndfile (bundled
# in the soundfile wheel) and refuses anything it will not take, rather than
# falling back to a second, much larger parser for untrusted input. See
# docs/api.md for what that costs in format coverage.

# ---------------------------------------------------------------- builder --
FROM python:3.11-slim-bookworm AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY pyproject.toml README.md ./
COPY musicintel ./musicintel

# A venv, so the runtime stage copies one self-contained tree.
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --upgrade pip setuptools wheel \
 && pip install '.[api,db]' \
 && pip install "gunicorn>=22.0"

# ---------------------------------------------------------------- runtime --
FROM python:3.11-slim-bookworm AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    # numba writes no JIT cache (cache=False); warm_up() pays compilation once
    # at start-up instead, so the package directory stays read-only.
    NUMBA_CACHE_DIR=/tmp \
    MUSICINTEL_ENVIRONMENT=production \
    MUSICINTEL_LOG_JSON=true \
    MUSICINTEL_CATALOG_ROOT=/var/lib/musicintel/catalogs

# libgomp is required by numba/llvmlite at runtime; curl is for the healthcheck.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 curl \
 && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv

# Unprivileged. The API worker never writes to disk, and the decode sandbox
# cannot: it sets RLIMIT_FSIZE to 0 before it parses anything.
RUN useradd --system --create-home --uid 10001 musicintel \
 && mkdir -p /var/lib/musicintel/catalogs \
 && chown -R musicintel:musicintel /var/lib/musicintel

WORKDIR /srv
USER musicintel

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
  CMD curl -fsS http://127.0.0.1:8000/v1/health || exit 1

# One uvicorn worker per process, several processes: the matcher's compiled
# kernel holds the GIL, so concurrency comes from processes, not threads.
# Each worker pays ~3 s of warm-up at start-up, hence the generous start period
# above and the long graceful timeout below.
CMD ["gunicorn", "musicintel.api.app:create_app", \
     "--factory", \
     "--worker-class", "uvicorn.workers.UvicornWorker", \
     "--workers", "2", \
     "--bind", "0.0.0.0:8000", \
     "--timeout", "60", \
     "--graceful-timeout", "30", \
     "--access-logfile", "-", \
     "--error-logfile", "-"]
