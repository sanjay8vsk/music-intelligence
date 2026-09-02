"""Stage 3 API tests.

The recognition core is frozen and tested elsewhere. What is under test here is
everything wrapped around it: authentication, tenant isolation, the decode
sandbox, limits, and the error contract. Where a test asserts a recognition
outcome it does so only to prove the wiring reaches the right catalog -- accuracy
belongs to the benchmark suite.
"""

from __future__ import annotations

import io
import json

import fakeredis.aioredis
import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from musicintel.api.app import create_app
from musicintel.api.auth import hash_key
from musicintel.api.config import Settings
from musicintel.api.ratelimit import RateLimiter
from musicintel.catalog.ingest import build_catalog_index, ingest_directory
from musicintel.catalog.store import CatalogStore

SR = 11025

# Import roots heavy enough that paying for one inside a request is the
# regression this suite guards. numba additionally compiles on first call.
LAZY_IMPORT_COST = frozenset({
    "librosa", "numba", "llvmlite", "scipy", "soxr", "soundfile", "sklearn",
})
ACME_KEY = "sk_test_acme_0000000000000000"
GLOBEX_KEY = "sk_test_globex_00000000000000"
REVOKED_KEY = "sk_test_revoked_0000000000000"
NARROW_KEY = "sk_test_narrow_00000000000000"
TINY_KEY = "sk_test_tiny_0000000000000000"


def _tone(seconds=12.0, seed=0):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, seconds, int(SR * seconds), endpoint=False)
    wob = 600.0 + 200.0 * np.sin(2 * np.pi * 0.5 * t + seed)
    return (0.50 * np.sin(2 * np.pi * (440.0 + 7 * seed) * t)
            + 0.30 * np.sin(2 * np.pi * wob * t)
            + 0.20 * np.sin(2 * np.pi * 1500.0 * t)
            + 0.02 * rng.standard_normal(t.size)).astype(np.float32)


def _wav_bytes(samples, sr=SR):
    buf = io.BytesIO()
    sf.write(buf, samples, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("api")
    store = CatalogStore(tmp / "store")
    sources = {}
    for cid, seeds in (("acme", [1, 2, 3]), ("globex", [11, 12, 13])):
        d = tmp / "audio" / cid
        d.mkdir(parents=True, exist_ok=True)
        for s in seeds:
            sf.write(d / f"{cid}_{s}.wav", _tone(seed=s), SR, subtype="PCM_16")
            sources[(cid, s)] = _tone(seed=s)
        r = ingest_directory(d)
        store.save(r.catalog, build_catalog_index(r.catalog, r.fingerprints),
                   catalog_id=cid)
    return tmp, store, sources


@pytest.fixture(scope="module")
def keys_file(tmp_path_factory):
    path = tmp_path_factory.mktemp("keys") / "keys.json"
    path.write_text(json.dumps([
        {"key_id": "k_acme", "tenant": "acme", "key_sha256": hash_key(ACME_KEY),
         "catalogs": ["acme"], "rate_limit_per_minute": 600,
         "rate_limit_burst": 200, "audio_seconds_per_day": 100000},
        {"key_id": "k_globex", "tenant": "globex", "key_sha256": hash_key(GLOBEX_KEY),
         "catalogs": ["globex"], "rate_limit_per_minute": 600,
         "rate_limit_burst": 200, "audio_seconds_per_day": 100000},
        {"key_id": "k_revoked", "tenant": "acme", "key_sha256": hash_key(REVOKED_KEY),
         "catalogs": ["acme"], "active": False},
        {"key_id": "k_narrow", "tenant": "acme", "key_sha256": hash_key(NARROW_KEY),
         "catalogs": ["acme"], "scopes": ["catalogs:read"],
         "rate_limit_per_minute": 600, "rate_limit_burst": 200},
        {"key_id": "k_tiny", "tenant": "tiny", "key_sha256": hash_key(TINY_KEY),
         "catalogs": ["acme"], "rate_limit_per_minute": 6, "rate_limit_burst": 3,
         "audio_seconds_per_day": 100000},
    ]))
    return path


@pytest.fixture(scope="module")
def client(world, keys_file):
    tmp, store, _ = world
    settings = Settings(
        catalog_root=store.root, api_keys_file=keys_file,
        environment="test", log_json=False, max_upload_bytes=1024 * 1024,
        max_decode_seconds=30.0,
    )
    app = create_app(settings)
    app.state.redis = fakeredis.aioredis.FakeRedis(decode_responses=False)
    with TestClient(app) as c:
        yield c


@pytest.fixture(autouse=True)
def _fresh_limits(client):
    """Each test starts with empty buckets so limit tests cannot cross-talk."""
    client.app.state.limiter = RateLimiter(
        fakeredis.aioredis.FakeRedis(decode_responses=False)
    )


def _auth(key=ACME_KEY):
    return {"Authorization": f"Bearer {key}"}


def _identify(client, audio, catalog="acme", key=ACME_KEY, filename="q.wav"):
    return client.post(
        "/v1/identify",
        files={"file": (filename, _wav_bytes(audio), "audio/wav")},
        data={"catalog_id": catalog},
        headers=_auth(key),
    )


# ------------------------------------------------------------------ service --
class TestServiceEndpoints:
    def test_health_needs_no_key_and_reports_readiness(self, client):
        r = client.get("/v1/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert body["recognition_ready"] is True

    def test_version_reports_formats_not_secrets(self, client):
        r = client.get("/v1/version")
        assert r.status_code == 200
        body = r.json()
        assert body["api_version"] == "v1"
        assert body["fingerprint_format_version"] == 1
        assert body["index_format_version"] == 1
        assert "key" not in json.dumps(body).lower()

    def test_metrics_are_exposed_in_prometheus_format(self, client):
        client.get("/v1/health")
        r = client.get("/v1/metrics")
        assert r.status_code == 200
        assert "musicintel_http_requests_total" in r.text

    def test_unknown_route_is_a_problem_document(self, client):
        r = client.get("/v1/nope")
        assert r.status_code == 404
        assert r.headers["content-type"].startswith("application/problem+json")
        assert r.json()["title"] == "Not Found"


# --------------------------------------------------------------------- auth --
class TestAuthentication:
    def test_missing_key_is_401(self, client):
        r = client.get("/v1/catalogs")
        assert r.status_code == 401
        assert r.headers["content-type"].startswith("application/problem+json")
        assert "www-authenticate" in {k.lower() for k in r.headers}

    @pytest.mark.parametrize("header", [
        {"Authorization": "Bearer wrong-key"},
        {"Authorization": "Basic abc"},
        {"Authorization": "Bearer "},
        {"X-API-Key": "nope"},
        {"Authorization": f"Bearer {REVOKED_KEY}"},
    ])
    def test_bad_or_revoked_keys_are_401(self, client, header):
        assert client.get("/v1/catalogs", headers=header).status_code == 401

    def test_x_api_key_header_also_works(self, client):
        r = client.get("/v1/catalogs", headers={"X-API-Key": ACME_KEY})
        assert r.status_code == 200

    def test_error_body_never_echoes_the_key(self, client):
        r = client.get("/v1/catalogs", headers={"Authorization": "Bearer sk_live_secret"})
        assert "sk_live_secret" not in r.text

    def test_scope_is_enforced(self, client, world):
        _, _, sources = world
        r = _identify(client, sources[("acme", 1)][:SR * 5], key=NARROW_KEY)
        assert r.status_code == 403
        assert r.json()["title"] == "Forbidden"


# ---------------------------------------------------------------- isolation --
class TestTenantIsolation:
    def test_listing_shows_only_permitted_catalogs(self, client):
        acme = client.get("/v1/catalogs", headers=_auth(ACME_KEY)).json()
        globex = client.get("/v1/catalogs", headers=_auth(GLOBEX_KEY)).json()
        assert [c["catalog_id"] for c in acme["catalogs"]] == ["acme"]
        assert [c["catalog_id"] for c in globex["catalogs"]] == ["globex"]

    def test_other_tenants_catalog_is_indistinguishable_from_missing(self, client):
        forbidden = client.get("/v1/catalogs/globex", headers=_auth(ACME_KEY))
        missing = client.get("/v1/catalogs/does-not-exist", headers=_auth(ACME_KEY))
        assert forbidden.status_code == missing.status_code == 404
        # Identical bodies: a probe cannot tell "exists but forbidden" from
        # "does not exist".
        assert forbidden.json()["detail"] == missing.json()["detail"]

    def test_identify_against_another_tenants_catalog_is_404(self, client, world):
        _, _, sources = world
        r = _identify(client, sources[("globex", 11)][:SR * 5],
                      catalog="globex", key=ACME_KEY)
        assert r.status_code == 404

    def test_audio_from_another_catalog_does_not_match(self, client, world):
        """The structural guarantee: globex audio cannot match inside acme."""
        _, _, sources = world
        r = _identify(client, sources[("globex", 11)][:SR * 5],
                      catalog="acme", key=ACME_KEY)
        assert r.status_code == 200
        assert r.json()["decision"] == "no_match"


# ----------------------------------------------------------------- catalogs --
class TestCatalogs:
    def test_detail_lists_tracks_without_leaking_paths(self, client):
        r = client.get("/v1/catalogs/acme", headers=_auth(ACME_KEY))
        assert r.status_code == 200
        body = r.json()
        assert body["track_count"] == 3
        assert len(body["content_hash"]) == 64
        assert "source_path" not in r.text and "/tmp" not in r.text

    def test_pagination(self, client):
        r = client.get("/v1/catalogs/acme?limit=2&offset=1", headers=_auth(ACME_KEY))
        assert r.status_code == 200 and r.json()["returned"] == 2

    @pytest.mark.parametrize("bad", [
        "../etc", "%2e%2e/etc", "a" * 100, "has space", ".hidden", "-lead",
        "semi;colon", "null%00byte",
    ])
    def test_malformed_catalog_ids_are_rejected(self, client, bad):
        r = client.get(f"/v1/catalogs/{bad}", headers=_auth(ACME_KEY))
        assert r.status_code in (404, 422), f"{bad!r} -> {r.status_code}"
        assert "/tmp" not in r.text and "store" not in r.text

    def test_trailing_slash_redirects_to_the_listing(self, client):
        r = client.get("/v1/catalogs/", headers=_auth(ACME_KEY),
                       follow_redirects=False)
        assert r.status_code == 307
        assert r.headers["location"].endswith("/v1/catalogs")


# ----------------------------------------------------------------- identify --
class TestIdentify:
    def test_matches_a_known_recording(self, client, world):
        _, _, sources = world
        audio = sources[("acme", 2)]
        r = _identify(client, audio[len(audio) // 2: len(audio) // 2 + SR * 5])
        assert r.status_code == 200
        body = r.json()
        assert body["decision"] == "match"
        assert body["match"]["track_id"] == "acme_2"
        assert body["catalog_id"] == "acme"
        assert body["evidence_score"] >= body["threshold"]
        assert body["query_duration_seconds"] == pytest.approx(5.0, abs=0.05)

    def test_response_carries_no_confidence(self, client, world):
        """`evidence_score` is a rate. The API must not imply otherwise."""
        _, _, sources = world
        r = _identify(client, sources[("acme", 1)][:SR * 5])
        body = r.json()
        assert "confidence" not in body and "probability" not in body
        assert json.dumps(body).count("confidence") == 0

    def test_unrelated_audio_is_no_match(self, client):
        rng = np.random.default_rng(99)
        r = _identify(client, (0.1 * rng.standard_normal(SR * 5)).astype(np.float32))
        assert r.status_code == 200
        body = r.json()
        assert body["decision"] == "no_match" and body["match"] is None

    def test_rate_limit_headers_are_present(self, client, world):
        _, _, sources = world
        r = _identify(client, sources[("acme", 1)][:SR * 5])
        assert "x-ratelimit-limit" in {k.lower() for k in r.headers}
        assert "x-quota-audio-seconds-remaining" in {k.lower() for k in r.headers}

    def test_missing_catalog_field_is_a_validation_problem(self, client, world):
        _, _, sources = world
        r = client.post(
            "/v1/identify",
            files={"file": ("q.wav", _wav_bytes(sources[("acme", 1)][:SR]), "audio/wav")},
            headers=_auth(),
        )
        assert r.status_code == 422
        assert r.headers["content-type"].startswith("application/problem+json")
        assert r.json()["title"] == "Validation Failed"


# ------------------------------------------------------- upload / decode S1-S4 --
class TestUploadSafety:
    @pytest.mark.parametrize("payload,label", [
        (b"", "empty"),
        (b"not audio at all" * 100, "plain text"),
        (b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n" + b"\x00" * 500, "pdf"),
        (b"MZ\x90\x00" + b"\x00" * 500, "windows executable"),
        (b"\x1f\x8b\x08\x00" + b"\x00" * 500, "gzip"),
        (b"RIFF\x24\x00\x00\x00AVI LIST", "avi in a RIFF container"),
        (b"\x00" * 4096, "nul bytes"),
    ])
    def test_non_audio_is_refused_and_the_worker_survives(self, client, payload, label):
        r = client.post(
            "/v1/identify",
            files={"file": ("x.wav", payload, "audio/wav")},
            data={"catalog_id": "acme"}, headers=_auth(),
        )
        assert r.status_code in (415, 422), f"{label}: got {r.status_code}"
        assert r.headers["content-type"].startswith("application/problem+json")
        # The API is still serving after the rejection.
        assert client.get("/v1/health").json()["status"] == "ok"

    def test_truncated_audio_is_refused_not_crashed(self, client, world):
        _, _, sources = world
        good = _wav_bytes(sources[("acme", 1)][:SR * 5])
        r = client.post(
            "/v1/identify",
            files={"file": ("x.wav", good[:len(good) // 3], "audio/wav")},
            data={"catalog_id": "acme"}, headers=_auth(),
        )
        # Either it decodes the surviving frames or it refuses; it must not 500.
        assert r.status_code in (200, 415, 422)

    def test_declared_extension_and_content_type_are_not_trusted(self, client):
        r = client.post(
            "/v1/identify",
            files={"file": ("song.mp3", b"%PDF-1.4 pretending", "audio/mpeg")},
            data={"catalog_id": "acme"}, headers=_auth(),
        )
        assert r.status_code in (415, 422)

    def test_oversized_upload_is_413(self, client):
        big = b"RIFF" + b"\x00" * (2 * 1024 * 1024)
        r = client.post(
            "/v1/identify",
            files={"file": ("big.wav", big, "audio/wav")},
            data={"catalog_id": "acme"}, headers=_auth(),
        )
        assert r.status_code == 413
        assert r.json()["title"] == "Payload Too Large"

    def test_decompression_bomb_is_refused_with_bounded_memory(self, client):
        """82 KB of FLAC that would decode to ~100 MiB of PCM."""
        buf = io.BytesIO()
        sf.write(buf, np.zeros(44100 * 600, dtype=np.float32), 44100, format="FLAC")
        bomb = buf.getvalue()
        assert len(bomb) < 1024 * 1024, "bomb must fit under the upload cap"
        r = client.post(
            "/v1/identify",
            files={"file": ("bomb.flac", bomb, "audio/flac")},
            data={"catalog_id": "acme"}, headers=_auth(),
        )
        assert r.status_code == 413
        assert client.get("/v1/health").json()["status"] == "ok"

    def test_many_channel_file_is_refused(self, client):
        buf = io.BytesIO()
        sf.write(buf, np.zeros((SR, 16), dtype=np.float32), SR, format="WAV")
        r = client.post(
            "/v1/identify",
            files={"file": ("multi.wav", buf.getvalue(), "audio/wav")},
            data={"catalog_id": "acme"}, headers=_auth(),
        )
        assert r.status_code == 422


# -------------------------------------------------------------------- limits --
class TestLimits:
    def test_request_rate_breach_returns_429_with_retry_after(self, client, world):
        """A key with burst 3 gets exactly 3 through, then 429s."""
        _, _, sources = world
        audio = sources[("acme", 1)][:SR * 2]
        codes = [_identify(client, audio, key=TINY_KEY).status_code for _ in range(6)]
        assert codes[:3] == [200, 200, 200], codes
        assert 429 in codes[3:], codes

        refused = _identify(client, audio, key=TINY_KEY)
        assert refused.status_code == 429
        body = refused.json()
        assert body["title"] == "Too Many Requests"
        assert body["status"] == 429
        assert refused.headers["content-type"].startswith("application/problem+json")
        assert int(refused.headers["retry-after"]) >= 1
        assert body["retry_after"] >= 1

    def test_rate_limits_are_per_key_not_global(self, client, world):
        """Exhausting one key must not refuse another."""
        _, _, sources = world
        audio = sources[("acme", 1)][:SR * 2]
        for _ in range(6):
            _identify(client, audio, key=TINY_KEY)
        assert _identify(client, audio, key=TINY_KEY).status_code == 429
        assert _identify(client, audio, key=ACME_KEY).status_code == 200

    def test_limiter_failure_is_503_not_silent_allow(self, client, world):
        """Redis down must fail closed. An outage is not a free-traffic event."""
        from musicintel.api.ratelimit import RateLimiter, RateLimiterUnavailable

        class Broken(RateLimiter):
            async def check_request_rate(self, *a, **k):
                raise RateLimiterUnavailable("redis is down")

        _, _, sources = world
        original = client.app.state.limiter
        client.app.state.limiter = Broken(
            fakeredis.aioredis.FakeRedis(decode_responses=False)
        )
        try:
            r = _identify(client, sources[("acme", 1)][:SR * 2])
            assert r.status_code == 503
            assert r.json()["title"] == "Service Unavailable"
        finally:
            client.app.state.limiter = original

    def test_audio_second_quota_breach_returns_429(self, client, world, keys_file):
        """A tenant with a tiny daily budget is cut off after a few seconds."""
        _, store, sources = world
        settings = Settings(
            catalog_root=store.root, api_keys_file=keys_file, environment="test",
            log_json=False,
        )
        small_key = "sk_test_small_000000000000000"
        keys = json.loads(keys_file.read_text()) + [{
            "key_id": "k_small", "tenant": "small", "key_sha256": hash_key(small_key),
            "catalogs": ["acme"], "rate_limit_per_minute": 600,
            "rate_limit_burst": 200, "audio_seconds_per_day": 6,
        }]
        path = keys_file.parent / "keys_small.json"
        path.write_text(json.dumps(keys))
        settings = Settings(catalog_root=store.root, api_keys_file=path,
                            environment="test", log_json=False)
        app = create_app(settings)
        app.state.redis = fakeredis.aioredis.FakeRedis(decode_responses=False)
        with TestClient(app) as c:
            audio = sources[("acme", 1)][:SR * 5]
            first = c.post("/v1/identify",
                           files={"file": ("q.wav", _wav_bytes(audio), "audio/wav")},
                           data={"catalog_id": "acme"},
                           headers={"Authorization": f"Bearer {small_key}"})
            assert first.status_code == 200
            second = c.post("/v1/identify",
                            files={"file": ("q.wav", _wav_bytes(audio), "audio/wav")},
                            data={"catalog_id": "acme"},
                            headers={"Authorization": f"Bearer {small_key}"})
            assert second.status_code == 200  # 10 s used of 6 -> next one refused
            third = c.post("/v1/identify",
                           files={"file": ("q.wav", _wav_bytes(audio), "audio/wav")},
                           data={"catalog_id": "acme"},
                           headers={"Authorization": f"Bearer {small_key}"})
            assert third.status_code == 429
            body = third.json()
            assert body["title"] == "Quota Exceeded"
            assert body["limit"] == 6
            assert int(third.headers["retry-after"]) >= 1


# ------------------------------------------------------------------ openapi --
class TestOpenAPI:
    def test_spec_is_served_and_describes_every_route(self, client):
        r = client.get("/v1/openapi.json")
        assert r.status_code == 200
        spec = r.json()
        assert spec["openapi"].startswith("3.")
        for path in ("/v1/health", "/v1/version", "/v1/catalogs",
                     "/v1/catalogs/{catalog_id}", "/v1/identify"):
            assert path in spec["paths"], path

    def test_identify_declares_its_error_responses(self, client):
        spec = client.get("/v1/openapi.json").json()
        responses = spec["paths"]["/v1/identify"]["post"]["responses"]
        for code in ("401", "403", "404", "413", "415", "422", "429", "503"):
            assert code in responses, code

    def test_no_schema_declares_a_confidence_field(self, client):
        """Prose may say there is no confidence; no *field* may be one."""
        spec = client.get("/v1/openapi.json").json()
        for name, schema in spec["components"]["schemas"].items():
            props = set(schema.get("properties", {}))
            assert not props & {"confidence", "probability", "score_pct"}, name

    def test_identify_response_fields_are_exactly_the_contract(self, client):
        spec = client.get("/v1/openapi.json").json()
        props = set(spec["components"]["schemas"]["IdentifyResponse"]["properties"])
        assert props == {
            "decision", "catalog_id", "match", "evidence_score", "threshold",
            "aligned_landmarks", "query_landmarks", "stage", "escalated",
            "query_duration_seconds", "latency_ms",
        }


# ------------------------------------------------------- contract hardening --
class TestErrorContract:
    """The published spec must describe the errors that are actually sent."""

    def test_every_error_response_is_declared_as_problem_json(self, client):
        spec = client.get("/v1/openapi.json").json()
        checked = 0
        for path, operations in spec["paths"].items():
            for method, operation in operations.items():
                for status, response in operation.get("responses", {}).items():
                    if not status[:1].isdigit() or int(status[0]) < 4:
                        continue
                    content = response.get("content", {})
                    assert list(content) == ["application/problem+json"], \
                        f"{method.upper()} {path} {status}: {list(content)}"
                    assert content["application/problem+json"]["schema"] == {
                        "$ref": "#/components/schemas/ProblemResponse"
                    }
                    checked += 1
        assert checked > 10, "expected many documented error responses"

    def test_fastapis_default_validation_schema_is_gone(self, client):
        """It documents a body shape this service never returns."""
        schemas = client.get("/v1/openapi.json").json()["components"]["schemas"]
        assert "HTTPValidationError" not in schemas
        assert "ValidationError" not in schemas
        assert "ProblemResponse" in schemas

    def test_every_error_actually_sent_matches_the_declared_media_type(self, client):
        for call in (
            lambda: client.get("/v1/catalogs"),                        # 401
            lambda: client.get("/v1/catalogs/nope", headers=_auth()),  # 404
            lambda: client.post("/v1/identify", headers=_auth()),      # 422
        ):
            r = call()
            assert r.status_code >= 400
            assert r.headers["content-type"].startswith("application/problem+json")
            body = r.json()
            assert set(body) >= {"type", "title", "status"}
            assert body["status"] == r.status_code

    def test_identify_documents_the_statuses_it_can_return(self, client):
        spec = client.get("/v1/openapi.json").json()
        declared = set(spec["paths"]["/v1/identify"]["post"]["responses"])
        assert {"400", "401", "403", "404", "413", "415", "422", "429",
                "500", "503"} <= declared


class TestStrictQueryParameters:
    """A typo in a query parameter must fail loudly, not silently paginate."""

    def test_unknown_query_parameter_is_rejected(self, client):
        r = client.get("/v1/catalogs/acme?limt=2", headers=_auth(ACME_KEY))
        assert r.status_code == 422
        body = r.json()
        assert body["errors"][0]["kind"] == "unexpected_parameter"
        assert body["errors"][0]["location"] == ["query", "limt"]

    def test_declared_query_parameters_still_work(self, client):
        r = client.get("/v1/catalogs/acme?limit=2&offset=0", headers=_auth(ACME_KEY))
        assert r.status_code == 200 and r.json()["returned"] == 2


class TestUploadShapes:
    """The upload field accepts both multipart spellings clients actually send."""

    @staticmethod
    def _multipart(blob: bytes, disposition_extra: str, catalog: str = "acme"):
        b = "abc123boundary"
        return b, (
            f'--{b}\r\nContent-Disposition: form-data; name="file"'
            f'{disposition_extra}\r\n\r\n'
        ).encode() + blob + (
            f'\r\n--{b}\r\nContent-Disposition: form-data; '
            f'name="catalog_id"\r\n\r\n{catalog}\r\n--{b}--\r\n'
        ).encode()

    def test_an_empty_filename_still_carries_binary_intact(self, client, world):
        """`filename=""` is what makes a part binary -- content-type does not."""
        _, _, sources = world
        blob = _wav_bytes(sources[("acme", 1)][:SR * 5])
        boundary, body = self._multipart(blob, '; filename=""')
        r = client.post("/v1/identify", content=body, headers={
            **_auth(), "Content-Type": f"multipart/form-data; boundary={boundary}"})
        assert r.status_code == 200
        assert r.json()["decision"] == "match"

    def test_part_without_a_filename_reaches_the_decoder(self, client, world):
        """Structurally accepted, then refused on content -- not "Field required".

        Starlette parses a part with no `filename=` as a *text* field, so binary
        audio is UTF-8 mangled before any application code sees it. Nothing can
        recover those bytes. What matters is the failure mode: the caller gets
        415 "not a recognised audio container", which points at the real problem,
        rather than 422 "file: Field required" for a field they plainly sent.
        """
        _, _, sources = world
        blob = _wav_bytes(sources[("acme", 1)][:SR * 5])
        boundary, body = self._multipart(blob, "")
        r = client.post("/v1/identify", content=body, headers={
            **_auth(), "Content-Type": f"multipart/form-data; boundary={boundary}"})
        assert r.status_code == 415
        assert r.json()["title"] == "Unsupported Media Type"

    def test_zero_byte_upload_is_rejected_by_the_declared_schema(self, client):
        r = client.post(
            "/v1/identify", files={"file": ("q.wav", b"", "audio/wav")},
            data={"catalog_id": "acme"}, headers=_auth(),
        )
        assert r.status_code == 422

    def test_unparseable_multipart_is_400_not_500(self, client):
        r = client.post(
            "/v1/identify", content=b"--wrong\r\ngarbage\r\n--wrong--\r\n",
            headers={**_auth(),
                     "Content-Type": "multipart/form-data; boundary=different"},
        )
        assert r.status_code == 400
        assert r.headers["content-type"].startswith("application/problem+json")


class TestWarmUp:
    """Cold-start cost belongs at construction, not in the first request."""

    def test_service_is_warm_before_it_serves(self, client):
        service = client.app.state.service
        assert service.warm_up_seconds > 0

    def test_first_request_pays_no_lazy_import_penalty(self, client, world):
        """Regression: an unwarmed first identify measured 1,735 ms.

        Asserted structurally rather than by wall clock. A `< 1000 ms` bound on
        the first request fails on a loaded machine even when warm-up worked
        perfectly, and passes on an idle one even if warm-up had quietly stopped
        covering a library -- it tests the symptom, and only sometimes. The cause
        is what matters: the first request must import nothing heavy and compile
        nothing, because construction already did both.
        """
        import sys
        from musicintel.recognition.matcher import _best_cluster_compiled

        _, _, sources = world
        audio = sources[("acme", 1)][:SR * 5]

        modules_before = set(sys.modules)
        compiled_before = len(_best_cluster_compiled.signatures)
        assert compiled_before > 0, "warm-up did not compile the matcher kernel"

        r = _identify(client, audio)

        assert r.status_code == 200
        lazily_imported = sorted(
            m for m in set(sys.modules) - modules_before
            if m.split(".")[0] in LAZY_IMPORT_COST
        )
        assert not lazily_imported, (
            f"first request lazily imported {lazily_imported}; "
            f"RecognitionService._warm_up must force these at construction")
        assert len(_best_cluster_compiled.signatures) == compiled_before, (
            "first request triggered numba compilation; warm_up_matcher() "
            "should have paid it already")
