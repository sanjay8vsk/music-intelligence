"""Immutable, content-addressed artifact storage and boot synchronisation.

Real filesystem temporary directories throughout, and the real `CatalogStore`
verification path -- no mocks, and no fake S3 server, because the local backend
is the implementation under test rather than a stand-in for one.

Nothing here touches recognition. Artifacts are moved and verified; how they are
searched is unchanged.
"""

from __future__ import annotations

import json
import shutil

import numpy as np
import pytest
import soundfile as sf

from musicintel.catalog.ingest import build_catalog_index, ingest_directory
from musicintel.catalog.store import CatalogStore, CatalogStoreError
from musicintel.storage.artifacts import (
    ARTIFACT_MEMBERS,
    ArtifactConflict,
    ArtifactIncomplete,
    ArtifactManifest,
    ArtifactNotFound,
    artifact_key,
    read_artifact_version,
    validate_version,
)
from musicintel.storage.local import LocalArtifactStorage, storage_from_url
from musicintel.storage.sync import (
    SyncError, parse_pins, resolve_version, sync_all, sync_catalog,
)

SR = 11025


def _build_catalog(tmp_path, catalog_id: str, seeds=(1, 2)) -> CatalogStore:
    audio = tmp_path / "audio" / catalog_id
    audio.mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        rng = np.random.default_rng(seed)
        t = np.linspace(0, 8.0, int(SR * 8), endpoint=False)
        y = (0.5 * np.sin(2 * np.pi * (440 + 11 * seed) * t)
             + 0.3 * np.sin(2 * np.pi * (900 + 7 * seed) * t)
             + 0.02 * rng.standard_normal(t.size)).astype(np.float32)
        sf.write(audio / f"{catalog_id}_{seed}.wav", y, SR, subtype="PCM_16")
    store = CatalogStore(tmp_path / "store")
    r = ingest_directory(audio)
    store.save(r.catalog, build_catalog_index(r.catalog, r.fingerprints),
               catalog_id=catalog_id)
    return store


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    """One store holding two independent catalogs."""
    tmp = tmp_path_factory.mktemp("artifacts")
    store = _build_catalog(tmp, "acme", seeds=(1, 2))
    _build_catalog(tmp, "globex", seeds=(11, 12))
    return store


@pytest.fixture
def storage(tmp_path):
    return LocalArtifactStorage(tmp_path / "objstore")


def _publish(storage, store, catalog_id):
    version = read_artifact_version(store.path_for(catalog_id))
    storage.put_artifact(catalog_id, version, store.path_for(catalog_id))
    return version


# ------------------------------------------------------------------ keys --
class TestKeyDerivation:
    def test_key_is_catalog_id_and_index_content_hash(self, built):
        version = read_artifact_version(built.path_for("acme"))
        assert artifact_key("acme", version) == f"catalogs/acme/{version}"

    def test_version_is_the_artifacts_own_index_content_hash(self, built):
        descriptor = json.loads((built.path_for("acme") / "artifact.json").read_text())
        assert read_artifact_version(built.path_for("acme")) == \
            descriptor["index_content_hash"]

    @pytest.mark.parametrize("bad", ["", "abc", "z" * 64, "0" * 63, "0" * 65])
    def test_a_non_hash_is_not_a_version(self, bad):
        with pytest.raises(Exception):
            validate_version(bad)

    def test_uppercase_hex_normalises_rather_than_forking_the_key(self):
        """Two spellings of one hash must not become two artifacts."""
        assert validate_version("A" * 64) == "a" * 64
        assert artifact_key("acme", "B" * 64) == artifact_key("acme", "b" * 64)

    def test_a_catalog_id_cannot_traverse_out_of_its_prefix(self):
        with pytest.raises(CatalogStoreError):
            artifact_key("../escape", "a" * 64)


# ------------------------------------------------------------- round trip --
class TestRoundTrip:
    def test_publish_then_fetch_reproduces_every_member(self, built, storage, tmp_path):
        version = _publish(storage, built, "acme")
        dest = tmp_path / "fetched" / "acme"
        manifest = storage.get_artifact("acme", version, dest)
        assert set(manifest.members) == set(ARTIFACT_MEMBERS)
        for rel in ARTIFACT_MEMBERS:
            assert (dest / rel).read_bytes() == (built.path_for("acme") / rel).read_bytes()

    def test_a_fetched_artifact_loads_and_verifies(self, built, storage, tmp_path):
        version = _publish(storage, built, "acme")
        root = tmp_path / "root"
        storage.get_artifact("acme", version, root / "acme")
        loaded = CatalogStore(root).load("acme", verify=True)
        assert loaded.artifact["index_content_hash"] == version
        assert loaded.track_count == 2

    def test_fetching_an_absent_version_raises(self, storage, tmp_path):
        with pytest.raises(ArtifactNotFound):
            storage.get_artifact("acme", "c" * 64, tmp_path / "x")

    def test_only_file_urls_are_implemented(self, tmp_path):
        assert isinstance(storage_from_url(f"file://{tmp_path}"), LocalArtifactStorage)
        with pytest.raises(NotImplementedError, match="No object-storage provider"):
            storage_from_url("s3://bucket/prefix")


# ----------------------------------------------------------- immutability --
class TestImmutability:
    def test_republishing_identical_content_is_a_no_op(self, built, storage):
        version = _publish(storage, built, "acme")
        before = sorted(p.name for p in (storage.root / "catalogs" / "acme").iterdir())
        key = storage.put_artifact("acme", version, built.path_for("acme"))
        after = sorted(p.name for p in (storage.root / "catalogs" / "acme").iterdir())
        assert key == artifact_key("acme", version)
        assert before == after == [version]

    def test_same_key_with_different_bytes_is_refused(self, built, storage, tmp_path):
        """A metadata-only edit changes neither content hash but changes bytes.

        `index_content_hash` covers TrackEntry(track_id, fingerprint_count,
        duration_sec); `catalog_content_hash` covers (track_id, sha256). Editing
        a title therefore collides on the key -- and must be refused, not
        silently overwritten, or every previously fetched copy of that version
        becomes unreproducible.
        """
        version = _publish(storage, built, "acme")
        edited = tmp_path / "edited"
        shutil.copytree(built.path_for("acme"), edited)
        catalog = json.loads((edited / "catalog.json").read_text())
        catalog["tracks"][0]["title"] = "A Different Title"
        (edited / "catalog.json").write_text(json.dumps(catalog, indent=2))

        with pytest.raises(ArtifactConflict, match="already holds different content"):
            storage.put_artifact("acme", version, edited)
        # The published bytes are untouched.
        assert (storage.root / artifact_key("acme", version) / "catalog.json"
                ).read_bytes() == (built.path_for("acme") / "catalog.json").read_bytes()

    def test_different_content_hashes_are_independent_versions(self, built, storage,
                                                               tmp_path):
        real = _publish(storage, built, "acme")
        # A second version under the same catalog, distinct content hash.
        other = tmp_path / "v2"
        shutil.copytree(built.path_for("acme"), other)
        fake = "a1" * 32
        descriptor = json.loads((other / "artifact.json").read_text())
        descriptor["index_content_hash"] = fake
        (other / "artifact.json").write_text(json.dumps(descriptor, indent=2))
        storage.put_artifact("acme", fake, other)

        assert sorted(storage.list_versions("acme")) == sorted([real, fake])
        assert storage.exists("acme", real) and storage.exists("acme", fake)

    def test_there_is_no_mutable_latest_pointer(self, built, storage):
        _publish(storage, built, "acme")
        names = {p.name for p in (storage.root / "catalogs" / "acme").iterdir()}
        assert not {"latest", "current", "HEAD"} & names
        assert all(len(n) == 64 for n in names)


# ---------------------------------------------------------------- listing --
class TestListing:
    def test_versions_and_catalogs_are_listed(self, built, storage):
        _publish(storage, built, "acme")
        _publish(storage, built, "globex")
        assert storage.list_catalogs() == ["acme", "globex"]
        assert len(storage.list_versions("acme")) == 1

    def test_listing_an_unknown_catalog_is_empty_not_an_error(self, storage):
        assert storage.list_versions("nothing") == []
        assert storage.list_catalogs() == []


# -------------------------------------------------------------- isolation --
class TestCatalogIsolation:
    def test_catalogs_occupy_disjoint_prefixes(self, built, storage):
        va = _publish(storage, built, "acme")
        vg = _publish(storage, built, "globex")
        assert va != vg
        assert artifact_key("acme", va).startswith("catalogs/acme/")
        assert artifact_key("globex", vg).startswith("catalogs/globex/")
        assert storage.exists("acme", va) and not storage.exists("globex", va)

    def test_syncing_one_catalog_does_not_materialise_another(self, built, storage,
                                                              tmp_path):
        va = _publish(storage, built, "acme")
        _publish(storage, built, "globex")
        root = tmp_path / "root"
        sync_catalog(storage, root, "acme", va)
        assert (root / "acme").is_dir()
        assert not (root / "globex").exists()


# ------------------------------------------------------------- corruption --
class TestCorruptionAndPartialDownload:
    def test_a_corrupted_stored_member_is_rejected_by_the_manifest(self, built,
                                                                   storage, tmp_path):
        version = _publish(storage, built, "acme")
        stored = storage.root / artifact_key("acme", version) / "index" / "hashes.npy"
        data = bytearray(stored.read_bytes())
        data[-1] ^= 0xFF                       # one flipped bit
        stored.write_bytes(bytes(data))
        with pytest.raises(ArtifactIncomplete, match="does not match its manifest"):
            storage.get_artifact("acme", version, tmp_path / "dest")

    def test_a_missing_stored_member_is_rejected(self, built, storage, tmp_path):
        version = _publish(storage, built, "acme")
        (storage.root / artifact_key("acme", version) / "index" / "track_ords.npy").unlink()
        with pytest.raises(ArtifactIncomplete, match="missing"):
            storage.get_artifact("acme", version, tmp_path / "dest")

    def test_a_corrupt_fetch_never_becomes_the_active_catalog(self, built, storage,
                                                              tmp_path):
        """The existing catalog must survive a failed sync untouched."""
        version = _publish(storage, built, "acme")
        root = tmp_path / "root"
        sync_catalog(storage, root, "acme", version)
        good = (root / "acme" / "index" / "hashes.npy").read_bytes()

        stored = storage.root / artifact_key("acme", version) / "catalog.json"
        stored.write_text("{ not json")
        with pytest.raises(SyncError):
            sync_catalog(storage, root, "acme", "b1" * 32)   # different version
        assert (root / "acme" / "index" / "hashes.npy").read_bytes() == good
        assert CatalogStore(root).load("acme", verify=True).track_count == 2

    def test_no_staging_directory_survives_a_failure(self, built, storage, tmp_path):
        version = _publish(storage, built, "acme")
        (storage.root / artifact_key("acme", version) / "index" / "hashes.npy").unlink()
        root = tmp_path / "root"
        with pytest.raises(SyncError):
            sync_catalog(storage, root, "acme", version)
        leftovers = list((root / ".sync").glob("*")) if (root / ".sync").is_dir() else []
        assert leftovers == [], f"staging left behind: {leftovers}"
        assert not (root / "acme").exists()

    def test_an_artifact_whose_declared_version_disagrees_is_refused(self, built,
                                                                     storage, tmp_path):
        """Storing bytes under the wrong key must not produce a served catalog."""
        version = read_artifact_version(built.path_for("acme"))
        wrong = "c3" * 32
        storage.put_artifact("acme", wrong, built.path_for("acme"))
        with pytest.raises(SyncError, match="declares version"):
            sync_catalog(storage, tmp_path / "root", "acme", wrong)


# ------------------------------------------------------- version resolution --
class TestVersionResolution:
    def test_a_pin_wins(self, built, storage):
        version = _publish(storage, built, "acme")
        assert resolve_version("acme", storage, pins={"acme": version}) == version

    def test_the_database_version_is_used_when_there_is_no_pin(self, built, storage):
        version = _publish(storage, built, "acme")
        assert resolve_version("acme", storage, pins={},
                               db_versions={"acme": version}) == version

    def test_a_sole_version_needs_no_pin(self, built, storage):
        version = _publish(storage, built, "acme")
        assert resolve_version("acme", storage, pins={}) == version

    def test_ambiguity_is_an_error_not_a_guess(self, built, storage, tmp_path):
        _publish(storage, built, "acme")
        other = tmp_path / "v2"
        shutil.copytree(built.path_for("acme"), other)
        storage.put_artifact("acme", "d4" * 32, other)
        with pytest.raises(SyncError, match="no pin or database row"):
            resolve_version("acme", storage, pins={})

    def test_no_versions_at_all_is_an_error(self, storage):
        with pytest.raises(SyncError, match="no artifact versions"):
            resolve_version("acme", storage, pins={})

    def test_pins_parse(self):
        v = "e5" * 32
        assert parse_pins(f"acme={v}") == {"acme": v}
        assert parse_pins("") == {}
        with pytest.raises(SyncError):
            parse_pins("acme")


# ------------------------------------------------- the service, end to end --
class _CountingStorage:
    """Wraps a backend and counts fetches, to prove none happen per request."""

    def __init__(self, inner):
        self.inner = inner
        self.get_calls = 0
        self.list_calls = 0

    def put_artifact(self, *a, **k):
        return self.inner.put_artifact(*a, **k)

    def get_artifact(self, *a, **k):
        self.get_calls += 1
        return self.inner.get_artifact(*a, **k)

    def exists(self, *a, **k):
        return self.inner.exists(*a, **k)

    def list_versions(self, *a, **k):
        self.list_calls += 1
        return self.inner.list_versions(*a, **k)

    def list_catalogs(self, *a, **k):
        self.list_calls += 1
        return self.inner.list_catalogs(*a, **k)

    def describe(self):
        return self.inner.describe()


class TestBootSynchronisation:
    KEY = "sk_test_storage_00000000000000"

    @staticmethod
    def _keys_file(tmp_path):
        from musicintel.api.auth import hash_key
        p = tmp_path / "keys.json"
        p.write_text(json.dumps([{
            "key_id": "k_s", "tenant": "acme",
            "key_sha256": hash_key(TestBootSynchronisation.KEY),
            "catalogs": [], "rate_limit_per_minute": 600,
            "rate_limit_burst": 200, "audio_seconds_per_day": 100000,
        }]))
        return p

    def _settings(self, tmp_path, **over):
        from musicintel.api.config import Settings
        base = dict(catalog_root=tmp_path / "root",
                    api_keys_file=self._keys_file(tmp_path),
                    environment="test", log_json=False)
        base.update(over)
        return Settings(**base)

    @staticmethod
    def _client(settings, spy=None, monkeypatch=None):
        import fakeredis.aioredis
        from fastapi.testclient import TestClient

        from musicintel.api.app import create_app

        if spy is not None:
            monkeypatch.setattr("musicintel.api.app.storage_from_url",
                                lambda url: spy)
        app = create_app(settings)
        app.state.redis = fakeredis.aioredis.FakeRedis(decode_responses=False)
        return app, TestClient(app)

    def test_artifacts_are_present_before_the_first_request(self, built, storage,
                                                            tmp_path):
        """Nothing is served until the artifact is local and verified."""
        version = _publish(storage, built, "acme")
        settings = self._settings(
            tmp_path, artifact_storage_url=f"file://{storage.root}")
        assert not (tmp_path / "root" / "acme").exists()

        app, client = self._client(settings)
        with client:
            # The artifact existed on disk before any request was handled.
            assert (tmp_path / "root" / "acme" / "artifact.json").is_file()
            assert [r.action for r in app.state.artifact_sync] == ["fetched"]
            r = client.get("/v1/catalogs",
                           headers={"Authorization": f"Bearer {self.KEY}"})
            assert r.status_code == 200
            assert [c["catalog_id"] for c in r.json()["catalogs"]] == ["acme"]
        assert read_artifact_version(tmp_path / "root" / "acme") == version

    def test_no_object_storage_fetch_happens_during_a_request(self, built, storage,
                                                              tmp_path, monkeypatch):
        _publish(storage, built, "acme")
        spy = _CountingStorage(storage)
        settings = self._settings(
            tmp_path, artifact_storage_url=f"file://{storage.root}")
        app, client = self._client(settings, spy=spy, monkeypatch=monkeypatch)
        with client:
            after_boot = spy.get_calls
            assert after_boot == 1, "boot should fetch exactly once"
            headers = {"Authorization": f"Bearer {self.KEY}"}
            for _ in range(5):
                assert client.get("/v1/catalogs", headers=headers).status_code == 200
                assert client.get("/v1/catalogs/acme", headers=headers).status_code == 200
            assert spy.get_calls == after_boot, "a request fetched from storage"

    def test_a_second_boot_does_not_refetch(self, built, storage, tmp_path,
                                            monkeypatch):
        """Re-downloading 175 MB every boot would be the costliest no-op here."""
        _publish(storage, built, "acme")
        settings = self._settings(
            tmp_path, artifact_storage_url=f"file://{storage.root}")
        _app, client = self._client(settings)
        with client:
            pass
        spy = _CountingStorage(storage)
        app2, client2 = self._client(settings, spy=spy, monkeypatch=monkeypatch)
        with client2:
            assert [r.action for r in app2.state.artifact_sync] == ["already-current"]
            assert spy.get_calls == 0

    def test_synchronisation_failure_prevents_startup(self, built, storage, tmp_path):
        """An instance that cannot verify its catalog must not serve queries."""
        version = _publish(storage, built, "acme")
        (storage.root / artifact_key("acme", version) / "index" / "hashes.npy").unlink()
        settings = self._settings(
            tmp_path, artifact_storage_url=f"file://{storage.root}")
        _app, client = self._client(settings)
        with pytest.raises(SyncError):
            with client:
                pass
        assert not (tmp_path / "root" / "acme").exists()

    def test_a_missing_pinned_catalog_prevents_startup(self, storage, tmp_path):
        settings = self._settings(
            tmp_path, artifact_storage_url=f"file://{storage.root}",
            artifact_pins=f"acme={'f' * 64}")
        _app, client = self._client(settings)
        with pytest.raises(SyncError):
            with client:
                pass

    def test_local_only_mode_is_unchanged(self, built, tmp_path):
        """No storage URL -> the Stage 3 local-volume path, byte for byte."""
        root = tmp_path / "root"
        shutil.copytree(built.path_for("acme"), root / "acme")
        settings = self._settings(tmp_path)
        assert settings.artifact_storage_enabled is False

        app, client = self._client(settings)
        with client:
            assert app.state.artifact_sync is None
            r = client.get("/v1/catalogs",
                           headers={"Authorization": f"Bearer {self.KEY}"})
            assert r.status_code == 200
            assert [c["catalog_id"] for c in r.json()["catalogs"]] == ["acme"]

    def test_sync_only_the_named_catalogs(self, built, storage, tmp_path):
        _publish(storage, built, "acme")
        _publish(storage, built, "globex")
        settings = self._settings(
            tmp_path, artifact_storage_url=f"file://{storage.root}",
            sync_catalogs="acme")
        app, client = self._client(settings)
        with client:
            assert [r.catalog_id for r in app.state.artifact_sync] == ["acme"]
            assert (tmp_path / "root" / "acme").is_dir()
            assert not (tmp_path / "root" / "globex").exists()
