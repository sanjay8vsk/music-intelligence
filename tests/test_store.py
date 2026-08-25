"""Tests for multi-tenant catalog storage.

Isolation here is structural -- each catalog owns its index artifact -- so these
test that the structure holds: that a catalog cannot be confused with another,
that an artifact cannot drift from the catalog that built it, and that a
catalog id cannot escape the store root.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import soundfile as sf

from musicintel.catalog.ingest import build_catalog_index, ingest_directory
from musicintel.catalog.models import Catalog
from musicintel.catalog.store import (
    ARTIFACT_FILENAME,
    ARTIFACT_VERSION,
    CATALOG_FILENAME,
    INDEX_DIRNAME,
    CatalogStore,
    CatalogStoreError,
    validate_catalog_id,
)

SR = 11025


def _write(path, seconds=6.0, seed=0):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, seconds, int(SR * seconds), endpoint=False)
    wob = 600.0 + 200.0 * np.sin(2 * np.pi * 0.5 * t + seed)
    y = (0.50 * np.sin(2 * np.pi * (440.0 + 7 * seed) * t)
         + 0.30 * np.sin(2 * np.pi * wob * t)
         + 0.20 * np.sin(2 * np.pi * 1500.0 * t)
         + 0.02 * rng.standard_normal(t.size)).astype(np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, y, SR, subtype="PCM_16")


def _ingest(tmp_path, name, seeds):
    d = tmp_path / "audio" / name
    for s in seeds:
        _write(d / f"{name}_{s}.wav", seed=s)
    r = ingest_directory(d)
    return r.catalog, build_catalog_index(r.catalog, r.fingerprints)


@pytest.fixture
def two_tenants(tmp_path):
    store = CatalogStore(tmp_path / "store")
    for cid, seeds in (("acme", [1, 2, 3]), ("globex", [11, 12, 13])):
        cat, idx = _ingest(tmp_path, cid, seeds)
        store.save(cat, idx, catalog_id=cid)
    return store


# ----------------------------------------------------------- identifiers ----
class TestCatalogIdValidation:
    @pytest.mark.parametrize("cid", ["acme", "a", "A-1", "cat.v2", "x_y-1.2"])
    def test_accepts_sane_ids(self, cid):
        assert validate_catalog_id(cid) == cid

    @pytest.mark.parametrize("cid", ["../etc", "a/b", "a\\b", "", ".", "..",
                                     "-lead", ".lead", "x" * 65, "a b"])
    def test_rejects_anything_that_could_escape_or_collide(self, cid):
        with pytest.raises(CatalogStoreError):
            validate_catalog_id(cid)

    def test_store_refuses_a_traversing_id(self, tmp_path):
        with pytest.raises(CatalogStoreError):
            CatalogStore(tmp_path).load("../etc")


# ------------------------------------------------------------- roundtrip ----
class TestSaveLoad:
    def test_roundtrip_preserves_catalog_and_index(self, two_tenants):
        loaded = two_tenants.load("acme")
        assert loaded.catalog_id == "acme"
        assert loaded.catalog.catalog_id == "acme"
        assert loaded.track_count == 3
        assert loaded.fingerprint_count == len(loaded.index)
        assert set(loaded.catalog.track_ids) == set(loaded.index.track_ids)

    def test_layout_is_one_directory_per_catalog(self, two_tenants):
        for cid in ("acme", "globex"):
            d = two_tenants.path_for(cid)
            assert (d / CATALOG_FILENAME).is_file()
            assert (d / INDEX_DIRNAME).is_dir()
            assert (d / ARTIFACT_FILENAME).is_file()

    def test_list_and_exists(self, two_tenants):
        assert two_tenants.list_catalogs() == ["acme", "globex"]
        assert two_tenants.exists("acme") and not two_tenants.exists("nope")

    def test_describe_does_not_need_the_index(self, two_tenants):
        a = two_tenants.describe("acme")
        assert a["catalog_id"] == "acme" and a["track_count"] == 3
        assert a["artifact_version"] == ARTIFACT_VERSION

    def test_unknown_catalog_raises(self, two_tenants):
        with pytest.raises(CatalogStoreError, match="no catalog"):
            two_tenants.load("missing")

    def test_empty_store_lists_nothing(self, tmp_path):
        assert CatalogStore(tmp_path / "void").list_catalogs() == []


# ------------------------------------------------------------- isolation ----
class TestIsolation:
    def test_a_loaded_catalog_contains_only_its_own_tracks(self, two_tenants):
        """Structural isolation: the other tenant's postings are not in the
        array being searched, so there is no filter that could be forgotten."""
        a, b = two_tenants.load("acme"), two_tenants.load("globex")
        assert set(a.index.track_ids).isdisjoint(set(b.index.track_ids))
        assert set(a.catalog.track_ids).isdisjoint(set(b.catalog.track_ids))

    def test_catalogs_have_independent_content_hashes(self, two_tenants):
        a, b = two_tenants.load("acme"), two_tenants.load("globex")
        assert a.catalog.content_hash() != b.catalog.content_hash()
        assert a.index.content_hash() != b.index.content_hash()

    def test_saving_one_catalog_does_not_disturb_another(self, two_tenants, tmp_path):
        before = two_tenants.describe("globex")
        cat, idx = _ingest(tmp_path, "acme2", [21, 22])
        two_tenants.save(cat, idx, catalog_id="acme")
        assert two_tenants.describe("globex") == before


# ------------------------------------------------------------- integrity ----
class TestArtifactIntegrity:
    def test_artifact_binds_catalog_to_index(self, two_tenants):
        a = two_tenants.describe("acme")
        loaded = two_tenants.load("acme")
        assert a["catalog_content_hash"] == loaded.catalog.content_hash()
        assert a["index_content_hash"] == loaded.index.content_hash()

    def test_reproducible_from_the_manifest(self, tmp_path):
        """Rebuilding from the same audio must give the same index hash."""
        store = CatalogStore(tmp_path / "s")
        cat, idx = _ingest(tmp_path, "rep", [1, 2])
        store.save(cat, idx, catalog_id="rep")
        first = store.describe("rep")["index_content_hash"]
        cat2, idx2 = _ingest(tmp_path, "rep", [1, 2])   # same audio, re-ingested
        assert idx2.content_hash() == first

    def test_timestamp_is_omitted_by_default(self, two_tenants):
        assert two_tenants.describe("acme")["built_utc"] is None

    def test_tampering_with_the_catalog_is_detected(self, two_tenants):
        d = two_tenants.path_for("acme")
        cat = Catalog.load(d / CATALOG_FILENAME)
        cat.tracks = cat.tracks[:-1]
        cat.save(d / CATALOG_FILENAME)
        with pytest.raises(CatalogStoreError, match="content hash"):
            two_tenants.load("acme")

    def test_a_renamed_directory_is_detected(self, two_tenants, tmp_path):
        import shutil
        src, dst = two_tenants.path_for("acme"), two_tenants.path_for("copied")
        shutil.copytree(src, dst)
        with pytest.raises(CatalogStoreError, match="renamed or copied"):
            two_tenants.load("copied")

    def test_an_unknown_artifact_version_is_refused(self, two_tenants):
        p = two_tenants.path_for("acme") / ARTIFACT_FILENAME
        a = json.loads(p.read_text()); a["artifact_version"] = 999
        p.write_text(json.dumps(a))
        with pytest.raises(CatalogStoreError, match="artifact version"):
            two_tenants.load("acme")

    def test_malformed_artifact_json_is_refused(self, two_tenants):
        (two_tenants.path_for("acme") / ARTIFACT_FILENAME).write_text("{nope")
        with pytest.raises(CatalogStoreError, match="not valid JSON"):
            two_tenants.load("acme")

    def test_saving_a_mismatched_catalog_and_index_is_refused(self, tmp_path):
        """An index built from different audio than the catalog describes would
        let a query return a track the catalog cannot explain."""
        store = CatalogStore(tmp_path / "s")
        cat_a, idx_a = _ingest(tmp_path, "a", [1, 2])
        cat_b, _ = _ingest(tmp_path, "b", [7, 8])
        with pytest.raises(CatalogStoreError, match="disagree"):
            store.save(cat_b, idx_a, catalog_id="mixed")

    def test_verify_can_be_skipped_explicitly(self, two_tenants):
        d = two_tenants.path_for("acme")
        cat = Catalog.load(d / CATALOG_FILENAME)
        cat.tracks = cat.tracks[:-1]
        cat.save(d / CATALOG_FILENAME)
        loaded = two_tenants.load("acme", verify=False)   # opt out, deliberately
        assert loaded.track_count == 2


# ------------------------------------------------- backward compatibility ---
class TestBackwardCompatibility:
    def test_a_catalog_written_before_tenancy_still_loads(self, tmp_path):
        """Stage 2's catalog.json had no catalog_id; it must not become unreadable."""
        cat, _ = _ingest(tmp_path, "old", [1])
        p = tmp_path / "old.json"
        cat.save(p)
        payload = json.loads(p.read_text())
        del payload["catalog_id"]
        p.write_text(json.dumps(payload))
        back = Catalog.load(p)
        assert back.catalog_id == "default"
        assert back.content_hash() == cat.content_hash()   # identity unaffected

    def test_catalog_id_does_not_affect_the_content_hash(self, tmp_path):
        cat, _ = _ingest(tmp_path, "h", [1, 2])
        before = cat.content_hash()
        cat.catalog_id = "renamed"
        assert cat.content_hash() == before
