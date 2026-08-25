"""Tests for catalog identity, ingestion and index construction.

The prototype these replace derived track identity with `file.split(".")[0]`.
Several tests below exist specifically so that cannot come back: it is a silent
data-loss bug, not a style problem, and a catalog that quietly merges two
recordings cannot be trusted about which one matched.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from musicintel.catalog.ingest import (
    AUDIO_EXTENSIONS,
    IngestError,
    build_catalog_index,
    config_digest,
    derive_track_id,
    discover_audio,
    ingest_directory,
    ingest_paths,
)
from musicintel.catalog.models import Catalog, CatalogTrack, sha256_file
from musicintel.recognition.fingerprint import FingerprintConfig

SR = 11025


def _write(path: Path, seconds=4.0, seed=0):
    """A short, deterministic, fingerprintable clip."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, seconds, int(SR * seconds), endpoint=False)
    wob = 600.0 + 200.0 * np.sin(2 * np.pi * 0.5 * t + seed)
    y = (0.50 * np.sin(2 * np.pi * (440.0 + 7 * seed) * t)
         + 0.30 * np.sin(2 * np.pi * wob * t)
         + 0.20 * np.sin(2 * np.pi * 1500.0 * t)
         + 0.02 * rng.standard_normal(t.size)).astype(np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, y, SR, subtype="PCM_16")
    return path


@pytest.fixture
def library(tmp_path):
    d = tmp_path / "audio"
    _write(d / "plain.wav", seed=1)
    _write(d / "mix.1.wav", seed=2)          # the prototype collapsed these
    _write(d / "mix.2.wav", seed=3)          # two onto a single id
    _write(d / "sub" / "Song feat. Artist.wav", seed=4)
    (d / "cover.jpg").write_bytes(b"not audio")
    (d / "notes.txt").write_text("not audio")
    (d / ".hidden.wav").write_bytes(b"not really audio either")
    return d


# --------------------------------------------------------- identity ---------
class TestTrackIdentity:
    def test_stem_mode_keeps_interior_dots(self):
        """The regression that matters: `split(".")[0]` truncates these."""
        cases = {"track.mp3": "track", "mix.1.wav": "mix.1", "mix.2.wav": "mix.2",
                 "Song feat. Artist.mp3": "Song feat. Artist", "a.b.flac": "a.b"}
        for name, want in cases.items():
            assert derive_track_id(name) == want
            if "." in Path(name).stem:
                assert name.split(".")[0] != want   # the old behaviour differed

    def test_two_dotted_siblings_get_distinct_ids(self):
        a, b = derive_track_id("mix.1.wav"), derive_track_id("mix.2.wav")
        assert a != b
        assert "mix.1.wav".split(".")[0] == "mix.2.wav".split(".")[0]  # old bug

    def test_content_mode_is_name_independent(self):
        sha = "a" * 64
        assert derive_track_id("one.mp3", mode="content", sha256=sha) == \
               derive_track_id("other.flac", mode="content", sha256=sha)

    def test_content_mode_needs_a_hash(self):
        with pytest.raises(IngestError):
            derive_track_id("x.mp3", mode="content")

    def test_unknown_mode_is_rejected(self):
        with pytest.raises(IngestError, match="unknown id mode"):
            derive_track_id("x.mp3", mode="magic")


# --------------------------------------------------------- discovery --------
class TestDiscovery:
    def test_finds_audio_recursively_and_ignores_the_rest(self, library):
        found = discover_audio(library)
        names = [p.name for p in found]
        # Sorted by full PATH, not by basename -- that is what makes two scans
        # of one directory produce the same track ordinals.
        assert found == sorted(found)
        assert "Song feat. Artist.wav" in names             # recursed into sub/
        assert "cover.jpg" not in names and "notes.txt" not in names
        assert ".hidden.wav" not in names
        assert len(names) == 4

    def test_extension_filter_is_case_insensitive(self, tmp_path):
        _write(tmp_path / "UPPER.WAV")
        assert len(discover_audio(tmp_path)) == 1

    def test_missing_directory_raises(self, tmp_path):
        with pytest.raises(IngestError, match="not a directory"):
            discover_audio(tmp_path / "nope")

    def test_extension_list_is_explicit(self):
        assert ".wav" in AUDIO_EXTENSIONS and ".mp3" in AUDIO_EXTENSIONS
        assert ".jpg" not in AUDIO_EXTENSIONS


# --------------------------------------------------------- ingestion --------
class TestIngestion:
    def test_ingests_a_library(self, library):
        r = ingest_directory(library)
        assert r.ingested == 4 and r.skipped == []
        assert set(r.catalog.track_ids) == {"plain", "mix.1", "mix.2", "Song feat. Artist"}
        assert all(t.fingerprint_count > 0 for t in r.catalog)
        assert all(len(t.sha256) == 64 for t in r.catalog)

    def test_source_path_is_provenance_not_identity(self, library):
        r = ingest_directory(library)
        t = r.catalog.by_id("Song feat. Artist")
        assert t.source_path.endswith("Song feat. Artist.wav")
        assert "sub" in t.source_path        # recorded, but the id has no path in it
        assert t.track_id == "Song feat. Artist"

    def test_duplicate_ids_raise_rather_than_overwrite(self, tmp_path):
        """The prototype's silent overwrite is the bug; refusing is the fix."""
        _write(tmp_path / "a.wav", seed=1)
        _write(tmp_path / "sub" / "a.wav", seed=2)
        with pytest.raises(IngestError, match="duplicate track_id"):
            ingest_directory(tmp_path)

    def test_duplicate_ids_can_be_skipped_explicitly(self, tmp_path):
        _write(tmp_path / "a.wav", seed=1)
        _write(tmp_path / "sub" / "a.wav", seed=2)
        r = ingest_directory(tmp_path, on_duplicate_id="skip")
        assert r.ingested == 1 and len(r.skipped) == 1
        assert "duplicate track_id" in r.skipped[0]["reason"]

    def test_an_undecodable_file_is_skipped_not_fatal(self, tmp_path):
        _write(tmp_path / "good.wav", seed=1)
        (tmp_path / "broken.wav").write_bytes(b"RIFF____not-a-wav")
        r = ingest_directory(tmp_path)
        assert r.ingested == 1 and len(r.skipped) == 1
        assert r.catalog.track_ids == ("good",)

    def test_is_deterministic(self, library):
        a, b = ingest_directory(library), ingest_directory(library)
        assert a.catalog.content_hash() == b.catalog.content_hash()
        assert a.catalog.track_ids == b.catalog.track_ids


# --------------------------------------------------------- cache ------------
class TestFingerprintCache:
    def test_second_ingest_hits_the_cache(self, library, tmp_path):
        cache = tmp_path / "cache"
        first = ingest_directory(library, cache_dir=cache)
        second = ingest_directory(library, cache_dir=cache)
        assert first.cache_hits == 0
        assert second.cache_hits == second.ingested == 4
        assert second.catalog.content_hash() == first.catalog.content_hash()

    def test_cached_fingerprints_are_identical(self, library, tmp_path):
        cache = tmp_path / "cache"
        a = ingest_directory(library, cache_dir=cache)
        b = ingest_directory(library, cache_dir=cache)
        for tid in a.fingerprints:
            assert np.array_equal(a.fingerprints[tid].hashes, b.fingerprints[tid].hashes)
            assert np.array_equal(a.fingerprints[tid].anchor_frames,
                                  b.fingerprints[tid].anchor_frames)

    def test_a_config_change_invalidates_the_cache(self, library, tmp_path):
        """Keying on audio alone would serve fingerprints made under other settings."""
        cache = tmp_path / "cache"
        ingest_directory(library, cache_dir=cache)
        other = FingerprintConfig(fan_out=3)
        r = ingest_directory(library, cache_dir=cache, config=other)
        assert r.cache_hits == 0

    def test_config_digest_tracks_the_config(self):
        assert config_digest(FingerprintConfig()) == config_digest(FingerprintConfig())
        assert config_digest(FingerprintConfig()) != config_digest(FingerprintConfig(fan_out=3))

    def test_a_corrupt_cache_entry_is_a_miss_not_a_crash(self, library, tmp_path):
        cache = tmp_path / "cache"
        ingest_directory(library, cache_dir=cache)
        for f in cache.glob("*.npz"):
            f.write_bytes(b"garbage")
        r = ingest_directory(library, cache_dir=cache)
        assert r.ingested == 4 and r.cache_hits == 0


# --------------------------------------------------------- the catalog ------
class TestCatalogModel:
    def test_content_hash_is_stable_and_path_independent(self, library, tmp_path):
        r = ingest_directory(library)
        moved = Catalog(tracks=[CatalogTrack(**{**t.to_dict(),
                                                "source_path": "elsewhere/" + t.source_path})
                                for t in r.catalog])
        assert moved.content_hash() == r.catalog.content_hash()

    def test_content_hash_changes_with_membership(self, library):
        r = ingest_directory(library)
        fewer = Catalog(tracks=r.catalog.tracks[:-1])
        assert fewer.content_hash() != r.catalog.content_hash()

    def test_save_load_roundtrip(self, library, tmp_path):
        r = ingest_directory(library)
        r.catalog.save(tmp_path / "catalog.json")
        back = Catalog.load(tmp_path / "catalog.json")
        assert back.content_hash() == r.catalog.content_hash()
        assert back.track_ids == tuple(sorted(r.catalog.track_ids))

    def test_saved_catalog_holds_no_audio(self, library, tmp_path):
        r = ingest_directory(library)
        p = r.catalog.save(tmp_path / "catalog.json")
        blob = p.read_text()
        assert "hashes" not in blob and "anchor_frames" not in blob
        assert json.loads(blob)["track_count"] == 4

    def test_verify_reports_a_missing_file(self, library, tmp_path):
        r = ingest_directory(library)
        (library / "plain.wav").unlink()
        assert any("missing audio" in p for p in r.catalog.verify(library))

    def test_verify_detects_changed_content(self, library):
        r = ingest_directory(library)
        _write(library / "plain.wav", seed=99)       # same name, different audio
        problems = r.catalog.verify(library, check_hashes=True)
        assert any("content changed" in p for p in problems)
        assert r.catalog.verify(library) == []       # cheap mode cannot see it

    def test_verify_catches_duplicate_ids(self):
        t = CatalogTrack("x", "a.wav", "a" * 64, 1.0)
        assert any("duplicate track_id" in p
                   for p in Catalog(tracks=[t, t]).verify(check_hashes=False))

    def test_duplicate_content_is_reported_not_rejected(self, tmp_path):
        """One recording under two ids is legal but always worth surfacing."""
        _write(tmp_path / "one.wav", seed=7)
        import shutil
        shutil.copy(tmp_path / "one.wav", tmp_path / "two.wav")
        r = ingest_directory(tmp_path)
        assert r.ingested == 2
        dupes = r.catalog.duplicate_content()
        assert len(dupes) == 1 and sorted(next(iter(dupes.values()))) == ["one", "two"]

    def test_malformed_track_is_rejected_on_load(self, tmp_path):
        (tmp_path / "c.json").write_text(json.dumps({"tracks": [{"track_id": "x"}]}))
        with pytest.raises(ValueError, match="missing fields"):
            Catalog.load(tmp_path / "c.json")

    def test_sha256_file_matches_hashlib(self, tmp_path):
        import hashlib
        p = tmp_path / "f.bin"
        p.write_bytes(b"hello world")
        assert sha256_file(p) == hashlib.sha256(b"hello world").hexdigest()


# --------------------------------------------------------- index ------------
class TestIndexConstruction:
    def test_builds_an_index_over_the_catalog(self, library):
        r = ingest_directory(library)
        idx = build_catalog_index(r.catalog, r.fingerprints)
        assert idx.n_tracks == 4
        assert set(idx.track_ids) == set(r.catalog.track_ids)
        assert len(idx) == r.catalog.total_fingerprints

    def test_index_is_ordered_by_track_id_and_reproducible(self, library):
        r = ingest_directory(library)
        a = build_catalog_index(r.catalog, r.fingerprints)
        shuffled = Catalog(tracks=list(reversed(r.catalog.tracks)))
        b = build_catalog_index(shuffled, r.fingerprints)
        assert a.track_ids == tuple(sorted(r.catalog.track_ids))
        assert a.content_hash() == b.content_hash()

    def test_missing_fingerprints_raise(self, library):
        r = ingest_directory(library)
        r.fingerprints.pop("plain")
        with pytest.raises(IngestError, match="no fingerprints"):
            build_catalog_index(r.catalog, r.fingerprints)

    def test_the_index_recognizes_its_own_catalog(self, library):
        from musicintel.recognition.decision import DecisionConfig, decide
        from musicintel.recognition.fingerprint import fingerprint, load_audio
        from musicintel.recognition.matcher import match
        r = ingest_directory(library)
        idx = build_catalog_index(r.catalog, r.fingerprints)
        y, sr = load_audio(library / "mix.2.wav", FingerprintConfig())
        d = decide(match(fingerprint(y, sr, FingerprintConfig()), idx),
                   config=DecisionConfig(threshold=0.05, min_aligned_landmarks=5))
        assert d.is_match and d.track_id == "mix.2"

    def test_empty_catalog_yields_an_empty_index(self):
        idx = build_catalog_index(Catalog(), {})
        assert idx.n_tracks == 0 and len(idx) == 0
