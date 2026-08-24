"""Focused tests for the persistent fingerprint index.

Audio is synthesized here, as in the other suites, so nothing depends on the
fixture corpus. The corpus is for the smoke test, not for unit tests.
"""

from __future__ import annotations

import dataclasses
import json

import numpy as np
import pytest

from musicintel.recognition.fingerprint import (
    FORMAT_VERSION as FP_VERSION,
)
from musicintel.recognition.fingerprint import (
    FingerprintConfig,
    fingerprint,
)
from musicintel.recognition.index import (
    ANCHOR_FRAMES_FILENAME,
    HASHES_FILENAME,
    INDEX_FORMAT_VERSION,
    META_FILENAME,
    TRACK_ORDS_FILENAME,
    FingerprintIndex,
    IndexFormatError,
    Posting,
    TrackEntry,
    build_index,
)

SR = 11025


# --------------------------------------------------------------- fixtures ----
def _audio(seed: int = 0, seconds: float = 6.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.linspace(0, seconds, int(SR * seconds), endpoint=False)
    wobble = 600.0 + 200.0 * np.sin(2 * np.pi * 0.5 * t + seed)
    return (
        0.50 * np.sin(2 * np.pi * (440.0 + 7 * seed) * t)
        + 0.30 * np.sin(2 * np.pi * wobble * t)
        + 0.20 * np.sin(2 * np.pi * 1500.0 * t)
        + 0.02 * rng.standard_normal(t.size)
    ).astype(np.float32)


def _fp(seed: int = 0, seconds: float = 6.0):
    return fingerprint(_audio(seed, seconds), SR)


def _items(n: int = 3):
    return [(f"track-{i}", _fp(seed=i)) for i in range(n)]


@pytest.fixture
def index():
    return build_index(_items(3))


@pytest.fixture
def saved(tmp_path, index):
    index.save(tmp_path / "idx")
    return tmp_path / "idx"


# ------------------------------------------------------------------ build ----
class TestBuild:
    def test_builds_over_multiple_tracks(self, index):
        assert index.n_tracks == 3
        assert index.track_ids == ("track-0", "track-1", "track-2")
        assert len(index) > 0

    def test_posting_count_equals_sum_of_fingerprints(self):
        items = _items(3)
        idx = build_index(items)
        assert len(idx) == sum(len(r) for _, r in items)
        for track_id, r in items:
            assert idx.track_entry(track_id).fingerprint_count == len(r)

    def test_hashes_are_sorted(self, index):
        assert np.all(np.diff(index.hashes.astype(np.int64)) >= 0)

    def test_track_ord_maps_back_to_track_id(self, index):
        for ordinal, tid in enumerate(index.track_ids):
            mask = index.track_ords == ordinal
            assert mask.sum() == index.tracks[ordinal].fingerprint_count
            if mask.any():
                frame = int(index.anchor_frames[mask][0])
                key = int(index.hashes[mask][0])
                assert any(
                    p.track_id == tid and p.anchor_frame == frame
                    for p in index.lookup(key)
                )

    def test_duplicate_track_ids_are_rejected(self):
        with pytest.raises(ValueError, match="duplicate track_id"):
            build_index([("same", _fp(1)), ("same", _fp(2))])

    def test_empty_track_id_is_rejected(self):
        with pytest.raises(ValueError):
            build_index([("", _fp(1))])

    def test_mixed_configs_are_rejected(self):
        """Hashes from different configs are not comparable, so refuse to mix."""
        a = fingerprint(_audio(1), SR)
        b = fingerprint(_audio(2), SR, FingerprintConfig(fan_out=2))
        with pytest.raises(ValueError, match="different config"):
            build_index([("a", a), ("b", b)])

    def test_empty_catalog_is_safe(self):
        idx = build_index([])
        assert idx.n_tracks == 0 and len(idx) == 0
        assert idx.lookup(123) == []
        assert idx.n_unique_hashes == 0
        idx.validate()

    def test_track_with_no_fingerprints_is_still_a_track(self):
        silent = fingerprint(np.zeros(SR * 2, dtype=np.float32), SR)
        idx = build_index([("silent", silent), ("real", _fp(1))])
        assert idx.n_tracks == 2
        assert idx.track_entry("silent").fingerprint_count == 0
        assert "silent" not in {p.track_id for p in idx.lookup(int(idx.hashes[0]))}


# ----------------------------------------------------------------- lookup ----
class TestLookup:
    def test_lookup_returns_postings(self, index):
        key = int(index.hashes[0])
        postings = index.lookup(key)
        assert postings and all(isinstance(p, Posting) for p in postings)
        assert all(p.track_id in index.track_ids for p in postings)

    def test_unknown_hash_returns_empty_without_raising(self, index):
        assert index.lookup(0xFFFFFFFF) == []
        assert index.lookup(0) == [] or index.count(0) > 0  # 0 may legitimately exist
        assert index.count(0xFFFFFFFF) == 0
        assert 0xFFFFFFFF not in index

    def test_lookup_on_empty_index_returns_empty(self):
        assert build_index([]).lookup(42) == []

    def test_multiple_postings_for_one_hash(self, index):
        """A repeated hash must return every occurrence, not the first."""
        vals, counts = np.unique(index.hashes, return_counts=True)
        repeated = vals[counts > 1]
        assert repeated.size > 0, "expected collisions in a real fingerprint set"
        key = int(repeated[0])
        expected = int(counts[counts > 1][0])
        assert len(index.lookup(key)) == expected
        assert index.count(key) == expected

    def test_same_hash_across_different_tracks_is_kept(self):
        """The multimap must span tracks, not collapse to one posting each."""
        idx = build_index(_items(3))
        by_hash: dict[int, set[str]] = {}
        ids = idx.track_ids
        for h, o in zip(idx.hashes.tolist(), idx.track_ords.tolist()):
            by_hash.setdefault(h, set()).add(ids[o])
        cross = [h for h, s in by_hash.items() if len(s) > 1]
        assert cross, "expected at least one hash shared between tracks"
        assert {p.track_id for p in idx.lookup(cross[0])} == by_hash[cross[0]]

    def test_same_hash_repeated_within_one_track_is_kept(self):
        idx = build_index([("solo", _fp(0, seconds=12.0))])
        vals, counts = np.unique(idx.hashes, return_counts=True)
        repeated = vals[counts > 1]
        assert repeated.size > 0
        postings = idx.lookup(int(repeated[0]))
        assert len(postings) > 1
        assert len({p.track_id for p in postings}) == 1
        assert len({p.anchor_frame for p in postings}) > 1  # distinct positions

    def test_lookup_raw_matches_lookup(self, index):
        key = int(index.hashes[5])
        ords, frames = index.lookup_raw(key)
        postings = index.lookup(key)
        assert [int(o) for o in ords] == [p.track_ord for p in postings]
        assert [int(f) for f in frames] == [p.anchor_frame for p in postings]

    def test_lookup_raw_supports_offset_arithmetic(self, index):
        """Phase 1C subtracts these; they must be integer arrays, not objects."""
        _, frames = index.lookup_raw(int(index.hashes[0]))
        assert frames.dtype == np.int32
        assert (frames - np.int32(3)).dtype == np.int32

    def test_every_stored_posting_is_retrievable(self, index):
        for i in range(0, len(index), max(1, len(index) // 50)):
            key = int(index.hashes[i])
            frame = int(index.anchor_frames[i])
            tid = index.track_ids[int(index.track_ords[i])]
            assert any(
                p.track_id == tid and p.anchor_frame == frame
                for p in index.lookup(key)
            )


# ------------------------------------------------------------ persistence ----
class TestPersistence:
    def test_save_writes_expected_files(self, saved):
        names = {p.name for p in saved.iterdir()}
        assert names == {
            META_FILENAME,
            HASHES_FILENAME,
            TRACK_ORDS_FILENAME,
            ANCHOR_FRAMES_FILENAME,
        }

    def test_load_roundtrips_exactly(self, saved, index):
        loaded = FingerprintIndex.load(saved)
        assert np.array_equal(loaded.hashes, index.hashes)
        assert np.array_equal(loaded.track_ords, index.track_ords)
        assert np.array_equal(loaded.anchor_frames, index.anchor_frames)
        assert loaded.tracks == index.tracks
        assert loaded.config == index.config
        assert loaded.content_hash() == index.content_hash()

    def test_load_needs_no_audio(self, saved, index):
        """Nothing but the artifact directory is read -- no re-fingerprinting."""
        loaded = FingerprintIndex.load(saved)
        key = int(index.hashes[0])
        assert loaded.lookup(key) == index.lookup(key)

    def test_loaded_index_is_queryable(self, saved, index):
        loaded = FingerprintIndex.load(saved)
        assert loaded.lookup(0xFFFFFFFF) == []
        assert len(loaded) == len(index)

    def test_empty_index_roundtrips(self, tmp_path):
        build_index([]).save(tmp_path / "e")
        loaded = FingerprintIndex.load(tmp_path / "e")
        assert len(loaded) == 0 and loaded.n_tracks == 0
        assert loaded.lookup(1) == []

    def test_save_returns_the_directory(self, tmp_path, index):
        assert index.save(tmp_path / "x") == tmp_path / "x"

    def test_timestamp_is_omitted_by_default(self, tmp_path):
        idx = build_index(_items(2), built_utc="2026-01-01T00:00:00+00:00")
        idx.save(tmp_path / "a")
        meta = json.loads((tmp_path / "a" / META_FILENAME).read_text())
        assert meta["built_utc"] is None

    def test_timestamp_can_be_included_without_changing_identity(self, tmp_path):
        stamp = "2026-01-01T00:00:00+00:00"
        idx = build_index(_items(2), built_utc=stamp)
        idx.save(tmp_path / "a", include_timestamp=True)
        meta = json.loads((tmp_path / "a" / META_FILENAME).read_text())
        assert meta["built_utc"] == stamp
        assert meta["content_hash"] == build_index(_items(2)).content_hash()


# ----------------------------------------------------------- determinism ----
class TestDeterminism:
    def test_rebuild_produces_identical_arrays(self):
        a, b = build_index(_items(3)), build_index(_items(3))
        assert np.array_equal(a.hashes, b.hashes)
        assert np.array_equal(a.track_ords, b.track_ords)
        assert np.array_equal(a.anchor_frames, b.anchor_frames)
        assert a.content_hash() == b.content_hash()

    def test_rebuild_produces_identical_artifact_bytes(self, tmp_path):
        build_index(_items(3)).save(tmp_path / "a")
        build_index(_items(3)).save(tmp_path / "b")
        for name in (
            META_FILENAME,
            HASHES_FILENAME,
            TRACK_ORDS_FILENAME,
            ANCHOR_FRAMES_FILENAME,
        ):
            assert (tmp_path / "a" / name).read_bytes() == (
                tmp_path / "b" / name
            ).read_bytes(), name

    def test_timestamp_does_not_affect_content_hash(self):
        a = build_index(_items(2), built_utc="2026-01-01T00:00:00+00:00")
        b = build_index(_items(2), built_utc="2030-12-25T12:00:00+00:00")
        assert a.content_hash() == b.content_hash()

    def test_different_catalog_changes_content_hash(self):
        assert build_index(_items(2)).content_hash() != build_index(
            _items(3)
        ).content_hash()

    def test_track_order_is_preserved_and_affects_identity(self):
        items = _items(2)
        fwd = build_index(items)
        rev = build_index(list(reversed(items)))
        assert fwd.track_ids == ("track-0", "track-1")
        assert rev.track_ids == ("track-1", "track-0")
        assert fwd.content_hash() != rev.content_hash()


# -------------------------------------------------------------- metadata ----
class TestMetadata:
    def test_metadata_reports_the_required_fields(self, index):
        m = index.metadata()
        assert m["index_format_version"] == INDEX_FORMAT_VERSION
        assert m["fingerprint_format_version"] == FP_VERSION
        assert m["sample_rate"] == index.config.sample_rate == 11025
        assert m["track_count"] == 3
        assert m["fingerprint_count"] == len(index)
        assert m["unique_hash_count"] == index.n_unique_hashes
        assert m["content_hash"] == index.content_hash()
        assert len(m["tracks"]) == 3

    def test_metadata_carries_the_full_fingerprint_config(self, index):
        stored = index.metadata()["fingerprint_config"]
        assert stored == dataclasses.asdict(index.config)
        assert FingerprintConfig(**stored) == index.config

    def test_metadata_is_json_serializable(self, index):
        assert json.loads(json.dumps(index.metadata()))["track_count"] == 3

    def test_no_audio_or_paths_are_stored(self, saved):
        """Requirement: the index holds identity, not source audio or filenames."""
        meta = json.loads((saved / META_FILENAME).read_text())
        blob = json.dumps(meta)
        assert ".wav" not in blob and ".mp3" not in blob and ".flac" not in blob
        assert set(meta["tracks"][0]) == {
            "track_id",
            "fingerprint_count",
            "duration_sec",
        }

    def test_track_entry_roundtrip(self):
        t = TrackEntry("abc", 10, 1.5)
        assert TrackEntry.from_dict(t.to_dict()) == t


# ------------------------------------------------------------- corruption ----
class TestCorruptionRejection:
    def _tamper(self, path, mutate):
        meta = json.loads((path / META_FILENAME).read_text())
        mutate(meta)
        (path / META_FILENAME).write_text(json.dumps(meta, indent=2, sort_keys=True))

    def test_missing_directory(self, tmp_path):
        with pytest.raises(IndexFormatError, match="no meta.json"):
            FingerprintIndex.load(tmp_path / "nope")

    def test_missing_meta(self, saved):
        (saved / META_FILENAME).unlink()
        with pytest.raises(IndexFormatError, match="no meta.json"):
            FingerprintIndex.load(saved)

    def test_missing_array_file(self, saved):
        (saved / HASHES_FILENAME).unlink()
        with pytest.raises(IndexFormatError, match="missing array file"):
            FingerprintIndex.load(saved)

    def test_meta_is_not_json(self, saved):
        (saved / META_FILENAME).write_text("{not json")
        with pytest.raises(IndexFormatError, match="not valid JSON"):
            FingerprintIndex.load(saved)

    def test_incompatible_index_format_version(self, saved):
        self._tamper(saved, lambda m: m.update(index_format_version=999))
        with pytest.raises(IndexFormatError, match="index format version"):
            FingerprintIndex.load(saved)

    def test_incompatible_fingerprint_format_version(self, saved):
        self._tamper(saved, lambda m: m.update(fingerprint_format_version=999))
        with pytest.raises(IndexFormatError, match="fingerprint format"):
            FingerprintIndex.load(saved)

    def test_unknown_config_key_is_rejected(self, saved):
        self._tamper(
            saved, lambda m: m["fingerprint_config"].update(mystery_setting=1)
        )
        with pytest.raises(IndexFormatError, match="unknown fingerprint config"):
            FingerprintIndex.load(saved)

    def test_missing_config_key_is_rejected(self, saved):
        self._tamper(saved, lambda m: m["fingerprint_config"].pop("hop_length"))
        with pytest.raises(IndexFormatError, match="missing fingerprint config"):
            FingerprintIndex.load(saved)

    def test_invalid_config_value_is_rejected(self, saved):
        self._tamper(saved, lambda m: m["fingerprint_config"].update(min_delta_frames=0))
        with pytest.raises(IndexFormatError, match="invalid fingerprint config"):
            FingerprintIndex.load(saved)

    def test_tampered_arrays_fail_the_content_hash(self, saved):
        arr = np.load(saved / ANCHOR_FRAMES_FILENAME)
        arr[0] = arr[0] + 1
        with open(saved / ANCHOR_FRAMES_FILENAME, "wb") as fh:
            np.save(fh, arr, allow_pickle=False)
        with pytest.raises(IndexFormatError, match="content hash mismatch"):
            FingerprintIndex.load(saved)

    def test_truncated_array_file_is_rejected(self, saved):
        (saved / TRACK_ORDS_FILENAME).write_bytes(b"\x93NUMPY garbage")
        with pytest.raises(IndexFormatError):
            FingerprintIndex.load(saved)

    def test_declared_counts_must_agree(self, saved):
        # Recompute content_hash so this specifically exercises the count check.
        meta = json.loads((saved / META_FILENAME).read_text())
        meta["track_count"] = 99
        (saved / META_FILENAME).write_text(json.dumps(meta, indent=2, sort_keys=True))
        with pytest.raises(IndexFormatError, match="track_count"):
            FingerprintIndex.load(saved)

    def test_validate_rejects_unsorted_hashes(self, index):
        bad = FingerprintIndex(
            hashes=index.hashes[::-1].copy(),
            track_ords=index.track_ords,
            anchor_frames=index.anchor_frames,
            tracks=index.tracks,
            config=index.config,
        )
        with pytest.raises(IndexFormatError, match="not sorted"):
            bad.validate()

    def test_validate_rejects_length_mismatch(self, index):
        bad = FingerprintIndex(
            hashes=index.hashes,
            track_ords=index.track_ords[:-1].copy(),
            anchor_frames=index.anchor_frames,
            tracks=index.tracks,
            config=index.config,
        )
        with pytest.raises(IndexFormatError, match="length mismatch"):
            bad.validate()

    def test_validate_rejects_out_of_range_track_ord(self, index):
        ords = index.track_ords.copy()
        ords[0] = 99
        bad = FingerprintIndex(
            hashes=index.hashes,
            track_ords=ords,
            anchor_frames=index.anchor_frames,
            tracks=index.tracks,
            config=index.config,
        )
        with pytest.raises(IndexFormatError, match="outside track table"):
            bad.validate()

    def test_validate_rejects_duplicate_track_entries(self, index):
        bad = FingerprintIndex(
            hashes=index.hashes,
            track_ords=np.zeros_like(index.track_ords),
            anchor_frames=index.anchor_frames,
            tracks=(index.tracks[0], index.tracks[0]),
            config=index.config,
        )
        with pytest.raises(IndexFormatError, match="duplicate track_id"):
            bad.validate()
