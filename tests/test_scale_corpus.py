"""Tests for the Stage 2 scale-validation corpus manifest.

The corpus exists to test the system at 500 tracks, so what matters here is
that it is genuinely 500 *distinct* recordings and genuinely separate from the
frozen Phase 0/1 evaluation corpora. A scale test contaminated with the
evaluation fixtures would measure the wrong thing and quietly invalidate the
frozen reports it sits beside.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from musicintel.eval.manifest import Manifest
from musicintel.eval.negatives import NegativeSet

REPO_ROOT = Path(__file__).resolve().parents[1]
SCALE = REPO_ROOT / "eval/fixtures/scale_corpus_manifest.json"
FROZEN = REPO_ROOT / "eval/fixtures/manifest.json"
NEGATIVES = REPO_ROOT / "eval/fixtures/negatives_manifest.json"
TARGET = 500


@pytest.fixture(scope="module")
def scale():
    if not SCALE.is_file():
        pytest.skip("scale corpus not built in this checkout")
    return Manifest.load(SCALE)


@pytest.fixture(scope="module")
def frozen_recordings():
    m = Manifest.load(FROZEN)
    ns = NegativeSet.load(NEGATIVES)
    return ({t.sha256 for t in m.tracks} | {s.sha256 for s in ns.sources},
            {t.track_id for t in m.tracks} | {s.track_id for s in ns.sources})


class TestSize:
    def test_reaches_the_stage2_acceptance_bar(self, scale):
        assert len(scale) >= TARGET

    def test_every_track_is_a_distinct_recording(self, scale):
        """The fetch tool does no content dedup; the screen must."""
        hashes = [t.sha256 for t in scale.tracks]
        assert len(set(hashes)) == len(hashes)

    def test_track_ids_are_unique(self, scale):
        ids = [t.track_id for t in scale.tracks]
        assert len(set(ids)) == len(ids)

    def test_carries_meaningful_audio(self, scale):
        assert all(t.duration_sec > 0 for t in scale.tracks)
        assert sum(t.duration_sec for t in scale.tracks) / 3600 > 5.0


class TestSeparationFromFrozenCorpora:
    def test_no_recording_overlaps_the_frozen_corpora(self, scale, frozen_recordings):
        """By content hash -- a re-encode under a new name would defeat ids alone."""
        frozen_sha, _ = frozen_recordings
        assert not {t.sha256 for t in scale.tracks} & frozen_sha

    def test_no_track_id_collides_with_the_frozen_corpora(self, scale, frozen_recordings):
        _, frozen_ids = frozen_recordings
        assert not {t.track_id for t in scale.tracks} & frozen_ids

    def test_the_frozen_manifests_are_untouched(self):
        """This corpus must be additive; the frozen hashes are cited in every
        Phase 0/1 report."""
        assert Manifest.load(FROZEN).content_hash() == (
            "4006aacb0abc1e7f2e12eee8a9f205a6f6b8cc563fef7a9396872cf797767a0e")
        assert NegativeSet.load(NEGATIVES).content_hash() == (
            "fe1b10251c26251571422f3ec43bac413dee926e33fc649aac43513d361c2564")


class TestProvenanceAndLicensing:
    def test_every_track_records_its_provenance(self, scale):
        for t in scale.tracks:
            assert t.source and t.license and t.license_url
            assert len(t.sha256) == 64
            assert t.path.startswith("data/")

    def test_licences_are_permissive(self, scale):
        """No -nc or -nd: the corpus supports a commercial product and is
        transformed by the benchmark."""
        for t in scale.tracks:
            u = t.license_url.lower()
            assert "-nd" not in u and "-nc" not in u and "nc-" not in u

    def test_licences_are_the_expected_creative_commons_set(self, scale):
        allowed = {"CC0-1.0", "CC-BY", "CC-BY-SA", "PDM"}
        assert set(scale.license_counts()) <= allowed

    def test_artist_diversity(self, scale):
        """The fetch caps two tracks per creator; a 500-track corpus drawn from
        a handful of artists would not exercise collision behaviour honestly."""
        artists = {t.artist for t in scale.tracks if t.artist}
        assert len(artists) >= len(scale) // 4


class TestManifestIntegrity:
    def test_content_hash_is_stable(self, scale):
        assert Manifest.load(SCALE).content_hash() == scale.content_hash()

    def test_stored_hash_matches_recomputation(self, scale):
        import json
        stored = json.loads(SCALE.read_text())["content_hash"]
        assert stored == scale.content_hash()

    def test_manifest_holds_no_audio(self):
        blob = SCALE.read_text() if SCALE.is_file() else ""
        if not blob:
            pytest.skip("scale corpus not built")
        assert "hashes" not in blob and "anchor_frames" not in blob
