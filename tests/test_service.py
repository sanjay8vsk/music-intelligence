"""Tests for the recognition service.

This layer owns wiring, not recognition: which catalog is loaded, what metadata
is attached, and that the frozen pipeline is called with the benchmarked
configuration. Accuracy is the cascade's and is tested against it.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import soundfile as sf

from musicintel.catalog.ingest import build_catalog_index, ingest_directory
from musicintel.catalog.store import CatalogStore, CatalogStoreError
from musicintel.recognition.decision import Decision
from musicintel.recognition.fingerprint import FingerprintConfig, load_audio
from musicintel.recognition.gated_cascade import GATED_RATE_GRID, GatedCascadeConfig
from musicintel.service.recognition import (
    GATE_THRESHOLD,
    STAGE1_THRESHOLD,
    STAGE2_THRESHOLD,
    Identification,
    RecognitionService,
    default_cascade_config,
)

SR = 11025


def _write(path, seconds=12.0, seed=0):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, seconds, int(SR * seconds), endpoint=False)
    wob = 600.0 + 200.0 * np.sin(2 * np.pi * 0.5 * t + seed)
    y = (0.50 * np.sin(2 * np.pi * (440.0 + 7 * seed) * t)
         + 0.30 * np.sin(2 * np.pi * wob * t)
         + 0.20 * np.sin(2 * np.pi * 1500.0 * t)
         + 0.02 * rng.standard_normal(t.size)).astype(np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, y, SR, subtype="PCM_16")


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("svc")
    store = CatalogStore(tmp / "store")
    audio = {}
    for cid, seeds in (("acme", [1, 2, 3]), ("globex", [11, 12, 13])):
        d = tmp / "audio" / cid
        for s in seeds:
            _write(d / f"{cid}_{s}.wav", seed=s)
        r = ingest_directory(d)
        store.save(r.catalog, build_catalog_index(r.catalog, r.fingerprints),
                   catalog_id=cid)
        audio[cid] = d
    return store, audio


def _excerpt(path, start=3.0, length=6.0):
    y, sr = load_audio(path, FingerprintConfig())
    return y[int(sr * start):int(sr * (start + length))], sr


# --------------------------------------------------------------- identify ---
class TestIdentify:
    def test_in_catalog_audio_matches_and_carries_metadata(self, world):
        store, audio = world
        svc = RecognitionService(store)
        y, sr = _excerpt(audio["acme"] / "acme_1.wav")
        r = svc.identify(y, sr, "acme")
        assert r.is_match and r.decision is Decision.MATCH
        assert r.track_id == "acme_1"
        assert r.track is not None
        assert r.track.track_id == "acme_1" and len(r.track.sha256) == 64
        assert r.track.duration_sec > 0

    def test_position_is_reported_in_reference_seconds(self, world):
        store, audio = world
        svc = RecognitionService(store)
        y, sr = _excerpt(audio["acme"] / "acme_2.wav", start=5.0, length=5.0)
        r = svc.identify(y, sr, "acme")
        assert r.is_match
        assert r.offset_seconds == pytest.approx(5.0, abs=0.5)

    def test_out_of_catalog_audio_is_rejected(self, world, tmp_path):
        store, _ = world
        svc = RecognitionService(store)
        _write(tmp_path / "outsider.wav", seed=99)
        y, sr = _excerpt(tmp_path / "outsider.wav", start=1.0, length=6.0)
        r = svc.identify(y, sr, "acme")
        assert not r.is_match
        assert r.track_id is None and r.track is None and r.offset_seconds is None

    def test_identify_file_decodes_and_identifies(self, world):
        store, audio = world
        svc = RecognitionService(store)
        r = svc.identify_file(audio["acme"] / "acme_3.wav", "acme")
        assert r.is_match and r.track_id == "acme_3"

    def test_latency_is_recorded(self, world):
        store, audio = world
        svc = RecognitionService(store)
        y, sr = _excerpt(audio["acme"] / "acme_1.wav")
        assert svc.identify(y, sr, "acme").latency_ms > 0


# -------------------------------------------------------------- isolation ---
class TestTenantIsolation:
    def test_a_track_in_one_catalog_is_not_found_in_another(self, world):
        """The acceptance criterion: cross-tenant isolation, proven by test."""
        store, audio = world
        svc = RecognitionService(store)
        y, sr = _excerpt(audio["acme"] / "acme_1.wav")
        assert svc.identify(y, sr, "acme").is_match
        other = svc.identify(y, sr, "globex")
        assert not other.is_match
        assert other.track_id is None
        assert other.catalog_id == "globex"

    def test_isolation_holds_in_both_directions(self, world):
        store, audio = world
        svc = RecognitionService(store)
        y, sr = _excerpt(audio["globex"] / "globex_11.wav")
        assert svc.identify(y, sr, "globex").is_match
        assert not svc.identify(y, sr, "acme").is_match

    def test_only_the_named_catalog_is_loaded(self, world):
        store, audio = world
        svc = RecognitionService(store)
        y, sr = _excerpt(audio["acme"] / "acme_1.wav")
        svc.identify(y, sr, "acme")
        assert set(svc._cache) == {"acme"}      # globex was never touched

    def test_unknown_catalog_raises(self, world):
        store, audio = world
        svc = RecognitionService(store)
        y, sr = _excerpt(audio["acme"] / "acme_1.wav")
        with pytest.raises(CatalogStoreError):
            svc.identify(y, sr, "nonexistent")

    def test_traversing_catalog_id_raises(self, world):
        store, audio = world
        svc = RecognitionService(store)
        y, sr = _excerpt(audio["acme"] / "acme_1.wav")
        with pytest.raises(CatalogStoreError):
            svc.identify(y, sr, "../store")


# ----------------------------------------------------------------- config ---
class TestConfiguration:
    def test_defaults_are_the_benchmarked_operating_point(self):
        """Changing these invalidates every published number."""
        c = default_cascade_config()
        assert c.stage1_threshold == STAGE1_THRESHOLD == 0.026316
        assert c.gate_threshold == GATE_THRESHOLD == 0.032520
        assert c.stage2_threshold == STAGE2_THRESHOLD == 0.028571
        assert c.rate_grid == GATED_RATE_GRID == (-4.0, -2.0, 2.0, 4.0)
        assert c.probe_seconds == 2.0 and c.min_aligned_landmarks == 5

    def test_the_cascade_config_is_actually_used(self, world):
        """A strict stage-1 threshold must change the outcome, proving the
        service passes its config through rather than ignoring it."""
        store, audio = world
        y, sr = _excerpt(audio["acme"] / "acme_1.wav")
        strict = GatedCascadeConfig(stage1_threshold=1.0, gate_threshold=1.0,
                                    stage2_threshold=1.0, min_aligned_landmarks=5)
        assert not RecognitionService(store, cascade_config=strict).identify(
            y, sr, "acme").is_match
        assert RecognitionService(store).identify(y, sr, "acme").is_match

    def test_stage_two_is_reachable_through_the_service(self, world):
        """Wiring check: with the gate open, a rate-corrected match surfaces as
        stage 2 with a non-zero correction. (Whether the calibrated gate opens
        on a given catalog is the cascade's business, not this layer's.)"""
        from musicintel.eval import degradation as dg
        store, audio = world
        y, sr = _excerpt(audio["acme"] / "acme_1.wav")
        sped = np.asarray(dg.change_speed(y, sr, 2.0)[0], dtype=np.float32)
        open_gate = GatedCascadeConfig(
            stage1_threshold=STAGE1_THRESHOLD, gate_threshold=0.0,
            stage2_threshold=STAGE2_THRESHOLD, min_aligned_landmarks=5)
        r = RecognitionService(store, cascade_config=open_gate).identify(sped, sr, "acme")
        if r.is_match:
            assert r.stage == 2 and r.rate_percent != 0.0 and r.escalated
            assert r.track_id == "acme_1"


# ------------------------------------------------------------------ cache ---
class TestCaching:
    def test_a_catalog_is_loaded_once(self, world):
        store, _ = world
        svc = RecognitionService(store)
        assert svc.get("acme") is svc.get("acme")

    def test_unload_clears_it(self, world):
        store, _ = world
        svc = RecognitionService(store)
        first = svc.get("acme")
        svc.unload("acme")
        assert svc.get("acme") is not first

    def test_caching_can_be_disabled(self, world):
        store, _ = world
        svc = RecognitionService(store, cache_catalogs=False)
        assert svc.get("acme") is not svc.get("acme")

    def test_catalogs_lists_what_the_store_holds(self, world):
        store, _ = world
        assert RecognitionService(store).catalogs() == ["acme", "globex"]


# --------------------------------------------------------------- contract ---
class TestContract:
    def test_result_is_json_serializable(self, world):
        store, audio = world
        y, sr = _excerpt(audio["acme"] / "acme_1.wav")
        d = RecognitionService(store).identify(y, sr, "acme").to_dict()
        assert json.loads(json.dumps(d))["decision"] == "MATCH"
        assert d["track"]["track_id"] == "acme_1"

    def test_no_probability_surface(self, world):
        store, audio = world
        y, sr = _excerpt(audio["acme"] / "acme_1.wav")
        r = RecognitionService(store).identify(y, sr, "acme")
        for banned in ("confidence", "probability", "certainty"):
            assert not hasattr(r, banned) and banned not in r.to_dict()

    def test_rejection_withholds_the_track(self, world, tmp_path):
        store, _ = world
        _write(tmp_path / "o.wav", seed=77)
        y, sr = _excerpt(tmp_path / "o.wav", start=1.0, length=6.0)
        r = RecognitionService(store).identify(y, sr, "acme")
        assert isinstance(r, Identification)
        assert r.decision is Decision.NO_MATCH
        assert r.track_id is None and r.track is None


class TestWarmUp:
    """Construction pays every first-use cost, so no query pays it.

    Asserted causally, in a subprocess with a cold module table, because that is
    the only place the property is observable. In-process it is not: any earlier
    test that fingerprints audio has already loaded librosa's lazy submodules,
    so the first request looks warm whether or not warm-up ran. An elapsed-time
    bound is no better -- it fails on a loaded machine and passes on an idle one
    regardless of whether warm-up worked.

    The regression: librosa's lazy loader imports scipy on the first `stft` and
    again on the first `resample`. An unwarmed first request spent 1,735 ms in
    recognition against 13 ms steady state, all of it imports.
    """

    HEAVY = ("scipy", "librosa", "numba", "llvmlite", "soxr", "soundfile", "sklearn")

    _PROBE = '''
import json, sys, tempfile
sys.path.insert(0, {repo!r})
import numpy as np
from musicintel.catalog.store import CatalogStore
from musicintel.recognition.cascade import apply_rate
from musicintel.recognition.fingerprint import fingerprint
from musicintel.recognition.matcher import _best_cluster_compiled
from musicintel.service.recognition import RecognitionService

svc = RecognitionService(CatalogStore(tempfile.mkdtemp()), warm_up={warm})

# Everything the first query needs is imported by now; what matters is whether
# calling it pulls in anything further.
settled = set(sys.modules)
compiled = len(_best_cluster_compiled.signatures)

cfg = svc.fingerprint_config
sr = cfg.sample_rate
t = np.arange(sr, dtype=np.float32) / sr
tone = (0.4 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
fingerprint(tone, sr, cfg)
apply_rate(tone, sr, 2.0)

heavy = {heavy!r}
print(json.dumps({{
    "lazily_imported": sorted({{m for m in set(sys.modules) - settled
                               if m.split(".")[0] in heavy}}),
    "compiled_after_construction": compiled,
}}))
'''

    def _probe(self, warm: bool) -> dict:
        import json as _json
        import subprocess
        import sys
        from pathlib import Path

        repo = str(Path(__file__).resolve().parent.parent)
        code = self._PROBE.format(repo=repo, warm=warm, heavy=self.HEAVY)
        p = subprocess.run([sys.executable, "-c", code],
                           capture_output=True, text=True)
        assert p.returncode == 0, p.stderr[-2000:]
        return _json.loads(p.stdout.strip().splitlines()[-1])

    def test_a_warmed_service_leaves_nothing_for_the_first_query_to_import(self):
        lazy = self._probe(warm=True)["lazily_imported"]
        # Summarised: the unwarmed case pulls in ~400 modules, and a failure
        # message that lists them all is unreadable.
        assert lazy == [], (
            f"first fingerprint/resample imported {len(lazy)} modules "
            f"({', '.join(lazy[:5])}...); RecognitionService._warm_up must "
            f"force these at construction")

    def test_a_warmed_service_has_already_compiled_the_matcher_kernel(self):
        assert self._probe(warm=True)["compiled_after_construction"] > 0

    def test_the_probe_can_see_the_cost_it_claims_to_prevent(self):
        """Control arm: without warm-up the same first call does import them.

        Without this the passing test above would be unfalsifiable -- it would
        look identical if librosa had simply stopped importing anything lazily.
        Should this ever fail, the warm-up may have become unnecessary rather
        than broken; check before deleting anything.
        """
        assert self._probe(warm=False)["lazily_imported"] != []

    def test_warm_up_is_reported_and_skippable(self):
        import tempfile
        store = CatalogStore(tempfile.mkdtemp())
        assert RecognitionService(store, warm_up=True).warm_up_seconds > 0
        assert RecognitionService(store, warm_up=False).warm_up_seconds == 0.0
