"""Focused tests for acoustic fingerprint extraction.

Like the harness tests, these synthesize their own audio rather than depending
on the fixture corpus, so the suite runs anywhere. The corpus is for the smoke
test and the eventual benchmark, not for unit tests: a unit test that needs
44 tracks is not a unit test.
"""

from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf

from musicintel.recognition import fingerprint as fp
from musicintel.recognition.fingerprint import (
    DELTA_BITS,
    FREQ_BITS,
    MAX_DELTA_FRAMES,
    MAX_FREQ_BIN,
    FingerprintConfig,
    fingerprint,
    fingerprint_file,
    pack_hash,
    unpack_hash,
)

SR = 11025


# --------------------------------------------------------------- fixtures ----
def _music_like(seconds: float = 8.0, seed: int = 0, sr: int = SR) -> np.ndarray:
    """Deterministic broadband audio with moving partials.

    Pure steady tones are a bad fixture: their spectrogram is constant in time,
    so every frame looks alike and the peak picker has nothing to discriminate.
    A vibrato-ish sweep plus a little noise gives real time structure.
    """
    rng = np.random.default_rng(seed)
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    wobble = 600.0 + 200.0 * np.sin(2 * np.pi * 0.5 * t + seed)
    y = (
        0.50 * np.sin(2 * np.pi * (440.0 + 7 * seed) * t)
        + 0.30 * np.sin(2 * np.pi * wobble * t)
        + 0.20 * np.sin(2 * np.pi * 1500.0 * t)
        + 0.02 * rng.standard_normal(t.size)
    )
    return y.astype(np.float32)


@pytest.fixture
def audio():
    return _music_like()


# --------------------------------------------------------------- packing ----
class TestHashPacking:
    def test_roundtrip(self):
        for f1, f2, dt in [(0, 0, 1), (161, 41, 5), (MAX_FREQ_BIN, 0, MAX_DELTA_FRAMES)]:
            assert unpack_hash(pack_hash(f1, f2, dt)) == (f1, f2, dt)

    def test_key_fits_in_32_bits(self):
        assert pack_hash(MAX_FREQ_BIN, MAX_FREQ_BIN, MAX_DELTA_FRAMES) < 2**32
        assert FREQ_BITS * 2 + DELTA_BITS <= 32

    def test_distinct_landmarks_give_distinct_keys(self):
        keys = {pack_hash(f1, f2, dt)
                for f1 in (10, 11) for f2 in (20, 21) for dt in (1, 2)}
        assert len(keys) == 8

    @pytest.mark.parametrize(
        "args", [(-1, 0, 1), (0, -1, 1), (0, 0, -1),
                 (MAX_FREQ_BIN + 1, 0, 1), (0, 0, MAX_DELTA_FRAMES + 1)]
    )
    def test_out_of_range_is_rejected(self, args):
        with pytest.raises(ValueError):
            pack_hash(*args)

    def test_packing_is_pure_integer_math(self):
        """No float enters the key, so it is identical on any platform."""
        assert isinstance(pack_hash(100, 200, 30), int)


# ------------------------------------------------------------ determinism ----
class TestDeterminism:
    def test_same_audio_gives_identical_fingerprints(self, audio):
        a = fingerprint(audio, SR)
        b = fingerprint(audio, SR)
        assert len(a) == len(b)
        assert np.array_equal(a.hashes, b.hashes)
        assert np.array_equal(a.anchor_frames, b.anchor_frames)
        assert a.peak_count == b.peak_count

    def test_same_file_gives_identical_fingerprints(self, tmp_path, audio):
        p = tmp_path / "a.wav"
        sf.write(p, audio, SR, subtype="PCM_16")
        a, b = fingerprint_file(p), fingerprint_file(p)
        assert np.array_equal(a.hashes, b.hashes)
        assert np.array_equal(a.anchor_frames, b.anchor_frames)

    def test_output_order_is_canonical(self, audio):
        """Sorted by (anchor_frame, hash) -- otherwise 'identical' is luck."""
        r = fingerprint(audio, SR)
        keys = list(zip(r.anchor_frames.tolist(), r.hashes.tolist()))
        assert keys == sorted(keys)

    def test_a_copy_of_the_samples_changes_nothing(self, audio):
        assert np.array_equal(
            fingerprint(audio, SR).hashes, fingerprint(audio.copy(), SR).hashes
        )

    def test_input_is_not_mutated(self, audio):
        before = audio.copy()
        fingerprint(audio, SR)
        assert np.array_equal(audio, before)


# ------------------------------------------------------------ degeneracy ----
class TestDegenerateInput:
    def test_empty_audio_is_safe(self):
        r = fingerprint(np.zeros(0, dtype=np.float32), SR)
        assert len(r) == 0 and r.peak_count == 0 and r.duration_sec == 0.0
        assert r.density == 0.0  # no ZeroDivisionError

    def test_very_short_audio_is_safe(self):
        for n in (1, 64, int(SR * 0.05)):
            r = fingerprint(_music_like(seconds=n / SR), SR)
            assert len(r) >= 0  # the point is that it does not raise

    def test_shorter_than_one_fft_window_is_safe(self):
        r = fingerprint(_music_like(seconds=512 / SR), SR)
        assert isinstance(len(r), int)

    def test_digital_silence_produces_no_peaks(self):
        """The failure mode this guards: a flat spectrum is all local maxima."""
        r = fingerprint(np.zeros(SR * 5, dtype=np.float32), SR)
        assert r.peak_count == 0
        assert len(r) == 0

    def test_constant_signal_produces_no_fingerprints(self):
        """A DC signal is not silence: the STFT zero-pads, so it steps 0 -> k -> 0
        and those two step edges are genuine broadband events. The correct
        behaviour is a couple of edge peaks and NO landmarks, because the only
        peak pair available is further apart than max_delta_frames."""
        y = np.full(SR * 3, 0.5, dtype=np.float32)
        spec = fp.spectrogram(y, fp.DEFAULT_CONFIG)
        _, frames = fp.find_peaks(spec, fp.DEFAULT_CONFIG)
        last = spec.shape[1] - 1
        assert set(frames.tolist()) <= {0, last}  # edges only, nothing sustained
        assert len(fingerprint(y, SR)) == 0

    def test_pure_noise_stays_within_the_density_cap(self):
        """Noise must not explode the peak count; the caps are the guarantee."""
        rng = np.random.default_rng(7)
        y = rng.standard_normal(SR * 5).astype(np.float32)
        r = fingerprint(y, SR)
        assert r.peak_density <= fp.DEFAULT_CONFIG.target_peak_density * 1.2


# ---------------------------------------------------------------- density ----
class TestDensity:
    def test_peak_density_is_near_target(self, audio):
        r = fingerprint(audio, SR)
        target = fp.DEFAULT_CONFIG.target_peak_density
        assert 0.4 * target <= r.peak_density <= 1.2 * target

    def test_fingerprint_density_is_bounded_by_fanout(self, audio):
        cfg = fp.DEFAULT_CONFIG
        r = fingerprint(audio, SR)
        assert r.density <= cfg.target_peak_density * cfg.fan_out * 1.2
        assert len(r) > 0

    def test_density_target_is_configurable(self, audio):
        sparse = fingerprint(audio, SR, FingerprintConfig(target_peak_density=10.0))
        dense = fingerprint(audio, SR, FingerprintConfig(target_peak_density=60.0))
        assert sparse.peak_count < dense.peak_count

    def test_fanout_controls_fingerprints_per_peak(self, audio):
        one = fingerprint(audio, SR, FingerprintConfig(fan_out=1))
        five = fingerprint(audio, SR, FingerprintConfig(fan_out=5))
        assert len(one) < len(five)

    def test_longer_audio_yields_proportionally_more(self):
        short = fingerprint(_music_like(seconds=4.0), SR)
        long_ = fingerprint(_music_like(seconds=12.0), SR)
        assert len(long_) > len(short)


# -------------------------------------------------------------- structure ----
class TestLandmarkStructure:
    def test_result_exposes_hash_and_anchor_time(self, audio):
        r = fingerprint(audio, SR)
        lm = r.landmarks()
        assert len(lm) == len(r)
        first = lm[0]
        assert isinstance(first.hash, int)
        assert isinstance(first.anchor_frame, int)
        assert isinstance(first.anchor_time, float)

    def test_anchor_times_are_valid(self, audio):
        r = fingerprint(audio, SR)
        times = r.anchor_times
        assert np.all(times >= 0.0)
        # Anchors may sit one window past the last sample because the STFT is
        # centered; they must never wander beyond that.
        limit = r.duration_sec + fp.DEFAULT_CONFIG.n_fft / SR
        assert np.all(times <= limit)
        assert np.all(np.diff(r.anchor_frames) >= 0)  # non-decreasing

    def test_delta_times_are_positive_and_bounded(self, audio):
        cfg = fp.DEFAULT_CONFIG
        r = fingerprint(audio, SR)
        deltas = np.array([unpack_hash(h)[2] for h in r.hashes])
        assert deltas.size > 0
        assert np.all(deltas >= cfg.min_delta_frames)
        assert np.all(deltas <= cfg.max_delta_frames)
        assert np.all(deltas > 0)

    def test_frequencies_stay_inside_the_configured_band(self, audio):
        cfg = fp.DEFAULT_CONFIG
        r = fingerprint(audio, SR)
        lo = int(np.ceil(cfg.freq_min_hz / cfg.bin_hz))
        hi = int(np.floor(cfg.freq_max_hz / cfg.bin_hz))
        for h in r.hashes:
            f1, f2, _ = unpack_hash(h)
            assert lo <= f1 <= hi and lo <= f2 <= hi

    def test_arrays_have_compact_dtypes(self, audio):
        r = fingerprint(audio, SR)
        assert r.hashes.dtype == np.uint32
        assert r.anchor_frames.dtype == np.int32
        assert r.nbytes == 8 * len(r)  # 4 + 4 bytes per landmark

    def test_index_rows_are_persistable(self, audio):
        r = fingerprint(audio, SR)
        rows = r.to_index_rows("track-x")
        assert len(rows) == len(r)
        h, tid, frame = rows[0]
        assert isinstance(h, int) and tid == "track-x" and isinstance(frame, int)
        # Integer frames, not floats: the matcher bins offset differences and
        # integers bin exactly.
        assert all(isinstance(f, int) for _, _, f in rows[:50])

    def test_temporal_information_is_preserved(self, audio):
        """The whole point versus MFCC mean/std: anchors span the timeline."""
        r = fingerprint(audio, SR)
        times = r.anchor_times
        assert times.max() - times.min() > 0.5 * r.duration_sec
        assert len(np.unique(r.anchor_frames)) > 10


# ------------------------------------------------------------ discrimination -
class TestDiscrimination:
    def test_different_audio_gives_different_fingerprints(self):
        a = fingerprint(_music_like(seed=1), SR)
        b = fingerprint(_music_like(seed=2), SR)
        assert not np.array_equal(a.hashes, b.hashes)
        overlap = len(set(a.hashes.tolist()) & set(b.hashes.tolist()))
        assert overlap < 0.9 * min(len(a), len(b))

    def test_noise_and_music_do_not_coincide(self):
        rng = np.random.default_rng(3)
        noise = fingerprint(rng.standard_normal(SR * 8).astype(np.float32), SR)
        music = fingerprint(_music_like(seconds=8.0), SR)
        assert not np.array_equal(noise.hashes, music.hashes)


# -------------------------------------------------------------- invariance ---
class TestGainInvariance:
    def test_uniform_gain_barely_moves_the_peak_set(self, audio):
        """The percentile gate is relative, so a volume change should not
        rewrite the fingerprint. Not asserted as exact: resampling-free gain is
        still float multiplication, so a few boundary peaks may flip."""
        quiet = fingerprint((audio * 0.25).astype(np.float32), SR)
        loud = fingerprint((audio * 2.0).astype(np.float32), SR)
        shared = len(set(quiet.hashes.tolist()) & set(loud.hashes.tolist()))
        assert shared > 0.8 * min(len(quiet), len(loud))


# ------------------------------------------------------------------ config ---
class TestConfig:
    def test_defaults_match_the_documented_representation(self):
        c = fp.DEFAULT_CONFIG
        assert (c.sample_rate, c.n_fft, c.hop_length) == (11025, 1024, 128)
        assert c.n_bins == 513
        assert round(c.frame_rate, 1) == 86.1

    def test_validate_rejects_unrepresentable_settings(self):
        with pytest.raises(ValueError):
            FingerprintConfig(n_fft=8192).validate()  # >1024 bins, will not pack
        with pytest.raises(ValueError):
            FingerprintConfig(max_delta_frames=MAX_DELTA_FRAMES + 1).validate()
        with pytest.raises(ValueError):
            FingerprintConfig(min_delta_frames=0).validate()
        with pytest.raises(ValueError):
            FingerprintConfig(freq_min_hz=3000, freq_max_hz=200).validate()
        with pytest.raises(ValueError):
            FingerprintConfig(fan_out=0).validate()

    def test_resampling_keeps_the_configured_rate(self):
        y = _music_like(seconds=4.0, sr=22050)
        r = fingerprint(y, 22050)
        assert r.config.sample_rate == SR
        assert abs(r.duration_sec - 4.0) < 0.05
