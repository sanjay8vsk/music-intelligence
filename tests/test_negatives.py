"""Tests for the Phase 1G negative-set construction.

These guard the measuring instrument, not the recognizer. A negative set that
quietly contains indexed audio, or that splits one recording across calibration
and evaluation, produces a false-accept rate that looks precise and is wrong.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from musicintel.eval.manifest import Manifest
from musicintel.eval.negatives import (
    CAT_MUSIC,
    NegativeExcerpt,
    NegativeSet,
    NegativeSource,
    assign_splits,
    base_track_id,
    interleave_calibration,
    norm_text,
    plan_disjoint_excerpts,
    screen_candidates,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _src(tid, dur=120.0, sha=None, artist="A", title="T"):
    return NegativeSource(
        track_id=tid, path=f"data/eval/negatives_corpus/{tid}.mp3",
        sha256=sha or ("%064x" % abs(hash(tid))), duration_sec=dur,
        license="CC-BY", license_url="https://creativecommons.org/licenses/by/4.0/",
        source="archive.org/netlabels", artist=artist, title=title,
    )


# ------------------------------------------------------------- identifiers --
class TestIdentifiers:
    def test_base_track_id_strips_duplicate_family_suffix(self):
        assert base_track_id("ia_adr-002_1") == "ia_adr-002"
        assert base_track_id("ia_adr-002") == "ia_adr-002"
        assert base_track_id("ia_track_2024") == "ia_track"  # any trailing _<digits>

    def test_norm_text_ignores_case_and_punctuation(self):
        assert norm_text("The Quick, Brown Fox!") == norm_text("the quick brown fox")
        assert norm_text(None) == ""


# ---------------------------------------------------------------- screening --
class TestScreening:
    def _screen(self, cands, **kw):
        base = dict(catalog_ids=set(), catalog_sha256=set(),
                    catalog_artist_title=set(), catalog_artists=set(), corpus_ids=set())
        base.update(kw)
        return screen_candidates(cands, **base)

    def test_clean_candidate_survives(self):
        kept, rej = self._screen([_src("ia_new")])
        assert len(kept) == 1 and rej == []

    def test_rejects_sha256_collision(self):
        c = _src("ia_new", sha="a" * 64)
        kept, rej = self._screen([c], catalog_sha256={"a" * 64})
        assert kept == [] and "sha256" in rej[0]["reason"]

    def test_rejects_track_id_already_in_corpus(self):
        kept, rej = self._screen([_src("ia_known")], corpus_ids={"ia_known"})
        assert kept == [] and "already in" in rej[0]["reason"]

    def test_rejects_near_duplicate_id_family(self):
        """`ia_adr-002_1` must not slip past a corpus containing `ia_adr-002`."""
        kept, rej = self._screen([_src("ia_adr-002_1")], corpus_ids={"ia_adr-002"})
        assert kept == [] and "near-duplicate id family" in rej[0]["reason"]

    def test_rejects_artist_title_match(self):
        c = _src("ia_other", artist="Some Artist", title="A Song")
        kept, rej = self._screen([c], catalog_artist_title={("some artist", "a song")})
        assert kept == [] and "artist+title" in rej[0]["reason"]

    def test_rejects_same_artist_as_a_catalog_track(self):
        """Stricter than strictly needed: netlabel artists reuse stems."""
        c = _src("ia_other", artist="Some Artist", title="Different Song")
        kept, rej = self._screen([c], catalog_artists={"some artist"})
        assert kept == [] and "same artist" in rej[0]["reason"]

    def test_rejects_duplicates_among_candidates(self):
        a, b = _src("ia_a", sha="b" * 64), _src("ia_b", sha="b" * 64)
        kept, rej = self._screen([a, b])
        assert len(kept) == 1 and len(rej) == 1

    def test_empty_artist_does_not_match_everything(self):
        c = _src("ia_new", artist=None)
        kept, _ = self._screen([c], catalog_artists={""})
        assert len(kept) == 1


# ----------------------------------------------------------------- planning --
class TestExcerptPlanning:
    def test_excerpts_never_overlap(self):
        es = plan_disjoint_excerpts(_src("t", dur=100.0))
        for a, b in zip(es, es[1:]):
            assert a.start_sec + a.duration <= b.start_sec + 1e-9

    def test_excerpts_stay_inside_the_source(self):
        s = _src("t", dur=60.0)
        for e in plan_disjoint_excerpts(s):
            assert e.start_sec >= 0
            assert e.start_sec + e.duration <= s.duration_sec

    def test_durations_cycle_so_content_is_used_once(self):
        es = plan_disjoint_excerpts(_src("t", dur=200.0))
        assert {e.duration for e in es} == {3.0, 5.0, 10.0}
        total = sum(e.duration for e in es)
        assert total <= 200.0

    def test_is_deterministic(self):
        a = plan_disjoint_excerpts(_src("t", dur=90.0))
        b = plan_disjoint_excerpts(_src("t", dur=90.0))
        assert [x.to_dict() for x in a] == [x.to_dict() for x in b]

    def test_short_source_yields_nothing_rather_than_raising(self):
        assert plan_disjoint_excerpts(_src("t", dur=2.0)) == []

    def test_query_ids_are_unique(self):
        es = plan_disjoint_excerpts(_src("t", dur=300.0))
        assert len({e.query_id for e in es}) == len(es)

    def test_all_music_excerpts_carry_their_source(self):
        for e in plan_disjoint_excerpts(_src("t", dur=60.0)):
            assert e.category == CAT_MUSIC and e.source_track == "t"


# ------------------------------------------------------------------ splits --
class TestSplits:
    def test_interleave_is_deterministic_and_halves(self):
        ids = [f"t{i}" for i in range(10)]
        cal = interleave_calibration(ids)
        assert cal == interleave_calibration(list(reversed(ids)))
        assert len(cal) == 5

    def test_a_source_never_straddles_the_split(self):
        es = plan_disjoint_excerpts(_src("t1", dur=90.0)) + plan_disjoint_excerpts(_src("t2", dur=90.0))
        out = assign_splits(es, calibration_sources={"t1"})
        sides = {}
        for e in out:
            sides.setdefault(e.source_track, set()).add(e.split)
        assert all(len(v) == 1 for v in sides.values())
        assert sides["t1"] == {"calibration"} and sides["t2"] == {"evaluation"}

    def test_synthetics_split_deterministically_by_id(self):
        e = NegativeExcerpt("synth__x__001", "negative_silence", None, 0.0, 3.0)
        a = assign_splits([e], calibration_sources=set())[0].split
        b = assign_splits([e], calibration_sources=set())[0].split
        assert a == b and a in ("calibration", "evaluation")


# --------------------------------------------------------------- the set ----
class TestNegativeSet:
    def _set(self):
        s1, s2 = _src("t1", dur=90.0), _src("t2", dur=90.0)
        es = plan_disjoint_excerpts(s1) + plan_disjoint_excerpts(s2)
        return NegativeSet(sources=[s1, s2],
                           excerpts=assign_splits(es, calibration_sources={"t1"}))

    def test_verify_accepts_a_sound_set(self):
        assert self._set().verify(catalog_ids={"catalog-a"}) == []

    def test_verify_rejects_a_catalog_track_as_a_source(self):
        ns = self._set()
        assert any("INDEXED catalog" in p for p in ns.verify(catalog_ids={"t1"}))

    def test_verify_rejects_a_straddling_source(self):
        ns = self._set()
        ns.excerpts[0] = NegativeExcerpt(**{**ns.excerpts[0].to_dict(), "split": "evaluation"})
        ns.excerpts[1] = NegativeExcerpt(**{**ns.excerpts[1].to_dict(), "split": "calibration"})
        assert any("both splits" in p for p in ns.verify(set()))

    def test_verify_rejects_overlapping_excerpts(self):
        ns = self._set()
        bad = NegativeExcerpt("x", CAT_MUSIC, "t1", 0.0, 60.0, "calibration")
        ns.excerpts.append(bad)
        assert any("overlapping" in p for p in ns.verify(set()))

    def test_verify_rejects_duplicate_query_ids(self):
        ns = self._set()
        ns.excerpts.append(ns.excerpts[0])
        assert any("duplicate query_id" in p for p in ns.verify(set()))

    def test_content_hash_is_stable_and_sensitive(self):
        a, b = self._set(), self._set()
        assert a.content_hash() == b.content_hash()
        b.excerpts.pop()
        assert a.content_hash() != b.content_hash()

    def test_counts_and_source_counts(self):
        ns = self._set()
        assert sum(ns.counts_by_split().values()) == len(ns.excerpts)
        assert ns.source_counts() == {"calibration": 1, "evaluation": 1}

    def test_save_load_roundtrip(self, tmp_path):
        ns = self._set()
        ns.save(tmp_path / "n.json")
        back = NegativeSet.load(tmp_path / "n.json")
        assert back.content_hash() == ns.content_hash()
        assert len(back.sources) == len(ns.sources)


# ------------------------------------------- the committed negative manifest -
class TestCommittedNegativeSet:
    @pytest.fixture
    def ns(self):
        p = REPO_ROOT / "eval/fixtures/negatives_manifest.json"
        if not p.is_file():
            pytest.skip("negative set not built in this checkout")
        return NegativeSet.load(p)

    @pytest.fixture
    def catalog_ids(self):
        m = Manifest.load(REPO_ROOT / "eval/fixtures/manifest.json")
        m.assign_holdout(12)
        return {t.track_id for t in m.catalog}

    def test_passes_its_own_verification(self, ns, catalog_ids):
        assert ns.verify(catalog_ids) == []

    def test_reaches_the_phase_1g_size_target(self, ns):
        assert len(ns.excerpts) >= 1000
        assert ns.counts_by_split().get("evaluation", 0) >= 500

    def test_no_indexed_catalog_track_is_a_negative_source(self, ns, catalog_ids):
        assert not {s.track_id for s in ns.sources} & catalog_ids

    def test_every_source_records_licence_and_hash(self, ns):
        for s in ns.sources:
            assert s.license and s.license_url and s.source
            assert len(s.sha256) == 64

    def test_licences_are_permissive(self, ns):
        for s in ns.sources:
            u = s.license_url.lower()
            assert "-nd" not in u and "-nc" not in u and "nc-" not in u

    def test_music_dominates_so_the_set_is_not_diluted(self, ns):
        """Padding with trivially-rejected silence would flatter the aggregate FAR."""
        c = ns.counts_by_category()
        assert c.get(CAT_MUSIC, 0) > 0.5 * len(ns.excerpts)

    def test_reports_source_recording_counts_per_split(self, ns):
        sc = ns.source_counts()
        assert sc.get("calibration", 0) >= 2 and sc.get("evaluation", 0) >= 2
