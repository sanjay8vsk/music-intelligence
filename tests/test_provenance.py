"""Tests for benchmark provenance -- specifically, that a report can identify
the code that produced it.

The Phase 1D audit found a report whose only source fingerprint covered the
evaluation harness, leaving the recognizer that actually produced the numbers
unidentified. These tests exist so that gap cannot reopen silently.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from musicintel.eval.provenance import (
    ALGORITHM_SOURCES,
    HARNESS_SOURCES,
    PHASE1_SOURCES,
    git_state,
    source_fingerprint,
    version_string,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


# ------------------------------------------------------- the Phase 1 manifest -
class TestPhase1Sources:
    def test_covers_the_whole_recognition_pipeline(self):
        """Every stage that can move a number must be pinned."""
        required = {
            "musicintel/recognition/fingerprint.py",
            "musicintel/recognition/index.py",
            "musicintel/recognition/matcher.py",
            "musicintel/recognition/decision.py",
        }
        assert required <= set(PHASE1_SOURCES)

    def test_every_listed_file_exists(self):
        for rel in PHASE1_SOURCES:
            assert (REPO_ROOT / rel).is_file(), rel

    def test_does_not_overlap_the_other_manifests(self):
        """Harness, algorithm and pipeline are three separable subsystems; a
        shared file would make one fingerprint move for the wrong reason."""
        p1 = set(PHASE1_SOURCES)
        assert not p1 & set(HARNESS_SOURCES)
        assert not p1 & set(ALGORITHM_SOURCES)

    def test_pipeline_fingerprint_differs_from_the_harness_fingerprint(self):
        """The exact confusion the Phase 1D audit caught: a report that quotes
        the harness fingerprint is not identifying the recognizer."""
        assert source_fingerprint(REPO_ROOT, PHASE1_SOURCES) != source_fingerprint(
            REPO_ROOT, HARNESS_SOURCES
        )


# ------------------------------------------------------------- fingerprinting -
class TestSourceFingerprint:
    def test_is_deterministic(self):
        a = source_fingerprint(REPO_ROOT, PHASE1_SOURCES)
        b = source_fingerprint(REPO_ROOT, PHASE1_SOURCES)
        assert a == b and len(a) == 64

    def test_is_order_independent(self):
        assert source_fingerprint(REPO_ROOT, PHASE1_SOURCES) == source_fingerprint(
            REPO_ROOT, tuple(reversed(PHASE1_SOURCES))
        )

    def test_changes_when_a_covered_file_changes(self, tmp_path):
        (tmp_path / "a.py").write_text("x = 1\n")
        (tmp_path / "b.py").write_text("y = 2\n")
        before = source_fingerprint(tmp_path, ("a.py", "b.py"))
        (tmp_path / "b.py").write_text("y = 3\n")
        assert source_fingerprint(tmp_path, ("a.py", "b.py")) != before

    def test_changes_when_a_covered_file_disappears(self, tmp_path):
        (tmp_path / "a.py").write_text("x = 1\n")
        (tmp_path / "b.py").write_text("y = 2\n")
        before = source_fingerprint(tmp_path, ("a.py", "b.py"))
        (tmp_path / "b.py").unlink()
        # A deleted file must move the fingerprint, not be quietly skipped.
        assert source_fingerprint(tmp_path, ("a.py", "b.py")) != before

    def test_path_matters_not_just_content(self, tmp_path):
        (tmp_path / "a.py").write_text("same\n")
        (tmp_path / "b.py").write_text("same\n")
        assert source_fingerprint(tmp_path, ("a.py",)) != source_fingerprint(
            tmp_path, ("b.py",)
        )

    def test_including_the_driver_changes_the_fingerprint(self):
        """A benchmark driver fingerprints PHASE1_SOURCES plus its own path;
        that must not collapse to the library-only value."""
        driver = "scripts/eval_phase1e.py"
        assert (REPO_ROOT / driver).is_file()
        assert source_fingerprint(
            REPO_ROOT, tuple(PHASE1_SOURCES) + (driver,)
        ) != source_fingerprint(REPO_ROOT, PHASE1_SOURCES)


# --------------------------------------------------------------- git state ----
class TestGitState:
    def test_reports_the_expected_shape(self):
        g = git_state(REPO_ROOT)
        assert set(g) == {"commit", "commit_short", "dirty", "dirty_paths"}
        assert isinstance(g["dirty"], bool)
        assert isinstance(g["dirty_paths"], list)
        assert g["commit_short"] == g["commit"][:7]

    def test_dirty_flag_agrees_with_the_path_list(self):
        g = git_state(REPO_ROOT)
        assert g["dirty"] == bool(g["dirty_paths"])

    def test_non_repository_degrades_without_raising(self, tmp_path):
        """Provenance must never be the thing that breaks a benchmark run."""
        g = git_state(tmp_path)
        assert g["commit"] == "unknown" or isinstance(g["commit"], str)

    def test_version_string_marks_a_dirty_tree(self, tmp_path):
        v = version_string("landmark", REPO_ROOT)
        assert v.startswith("landmark@")
        assert v.endswith("+dirty") == git_state(REPO_ROOT)["dirty"]


# ------------------------------------------------- the report actually uses it -
class TestPhase1EReportProvenance:
    """The report is a committed artifact; these assert it carries what it must."""

    @pytest.fixture
    def report(self):
        import json

        p = REPO_ROOT / "eval/reports/phase1e_benchmark.json"
        if not p.is_file():
            pytest.skip("Phase 1E report not generated in this checkout")
        return json.loads(p.read_text())

    def test_records_the_phase1_fingerprint_and_its_inputs(self, report):
        pv = report["provenance"]
        assert len(pv["phase1_source_sha256"]) == 64
        assert set(PHASE1_SOURCES) <= set(pv["phase1_sources"])

    def test_records_commit_and_dirty_state(self, report):
        pv = report["provenance"]
        assert len(pv["git_commit"]) == 40
        assert isinstance(pv["git_dirty"], bool)

    def test_records_versions_hashes_and_counts(self, report):
        pv, ds = report["provenance"], report["dataset"]
        assert pv["fingerprint_format_version"] >= 1
        assert pv["index_format_version"] >= 1
        assert pv["recognizer_version"].startswith("landmark@")
        assert len(ds["manifest_hash"]) == 64 and len(ds["split_hash"]) == 64
        assert ds["catalog_count"] > 0 and ds["queries"] > 0

    def test_records_the_decision_configuration(self, report):
        d = report["configuration"]["decision"]
        assert d["score_is_probability"] is False
        assert 0.0 < d["threshold"] < 1.0
        assert d["min_aligned_landmarks"] >= 1

    def test_does_not_claim_a_probability_anywhere(self, report):
        import json

        blob = json.dumps(report).lower()
        assert "confidence" not in blob and "probability" in blob  # only as a denial
        assert '"score_is_probability": false' in json.dumps(report).lower()
