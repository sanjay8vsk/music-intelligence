"""The 20-track pilot annotation sheet and its validator.

The sheet is filled in by a human outside this environment. Every test that
matters here is therefore a rejection test: the validator's job is to catch a
mistyped key, a half-completed row or a spreadsheet that reordered the file,
before any of it reaches the ground truth.
"""

from __future__ import annotations

import csv
import importlib.util
import io
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
CSV_PATH = REPO / "eval/fixtures/bpm_key_pilot_annotation.csv"
MANIFEST = REPO / "eval/fixtures/bpm_key_annotation_manifest.json"

spec = importlib.util.spec_from_file_location(
    "validate_pilot_annotations", REPO / "scripts/validate_pilot_annotations.py")
vpa = importlib.util.module_from_spec(spec)
sys.modules["validate_pilot_annotations"] = vpa
spec.loader.exec_module(vpa)


def _instructions() -> list[str]:
    return [ln for ln in CSV_PATH.read_text().splitlines() if ln.startswith("#")]


def _rows() -> list[dict[str, str]]:
    return vpa.read_rows(CSV_PATH)


def _write(tmp_path: Path, rows: list[dict[str, str]], name="sheet.csv") -> Path:
    out = tmp_path / name
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=vpa.COLUMNS, lineterminator="\n")
    w.writeheader()
    w.writerows(rows)
    out.write_text("\n".join(_instructions()) + "\n" + buf.getvalue())
    return out


def _errors(tmp_path: Path, mutate) -> list[str]:
    rows = _rows()
    mutate(rows)
    errs, _ = vpa.validate(_write(tmp_path, rows), MANIFEST)
    return errs


def _annotate(row, *, bpm="120", key="C major", annotator="tester",
              when="2026-08-31T14:05:00Z", notes=""):
    row.update({"bpm": bpm, "key": key, "annotation_status": "annotated",
                "annotator": annotator, "annotated_utc": when, "notes": notes})


# ------------------------------------------------------------------ sheet --
class TestSheetAsShipped:
    def test_it_has_exactly_the_20_pilot_rows(self):
        rows = _rows()
        assert len(rows) == 20
        manifest = json.loads(MANIFEST.read_text())["tracks"]
        expected = [t["track_id"] for t in
                    sorted(manifest, key=lambda t: t["track_id"])[:20]]
        assert [r["track_id"] for r in rows] == expected

    def test_columns_are_exactly_as_specified(self):
        assert list(_rows()[0]) == [
            "track_id", "path", "title", "artist", "duration_sec", "license",
            "license_url", "bpm", "key", "annotation_status", "annotator",
            "annotated_utc", "notes"]

    def test_every_label_field_is_empty_and_pending(self):
        for r in _rows():
            assert r["bpm"] == "" and r["key"] == ""
            assert r["annotator"] == "" and r["annotated_utc"] == ""
            assert r["annotation_status"] == "pending"

    def test_it_matches_the_manifest_metadata_exactly(self):
        by_id = {t["track_id"]: t
                 for t in json.loads(MANIFEST.read_text())["tracks"]}
        for r in _rows():
            t = by_id[r["track_id"]]
            for col in ("path", "title", "artist", "license", "license_url"):
                assert r[col] == str(t[col]), (r["track_id"], col)
            assert float(r["duration_sec"]) == pytest.approx(t["duration_sec"], abs=0.05)

    def test_every_referenced_audio_file_exists(self):
        """The audio lives under data/, which is gitignored, so a bare checkout
        cannot have it. Skip there rather than fail; where the corpus IS built,
        a row pointing at a file that is not on disk is a real error and fails.
        Same convention as tests/test_scale_corpus.py.
        """
        rows = _rows()
        roots = {(REPO / r["path"]).parent for r in rows}
        if not any(root.is_dir() for root in roots):
            pytest.skip("audio corpus not built in this checkout (data/ is gitignored)")
        for r in rows:
            assert (REPO / r["path"]).exists(), r["path"]

    def test_the_shipped_sheet_validates_clean(self):
        errors, summary = vpa.validate(CSV_PATH, MANIFEST)
        assert errors == []
        assert summary["annotated"] == 0 and summary["pending"] == 20

    def test_instructions_state_the_non_negotiables(self):
        text = "\n".join(_instructions())
        assert "no_stable_tempo" in text and "no_tonal_centre" in text
        assert "C# minor" in text                      # sharp normalisation
        assert "LISTEN" in text.upper()
        for banned in ("librosa", "Essentia", "MusicBrainz", "AcousticBrainz"):
            assert banned in text
        assert "C major" in text and "A minor" in text  # the 24 classes listed

    def test_instructions_survive_a_parse(self):
        """The comment block must not become a data row."""
        assert all(not r["track_id"].startswith("#") for r in _rows())


# -------------------------------------------------------------- rejection --
class TestValidatorRejects:
    def test_a_duplicate_track_id(self, tmp_path):
        errs = _errors(tmp_path, lambda rows: rows.__setitem__(
            1, dict(rows[0])))
        assert any("duplicate" in e for e in errs)

    def test_a_track_outside_the_pilot(self, tmp_path):
        def mutate(rows):
            rows[0] = dict(rows[0], track_id="ia_Ethnotronic-Tronbotik")
        errs = _errors(tmp_path, mutate)
        assert any("not one of the 20 pilot tracks" in e for e in errs)
        assert any("missing pilot track" in e for e in errs)

    def test_a_deleted_row(self, tmp_path):
        errs = _errors(tmp_path, lambda rows: rows.pop())
        assert any("expected exactly 20 rows" in e for e in errs)
        assert any("missing pilot track" in e for e in errs)

    def test_an_extra_row(self, tmp_path):
        errs = _errors(tmp_path, lambda rows: rows.append(dict(rows[0])))
        assert any("expected exactly 20 rows" in e for e in errs)

    @pytest.mark.parametrize("bad", ["fast", "120bpm", "", "0", "5", "900"])
    def test_a_bpm_that_is_neither_a_number_nor_the_exclusion(self, tmp_path, bad):
        errs = _errors(tmp_path, lambda rows: _annotate(rows[0], bpm=bad))
        assert any("bpm" in e for e in errs), (bad, errs)

    @pytest.mark.parametrize("bad", ["H major", "C", "c# minor", "Cmaj",
                                     "no_key", "atonal", ""])
    def test_a_key_that_is_not_one_of_the_24_classes(self, tmp_path, bad):
        errs = _errors(tmp_path, lambda rows: _annotate(rows[0], key=bad))
        assert any("key" in e for e in errs), (bad, errs)

    def test_a_flat_spelling_is_corrected_not_silently_accepted(self, tmp_path):
        errs = _errors(tmp_path, lambda rows: _annotate(rows[0], key="Db minor"))
        assert any("C# minor" in e for e in errs)

    def test_an_annotated_row_missing_the_annotator(self, tmp_path):
        errs = _errors(tmp_path, lambda rows: _annotate(rows[0], annotator=""))
        assert any("annotator is required" in e for e in errs)

    def test_an_annotated_row_missing_the_timestamp(self, tmp_path):
        errs = _errors(tmp_path, lambda rows: _annotate(rows[0], when=""))
        assert any("annotated_utc is required" in e for e in errs)

    @pytest.mark.parametrize("bad", ["31/08/2026", "2026-08-31",
                                     "2026-08-31 14:05", "yesterday"])
    def test_a_timestamp_that_is_not_utc_iso8601(self, tmp_path, bad):
        errs = _errors(tmp_path, lambda rows: _annotate(rows[0], when=bad))
        assert any("ISO-8601" in e for e in errs)

    def test_labels_left_on_a_pending_row(self, tmp_path):
        """Work that would be silently dropped on import."""
        def mutate(rows):
            rows[0] = dict(rows[0], bpm="128", key="A minor")
        errs = _errors(tmp_path, mutate)
        assert any("'pending' row" in e for e in errs)

    def test_an_unknown_status(self, tmp_path):
        def mutate(rows):
            rows[0] = dict(rows[0], annotation_status="done")
        errs = _errors(tmp_path, mutate)
        assert any("annotation_status" in e for e in errs)

    def test_edited_metadata(self, tmp_path):
        def mutate(rows):
            rows[0] = dict(rows[0], title="something else")
        errs = _errors(tmp_path, mutate)
        assert any("title was edited" in e for e in errs)

    def test_a_repointed_path(self, tmp_path):
        """The label must stay attached to the audio it was made from."""
        def mutate(rows):
            rows[0] = dict(rows[0], path=rows[1]["path"])
        errs = _errors(tmp_path, mutate)
        assert any("path was edited" in e for e in errs)

    def test_a_missing_sheet(self, tmp_path):
        errs, _ = vpa.validate(tmp_path / "nope.csv", MANIFEST)
        assert any("missing annotation sheet" in e for e in errs)


# ---------------------------------------------------------------- accepts --
class TestValidatorAccepts:
    def test_a_fully_valid_annotation(self, tmp_path):
        assert _errors(tmp_path, lambda rows: _annotate(
            rows[0], bpm="143.2", key="F# minor", notes="drifts late")) == []

    def test_partial_completion(self, tmp_path):
        """The sheet is filled in over time; half-done must not be an error."""
        def mutate(rows):
            for r in rows[:7]:
                _annotate(r)
        rows = _rows(); mutate(rows)
        errs, summary = vpa.validate(_write(tmp_path, rows), MANIFEST)
        assert errs == []
        assert summary["annotated"] == 7 and summary["pending"] == 13

    def test_the_two_exclusion_categories(self, tmp_path):
        def mutate(rows):
            _annotate(rows[0], bpm="no_stable_tempo", key="C major",
                      notes="free-time ambient")
            _annotate(rows[1], bpm="96", key="no_tonal_centre",
                      notes="percussion only")
        rows = _rows(); mutate(rows)
        errs, summary = vpa.validate(_write(tmp_path, rows), MANIFEST)
        assert errs == []
        assert summary["no_stable_tempo"] == 1 and summary["no_tonal_centre"] == 1

    def test_both_exclusions_on_one_track(self, tmp_path):
        assert _errors(tmp_path, lambda rows: _annotate(
            rows[0], bpm="no_stable_tempo", key="no_tonal_centre",
            notes="spoken word")) == []

    def test_notes_stay_optional(self, tmp_path):
        assert _errors(tmp_path, lambda rows: _annotate(rows[0], notes="")) == []

    def test_a_non_integer_tempo(self, tmp_path):
        """Do not round to a nicer number -- 143.2 must be accepted verbatim."""
        assert _errors(tmp_path, lambda rows: _annotate(rows[0], bpm="143.2")) == []

    def test_all_24_key_classes_are_accepted(self, tmp_path):
        from musicintel.analysis.keys import KEY_LABELS
        for label in KEY_LABELS:
            assert _errors(tmp_path, lambda rows, k=label:
                           _annotate(rows[0], key=k)) == [], label

    def test_a_spreadsheet_round_trip_of_the_comment_block(self, tmp_path):
        """Excel may quote the instruction lines and pad them with commas."""
        rows = _rows()
        out = tmp_path / "excel.csv"
        buf = io.StringIO()
        w = csv.DictWriter(buf, fieldnames=vpa.COLUMNS, lineterminator="\n")
        w.writeheader(); w.writerows(rows)
        mangled = "\n".join(f'"{ln}",,,,,,,,,,,,' for ln in _instructions())
        out.write_text(mangled + "\n" + buf.getvalue())
        errs, summary = vpa.validate(out, MANIFEST)
        assert errs == [] and summary["rows"] == 20

    def test_trailing_blank_lines(self, tmp_path):
        rows = _rows()
        p = _write(tmp_path, rows)
        p.write_text(p.read_text() + "\n,,,,,,,,,,,,\n\n")
        errs, summary = vpa.validate(p, MANIFEST)
        assert errs == [] and summary["rows"] == 20


class TestCli:
    def test_it_exits_zero_on_the_shipped_sheet(self, capsys):
        assert vpa.main(["--csv", str(CSV_PATH)]) == 0
        assert "ready for a human to fill in" in capsys.readouterr().out

    def test_it_exits_nonzero_and_names_the_problem(self, tmp_path, capsys):
        rows = _rows()
        _annotate(rows[0], key="H major")
        assert vpa.main(["--csv", str(_write(tmp_path, rows))]) == 1
        assert "H major" in capsys.readouterr().out

    def test_quiet_prints_nothing(self, tmp_path, capsys):
        assert vpa.main(["--csv", str(CSV_PATH), "--quiet"]) == 0
        assert capsys.readouterr().out == ""


class TestManifestUntouched:
    def test_the_json_manifest_has_no_labels_written_to_it(self):
        d = json.loads(MANIFEST.read_text())
        assert d["annotated_count"] == 0
        assert all(t["bpm"] is None and t["key"] is None for t in d["tracks"])
        assert all(t["annotation_status"] == "pending" for t in d["tracks"])

    def test_the_manifest_content_hash_is_unchanged(self):
        d = json.loads(MANIFEST.read_text())
        assert d["content_hash"] == (
            "6d2e9276e580f2698a4d76de46458775316840bcda4c62dfebe0e51235509d29")

    def test_no_detector_has_been_implemented(self):
        names = {f.name for f in (REPO / "musicintel/analysis").glob("*.py")}
        assert names == {"__init__.py", "features.py", "keys.py",
                         "fixtures.py", "evaluation.py"}, names
