#!/usr/bin/env python3
"""Validate the 20-track pilot annotation sheet.

The sheet is filled in by a human outside this environment, so it is validated
rather than trusted: labels can be mistyped, rows reordered by a spreadsheet, a
row half-completed, or a `key` spelled flat when the metric expects sharps.

Partial completion is legitimate. A `pending` row with empty labels is not an
error -- it is the normal state of a sheet that is still being worked through.
What is an error is a row that claims to be `annotated` without the evidence to
back it, or a label that would silently corrupt the ground truth.

    python scripts/validate_pilot_annotations.py
    python scripts/validate_pilot_annotations.py --csv other.csv --quiet

Exit code 0 = valid (annotated rows may be 0..20), 1 = errors found.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from musicintel.analysis.evaluation import (                     # noqa: E402
    EXCLUDED, MANIFEST_STATUSES, NO_KEY, NO_TEMPO, PENDING, PILOT_STATUSES,
)
from musicintel.analysis.keys import KeyParseError, normalize_key  # noqa: E402

DEFAULT_CSV = REPO / "eval/fixtures/bpm_key_pilot_annotation.csv"
DEFAULT_MANIFEST = REPO / "eval/fixtures/bpm_key_annotation_manifest.json"

PILOT_SIZE = 20
COLUMNS = ["track_id", "path", "title", "artist", "duration_sec", "license",
           "license_url", "bpm", "key", "annotation_status", "annotator",
           "annotated_utc", "notes"]
METADATA_COLUMNS = ["path", "title", "artist", "license", "license_url"]
# The shared vocabulary from musicintel.analysis.evaluation, not a local copy:
# the evaluation harness selects rows by these same constants. A sheet ROW is
# pending or annotated; only a manifest TRACK can be excluded (whole recording).
STATUSES = PILOT_STATUSES

# Generous, but not so generous that a typo slips through. Below 20 or above 300
# is far likelier to be a mistake than a real global tempo.
BPM_MIN, BPM_MAX = 20.0, 300.0
UTC_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}(:\d{2}(\.\d+)?)?Z$")


def parse_rows(text: str) -> list[dict[str, str]]:
    """Parse the sheet's text, skipping the `#` instruction block and blank lines.

    Tolerates a spreadsheet round-trip, which may quote the comment lines and
    pad them with trailing commas.
    """
    lines = []
    for raw in text.splitlines():
        stripped = raw.lstrip().lstrip('"').lstrip()
        if stripped.startswith("#") or not raw.strip().strip(","):
            continue
        lines.append(raw)
    return list(csv.DictReader(lines))


def read_rows(path: Path) -> list[dict[str, str]]:
    return parse_rows(Path(path).read_text())


def pilot_of(tracks: list[dict]) -> list[dict]:
    return sorted(tracks, key=lambda t: t["track_id"])[:PILOT_SIZE]


def expected_pilot(manifest: Path) -> list[dict]:
    return pilot_of(json.loads(manifest.read_text())["tracks"])


def manifest_status_errors(tracks: list[dict]) -> list[str]:
    """Every manifest track's status must be in MANIFEST_STATUSES, and a
    whole-recording exclusion must say why -- an exclusion is a recorded
    decision, never a silent withdrawal."""
    errors = []
    for t in tracks:
        tid, status = t.get("track_id"), t.get("annotation_status")
        if status not in MANIFEST_STATUSES:
            errors.append(f"manifest track {tid!r}: annotation_status {status!r} "
                          f"must be one of {sorted(MANIFEST_STATUSES)}")
        elif status == EXCLUDED and not str(t.get("notes") or "").strip():
            errors.append(f"manifest track {tid!r}: '{EXCLUDED}' requires the "
                          "reason in notes")
    return errors


def _check_bpm(value: str, where: str, errors: list[str]) -> None:
    if value == NO_TEMPO:
        return
    try:
        bpm = float(value)
    except ValueError:
        errors.append(f"{where}: bpm {value!r} is neither a number nor '{NO_TEMPO}'")
        return
    if not (BPM_MIN <= bpm <= BPM_MAX):
        errors.append(f"{where}: bpm {bpm} outside plausible range "
                      f"{BPM_MIN}-{BPM_MAX}; if it is real, say so in notes")


def _check_key(value: str, where: str, errors: list[str]) -> None:
    if value == NO_KEY:
        return
    try:
        canonical = normalize_key(value)
    except KeyParseError:
        errors.append(f"{where}: key {value!r} is not one of the 24 classes "
                      f"nor '{NO_KEY}'")
        return
    if canonical != value:
        errors.append(f"{where}: key {value!r} should be written {canonical!r} "
                      f"(sharp-normalised canonical spelling)")


def validate(csv_path: Path = DEFAULT_CSV,
             manifest_path: Path = DEFAULT_MANIFEST) -> tuple[list[str], dict]:
    """Return (errors, summary). An empty error list means the sheet is usable."""
    if not csv_path.exists():
        return [f"missing annotation sheet: {csv_path}"], {}
    return validate_parsed(read_rows(csv_path),
                           json.loads(Path(manifest_path).read_text())["tracks"])


def validate_parsed(rows: list[dict[str, str]],
                    tracks: list[dict]) -> tuple[list[str], dict]:
    """`validate` on already-parsed sheet rows and manifest tracks, so a caller
    that has read and hashed the files validates exactly what it read."""
    errors: list[str] = []
    expected = pilot_of(tracks)
    by_id = {t["track_id"]: t for t in expected}
    errors.extend(manifest_status_errors(tracks))

    header = list(rows[0]) if rows else []
    if header != COLUMNS:
        errors.append(f"columns are {header}, expected {COLUMNS}")
        return errors, {}

    # -- row set: exactly the 20 pilot tracks, each once ----------------------
    seen: dict[str, int] = {}
    for i, row in enumerate(rows, start=1):
        tid = row["track_id"].strip()
        if tid in seen:
            errors.append(f"row {i}: duplicate track_id {tid!r} "
                          f"(first seen at row {seen[tid]})")
        seen[tid] = i
        if tid not in by_id:
            errors.append(f"row {i}: {tid!r} is not one of the 20 pilot tracks")

    for tid in by_id:
        if tid not in seen:
            errors.append(f"missing pilot track: {tid}")

    if len(rows) != PILOT_SIZE:
        errors.append(f"expected exactly {PILOT_SIZE} rows, found {len(rows)}")

    annotated = pending = no_tempo = no_key = 0

    for i, row in enumerate(rows, start=1):
        tid = row["track_id"].strip()
        where = f"row {i} ({tid})"
        truth = by_id.get(tid)

        # -- metadata must still describe the same audio ----------------------
        if truth:
            for col in METADATA_COLUMNS:
                if row[col].strip() != str(truth[col]):
                    errors.append(f"{where}: {col} was edited; expected "
                                  f"{truth[col]!r}, found {row[col].strip()!r}")
            try:
                duration = float(row["duration_sec"] or 0)
            except ValueError:
                duration = None
            if duration is None or abs(duration - truth["duration_sec"]) > 0.05:
                errors.append(f"{where}: duration_sec was edited")

        status = row["annotation_status"].strip()
        bpm = row["bpm"].strip()
        key = row["key"].strip()
        annotator = row["annotator"].strip()
        when = row["annotated_utc"].strip()

        if status not in STATUSES:
            errors.append(f"{where}: annotation_status {status!r} "
                          f"must be one of {sorted(STATUSES)}")
            continue

        if status == PENDING:
            pending += 1
            # Empty is the normal pending state. Labels without the status are
            # not -- that is work that would be silently dropped on import.
            for col, val in (("bpm", bpm), ("key", key),
                             ("annotator", annotator), ("annotated_utc", when)):
                if val:
                    errors.append(f"{where}: {col}={val!r} on a 'pending' row; "
                                  f"set annotation_status to 'annotated'")
            continue

        annotated += 1
        for col, val in (("bpm", bpm), ("key", key),
                         ("annotator", annotator), ("annotated_utc", when)):
            if not val:
                errors.append(f"{where}: {col} is required once "
                              f"annotation_status is 'annotated'")
        if bpm:
            _check_bpm(bpm, where, errors)
            no_tempo += bpm == NO_TEMPO
        if key:
            _check_key(key, where, errors)
            no_key += key == NO_KEY
        if when and not UTC_RE.match(when):
            errors.append(f"{where}: annotated_utc {when!r} is not UTC ISO-8601 "
                          f"(e.g. 2026-08-31T14:05:00Z)")
        # notes stay optional, by design

    annotators = sorted({r["annotator"].strip() for r in rows
                         if r["annotator"].strip()})
    summary = {
        "rows": len(rows), "annotated": annotated, "pending": pending,
        "no_stable_tempo": no_tempo, "no_tonal_centre": no_key,
        "manifest_excluded": sorted(tid for tid, t in by_id.items()
                                    if t.get("annotation_status") == EXCLUDED),
        "annotators": annotators, "errors": len(errors),
    }
    return errors, summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    try:
        errors, summary = validate(args.csv, args.manifest)
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        # An input that cannot be read or parsed is reported, not a traceback.
        errors, summary = [f"cannot read the sheet or manifest: "
                           f"{type(exc).__name__}: {exc}"], {}

    if not args.quiet:
        print(f"Pilot annotation sheet: {args.csv}")
        if summary:
            print(f"  rows            : {summary['rows']}")
            print(f"  annotated       : {summary['annotated']}")
            print(f"  pending         : {summary['pending']}")
            print(f"  no_stable_tempo : {summary['no_stable_tempo']}")
            print(f"  no_tonal_centre : {summary['no_tonal_centre']}")
            print(f"  manifest-excluded recordings : "
                  f"{', '.join(summary['manifest_excluded']) or '(none)'}")
            print(f"  annotators      : {', '.join(summary['annotators']) or '(none yet)'}")
        print()
        if errors:
            print(f"  {len(errors)} problem(s):")
            for e in errors:
                print(f"    - {e}")
        elif summary.get("annotated", 0) == 0:
            print("  Valid, and entirely unannotated -- ready for a human to fill in.")
        else:
            print(f"  Valid. {summary['annotated']}/{PILOT_SIZE} annotated, "
                  f"{summary['pending']} still pending.")

    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
