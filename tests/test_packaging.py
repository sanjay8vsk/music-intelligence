"""What actually ships in the wheel.

`musicintel/db/migrate.py` resolves its migrations relative to the installed
package. A source checkout always has them, so nothing in the rest of the suite
notices when the built distribution does not -- and the failure is silent:
`_migration_files` globs a directory that is not there, finds nothing, and
`apply_migrations` reports applying no migrations rather than raising. A
container built from the Dockerfile would come up against an unmigrated
database and say it was fine.

These tests therefore inspect a real wheel, not the working tree.
"""

from __future__ import annotations

import glob
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
MIGRATIONS = REPO / "musicintel/db/migrations"


def _build_wheel(into: Path) -> Path:
    """Build a real wheel, or skip if this environment cannot build one.

    pip's build isolation fetches its own setuptools, so an offline machine
    cannot run this. Skipping is honest -- packaging is then unverified -- and is
    not the same as passing; the declaration test below still runs.
    """
    proc = subprocess.run(
        [sys.executable, "-m", "pip", "wheel", "--no-deps", "-w", str(into), str(REPO)],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        pytest.skip("cannot build a wheel here (build deps unavailable or "
                    f"offline); packaging is UNVERIFIED: {proc.stderr[-200:]}")
    built = glob.glob(str(into / "*.whl"))
    assert built, "pip reported success but produced no wheel"
    return Path(built[0])


class TestMigrationsAreDistributed:
    def test_the_wheel_carries_every_migration(self, tmp_path):
        wheel = _build_wheel(tmp_path)
        shipped = {Path(n).name for n in zipfile.ZipFile(wheel).namelist()
                   if n.startswith("musicintel/db/migrations/") and n.endswith(".sql")}
        on_disk = {p.name for p in MIGRATIONS.glob("*.sql")}
        assert on_disk, "no migrations in the source tree -- the test is misdirected"
        assert shipped == on_disk, (
            f"wheel ships {sorted(shipped)}, source has {sorted(on_disk)}")

    def test_an_installed_package_discovers_them(self, tmp_path):
        """The end the bug actually bit: resolution from an installed package.

        Run against the extracted wheel only, with the editable finder stripped,
        so the development checkout cannot satisfy the import.
        """
        wheel = _build_wheel(tmp_path)
        site = tmp_path / "site"
        zipfile.ZipFile(wheel).extractall(site)

        probe = (
            "import json, os, sys\n"
            "sys.meta_path = [f for f in sys.meta_path if '__editable__' not in "
            "getattr(f, '__module__', type(f).__module__)]\n"
            f"sys.path = [p for p in sys.path if p and "
            f"os.path.realpath(p) != {str(REPO)!r}]\n"
            f"sys.path.insert(0, {str(site)!r})\n"
            "from musicintel.db.migrate import MIGRATIONS_DIR, _migration_files\n"
            "import musicintel.db.migrate as m\n"
            "print(json.dumps({'module': m.__file__,\n"
            "                  'dir': str(MIGRATIONS_DIR),\n"
            "                  'found': sorted(p.name for p in _migration_files())}))"
        )
        proc = subprocess.run([sys.executable, "-c", probe],
                              capture_output=True, text=True, cwd=tmp_path)
        assert proc.returncode == 0, proc.stderr[-2000:]
        result = json.loads(proc.stdout.strip().splitlines()[-1])

        assert result["module"].startswith(str(site)), (
            f"import leaked outside the extracted wheel: {result['module']}")
        assert result["dir"].startswith(str(site))
        assert result["found"] == sorted(p.name for p in MIGRATIONS.glob("*.sql"))


class TestPackageDataDeclaration:
    """Offline guard. Weaker than the wheel tests, and not a substitute for them:
    it checks the declaration rather than the artefact, so it runs everywhere.
    """

    def test_every_migration_is_covered_by_the_declaration(self):
        import tomllib

        with open(REPO / "pyproject.toml", "rb") as fh:
            cfg = tomllib.load(fh)
        patterns = cfg["tool"]["setuptools"]["package-data"]["musicintel.db"]
        assert any(p.startswith("migrations/") and p.endswith(".sql")
                   for p in patterns), patterns
        # A migration added with any other suffix would ship silently missing.
        for path in MIGRATIONS.iterdir():
            if path.is_file():
                assert path.suffix == ".sql", (
                    f"{path.name} is not matched by {patterns}")
