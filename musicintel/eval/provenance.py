"""Provenance for benchmark reports: which code actually produced a result.

A commit hash alone is not enough to make a report citable. If the harness is
uncommitted, or the tree is dirty, the recorded commit names a state that does
NOT contain the code that ran -- which is exactly how the first Phase 0 baseline
came to cite a commit containing none of the evaluation harness.

So a report records three things instead of one:

  * the commit HEAD pointed at,
  * whether the tree was dirty (and which paths were dirty), and
  * a content fingerprint of the source files that actually executed.

The fingerprint is the part that survives a dirty tree: it identifies the exact
bytes that were run even when no commit describes them.
"""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

# The prototype under measurement. Its identity is these files' contents.
ALGORITHM_SOURCES = (
    "src/build_index.py",
    "src/audio_processing.py",
    "src/music_recognition.py",
)

# The benchmark itself. Changing any of these can move a number.
HARNESS_SOURCES = (
    "musicintel/eval/recognition.py",
    "musicintel/eval/recognizer.py",
    "musicintel/eval/manifest.py",
    "musicintel/eval/metrics.py",
    "musicintel/eval/degradation.py",
    "musicintel/eval/provenance.py",
)


# The Phase 1 landmark recognition pipeline. Separate from HARNESS_SOURCES on
# purpose: the harness is the thing that MEASURES, this is the thing being
# measured, and a report needs to pin both. The Phase 1D audit found a report
# whose only source fingerprint covered the harness, leaving the recognizer that
# actually produced the numbers unidentified -- this tuple closes that.
#
# A benchmark driver should fingerprint PHASE1_SOURCES plus its own path, so the
# record covers the evaluation code as well as the engine.
PHASE1_SOURCES = (
    "musicintel/recognition/fingerprint.py",
    "musicintel/recognition/index.py",
    "musicintel/recognition/matcher.py",
    "musicintel/recognition/decision.py",
)


def _git(args: list[str], repo_root: str | Path) -> str:
    """Run a git command, returning stripped stdout ('' on any failure)."""
    try:
        r = subprocess.run(
            ["git", *args], cwd=repo_root,
            capture_output=True, text=True, timeout=10,
        )
        return r.stdout.strip() if r.returncode == 0 else ""
    except Exception:  # noqa: BLE001 -- provenance must never break a run
        return ""


def git_state(repo_root: str | Path) -> dict:
    """Commit, dirtiness, and the dirty paths.

    Untracked files count as dirty: the manifest and the harness were both
    untracked when the first baseline was measured, and calling that tree clean
    is precisely the false claim this exists to prevent.
    """
    commit = _git(["rev-parse", "HEAD"], repo_root) or "unknown"
    porcelain = _git(["status", "--porcelain"], repo_root)
    dirty_paths = sorted(
        line[3:].strip() for line in porcelain.splitlines() if line.strip()
    )
    return {
        "commit": commit,
        "commit_short": commit[:7] if commit != "unknown" else "unknown",
        "dirty": bool(dirty_paths),
        "dirty_paths": dirty_paths,
    }


def source_fingerprint(repo_root: str | Path, rel_paths: tuple[str, ...]) -> str:
    """SHA-256 over (path, file-hash) pairs -- stable, order-independent.

    A missing file is recorded as "missing" rather than skipped, so deleting a
    source file changes the fingerprint instead of silently preserving it.
    """
    root = Path(repo_root)
    h = hashlib.sha256()
    for rel in sorted(rel_paths):
        p = root / rel
        digest = (
            hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else "missing"
        )
        h.update(f"{rel}:{digest}\n".encode())
    return h.hexdigest()


def version_string(prefix: str, repo_root: str | Path) -> str:
    """Version label derived from git, e.g. `prototype@1b07eba+dirty`.

    The `+dirty` suffix is not cosmetic: it is the difference between a label a
    reader can `git checkout` and one they cannot.
    """
    g = git_state(repo_root)
    return f"{prefix}@{g['commit_short']}" + ("+dirty" if g["dirty"] else "")
