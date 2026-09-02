"""Sandboxed decode worker. NOT importable API -- run as `python -m`.

This process is the only place untrusted audio meets a C parser (S1). It is
spawned per request, reads the payload from stdin, writes raw PCM to stdout, and
exits. It never touches the filesystem and never opens a socket.

Order is the whole point: resource limits are installed as the *first*
statements, before `soundfile` -- and therefore libsndfile -- is imported. A
heap overflow in a codec then lands inside a process that has no file
descriptors worth stealing, cannot allocate beyond a ceiling, cannot spend more
than a few CPU-seconds, and cannot write a byte to disk.

Exit codes are the parent's only channel for *why* a decode failed; stderr is
captured but never surfaced to a client.
"""

from __future__ import annotations

import json
import os
import resource
import sys

EXIT_OK = 0
EXIT_UNSUPPORTED = 10      # magic bytes fine, but no parser would take it
EXIT_EMPTY = 11            # decoded to nothing
EXIT_TOO_MANY_CHANNELS = 12
EXIT_INTERNAL = 13
EXIT_BAD_INVOCATION = 14

MAX_CHANNELS = 8


def _install_limits(address_space: int, cpu_seconds: int) -> None:
    """Cage this process before it parses anything."""
    # No files, at all. Defeats temp-file leakage (S4) by construction rather
    # than by discipline, and blocks a codec that tries to spill to disk.
    resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))
    # No child processes: nothing here should ever fork or exec.
    try:
        resource.setrlimit(resource.RLIMIT_NPROC, (0, 0))
    except (ValueError, OSError):
        pass  # not enforceable everywhere; the other limits still hold
    # No core dumps -- a crash must not spill decoded audio onto disk.
    try:
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    except (ValueError, OSError):
        pass
    resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
    try:
        resource.setrlimit(resource.RLIMIT_AS, (address_space, address_space))
    except (ValueError, OSError):
        # Darwin enforces RLIMIT_AS inconsistently. The decoded-frame cap below
        # is the load-bearing memory bound; this is defence in depth.
        pass


def main(argv: list[str]) -> int:
    if len(argv) != 5:
        return EXIT_BAD_INVOCATION
    target_sr = int(argv[1])
    max_seconds = float(argv[2])
    address_space = int(argv[3])
    cpu_seconds = int(argv[4])

    _install_limits(address_space, cpu_seconds)

    data = sys.stdin.buffer.read()
    if not data:
        return EXIT_EMPTY

    # Imported only after the cage is up.
    import io

    import numpy as np
    import soundfile as sf

    try:
        handle = sf.SoundFile(io.BytesIO(data))
    except Exception:
        return EXIT_UNSUPPORTED

    with handle:
        if handle.channels > MAX_CHANNELS:
            return EXIT_TOO_MANY_CHANNELS
        native_sr = int(handle.samplerate)
        if native_sr <= 0:
            return EXIT_UNSUPPORTED

        # THE bomb defence (S2): never read more than `max_seconds` of NATIVE
        # frames. A 1 MB file claiming to hold nine hours yields the same
        # bounded array as a 1 MB file holding thirty seconds, because the cap
        # is on what we read, not on what the container asserts.
        max_frames = int(max_seconds * native_sr)
        try:
            # One frame past the cap: if it comes back, the source is longer
            # than the limit and the parent rejects it. Reading `max_frames`
            # exactly could not distinguish "exactly at the cap" from "longer",
            # and `handle.frames` is container metadata an attacker controls.
            audio = handle.read(
                frames=max_frames + 1, dtype="float32", always_2d=True,
                fill_value=None,
            )
        except Exception:
            return EXIT_UNSUPPORTED
        truncated = audio.shape[0] > max_frames
        if truncated:
            audio = audio[:max_frames]

    if audio.size == 0:
        return EXIT_EMPTY

    # Mono-mix exactly as librosa.load(mono=True) does: mean across channels.
    mono = audio.mean(axis=1) if audio.shape[1] > 1 else audio[:, 0]

    if native_sr != target_sr:
        import soxr
        try:
            mono = soxr.resample(mono, native_sr, target_sr, quality="HQ")
        except Exception:
            return EXIT_INTERNAL

    mono = np.ascontiguousarray(mono, dtype=np.float32)
    if mono.size == 0:
        return EXIT_EMPTY

    header = json.dumps({
        "sample_rate": target_sr,
        "native_sample_rate": native_sr,
        "channels": int(audio.shape[1]),
        "samples": int(mono.size),
        "duration_seconds": float(mono.size / target_sr),
        "truncated": bool(truncated),
    }).encode("utf-8")

    out = sys.stdout.buffer
    out.write(len(header).to_bytes(4, "big"))
    out.write(header)
    out.write(mono.tobytes())
    out.flush()
    return EXIT_OK


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv))
    except MemoryError:
        sys.exit(EXIT_INTERNAL)
    except Exception:
        sys.exit(EXIT_INTERNAL)
