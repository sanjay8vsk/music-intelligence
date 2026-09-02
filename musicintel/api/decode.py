"""Parent side of the decode sandbox (S1, S2, S3, S4).

`decode_audio` never parses audio. It sniffs magic bytes, hands the payload to a
short-lived worker over a pipe, and interprets an exit code. The worker is the
only thing that touches libsndfile, and it is dead before the response is
written.

WHY A SUBPROCESS AND NOT A THREAD
---------------------------------
Audio codecs are C. libsndfile, libFLAC, libvorbis and the MPEG decoders have a
long history of heap overflows, and a corrupted heap in a worker thread is a
corrupted heap in the API process -- one that serves other tenants' requests.
A separate process means the blast radius of a parser bug is one request, and
`setrlimit` gives the kernel a say in how large that radius can get.

WHY NOT librosa IN THE WORKER
-----------------------------
It would be the obvious choice, since `load_audio` uses it and matching the
reference decode exactly is non-negotiable. Measured, a cold worker importing
librosa costs **2.3 s** -- eight times the entire Stage 3 latency budget --
because librosa's lazy loader pulls in scipy and numba on first use. The worker
uses `soundfile` + `soxr` directly, which costs **0.12 s**, and reproduces
`librosa.load(..., sr=11025, mono=True)` exactly: mean across channels, then
soxr HQ resampling. Verified against 45 real corpus tracks at 8 kHz, 22.05 kHz
and 44.1 kHz -- **45/45 produced byte-identical fingerprints**, maximum sample
difference 2.4e-07, which is float32 epsilon.

The cost of that choice is coverage: libsndfile opens 478 of the 500 corpus
tracks (95.6%). The remaining 4.4% are MP3 variants only ffmpeg/audioread will
take, and they are rejected with 415 rather than handed to a second, larger
parser. For untrusted input that is the right trade -- one whitelisted parser,
not two.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from musicintel.api.upload import sniff_format

_DECODER = Path(__file__).with_name("_decoder.py")

EXIT_OK = 0
EXIT_UNSUPPORTED = 10
EXIT_EMPTY = 11
EXIT_TOO_MANY_CHANNELS = 12
EXIT_INTERNAL = 13
EXIT_BAD_INVOCATION = 14


class DecodeError(Exception):
    """Decode refused. `reason` is a stable slug for metrics and mapping."""

    def __init__(self, reason: str, message: str) -> None:
        super().__init__(message)
        self.reason = reason
        self.message = message


@dataclass(frozen=True)
class DecodedAudio:
    samples: np.ndarray          # float32, mono, at `sample_rate`
    sample_rate: int
    native_sample_rate: int
    channels: int
    duration_seconds: float
    truncated: bool              # hit the decode-seconds cap
    decode_seconds: float
    format: str

    def __len__(self) -> int:
        return int(self.samples.size)


def decode_audio(
    data: bytes,
    *,
    target_sample_rate: int = 11025,
    max_seconds: float = 30.0,
    timeout_seconds: float = 10.0,
    memory_limit_bytes: int = 512 * 1024 * 1024,
    cpu_seconds: int = 10,
) -> DecodedAudio:
    """Decode untrusted bytes to mono float32. Raises DecodeError on refusal."""
    if not data:
        raise DecodeError("empty", "The uploaded file is empty.")

    sniffed = sniff_format(data)
    if sniffed is None:
        # Refused before a byte reaches a C parser.
        raise DecodeError(
            "unsupported_format",
            "The upload is not a recognised audio container.",
        )
    fmt, _media = sniffed

    argv = [
        sys.executable, "-I", str(_DECODER),
        str(int(target_sample_rate)), str(float(max_seconds)),
        str(int(memory_limit_bytes)), str(int(cpu_seconds)),
    ]
    started = time.perf_counter()
    try:
        proc = subprocess.Popen(
            argv,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            # A bare environment: no inherited credentials, no PYTHONPATH games.
            env={"PATH": "/usr/bin:/bin"},
            close_fds=True,
        )
    except OSError as exc:  # pragma: no cover - fork failure
        raise DecodeError("decoder_unavailable", "Decoder could not be started.") from exc

    try:
        stdout, _stderr = proc.communicate(input=data, timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.communicate()
        raise DecodeError(
            "decode_timeout",
            f"Decoding did not finish within {timeout_seconds:g} seconds.",
        )
    finally:
        if proc.poll() is None:  # pragma: no cover - belt and braces
            proc.kill()

    elapsed = time.perf_counter() - started
    code = proc.returncode

    if code != EXIT_OK:
        raise DecodeError(*_explain(code))

    if len(stdout) < 4:
        raise DecodeError("decode_failed", "Decoder produced no output.")
    header_len = int.from_bytes(stdout[:4], "big")
    if header_len <= 0 or len(stdout) < 4 + header_len:
        raise DecodeError("decode_failed", "Decoder output was truncated.")
    try:
        header = json.loads(stdout[4:4 + header_len])
    except json.JSONDecodeError:
        raise DecodeError("decode_failed", "Decoder output was malformed.")

    payload = stdout[4 + header_len:]
    samples = np.frombuffer(payload, dtype=np.float32)
    if samples.size != int(header.get("samples", -1)):
        raise DecodeError("decode_failed", "Decoder output was incomplete.")

    duration = float(samples.size) / float(target_sample_rate)
    # S2 says enforce after decode too. The worker caps what it reads; this is
    # the independent check that the cap actually held, and it does not trust
    # the worker's own duration field to say so.
    if duration > max_seconds + 1.0:
        raise DecodeError(
            "decoded_too_long",
            f"Decoded audio exceeds the {max_seconds:g} second limit.",
        )

    return DecodedAudio(
        samples=np.ascontiguousarray(samples, dtype=np.float32),
        sample_rate=int(header["sample_rate"]),
        native_sample_rate=int(header["native_sample_rate"]),
        channels=int(header["channels"]),
        duration_seconds=duration,
        truncated=bool(header.get("truncated", False)),
        decode_seconds=elapsed,
        format=fmt,
    )


def _explain(code: int) -> tuple[str, str]:
    """Exit code -> (reason slug, client-safe message)."""
    if code == EXIT_UNSUPPORTED:
        return ("unsupported_format",
                "The audio could not be decoded. The container may be corrupt "
                "or use an unsupported codec.")
    if code == EXIT_EMPTY:
        return ("empty_audio", "The upload decoded to no audio.")
    if code == EXIT_TOO_MANY_CHANNELS:
        return ("too_many_channels", "The audio has too many channels.")
    if code == EXIT_BAD_INVOCATION:  # pragma: no cover - programming error
        return ("decode_failed", "Decoder was invoked incorrectly.")
    if code == EXIT_INTERNAL:
        return ("decode_failed", "The audio could not be decoded.")
    if code < 0:
        # Killed by a signal: SIGXCPU from RLIMIT_CPU, SIGSEGV from a codec
        # crash, SIGKILL from the OOM killer. The cage did its job; the client
        # gets one message and the process that died was not the API worker.
        return ("decode_crashed",
                "The audio could not be decoded and was rejected.")
    return ("decode_failed", "The audio could not be decoded.")


__all__ = ["DecodeError", "DecodedAudio", "decode_audio"]
