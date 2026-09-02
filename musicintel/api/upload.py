"""Upload size enforcement and format sniffing (S2, S3).

The size cap is pure ASGI rather than a `BaseHTTPMiddleware`, because it has to
act on the body *as it arrives*. A Content-Length check alone is not a limit: a
chunked request has no Content-Length, and a lying one is trivial to send. This
counts the bytes actually received and aborts the stream the moment the cap is
passed, so an attacker cannot make the process buffer a gigabyte by claiming it
is small.

Format sniffing is magic bytes only (S3). Neither the filename nor the declared
Content-Type is trusted for anything; they are advisory at best and
attacker-controlled at worst. Sniffing narrows what reaches the decoder, and the
decoder's successful parse is the real validation.
"""

from __future__ import annotations

from starlette.datastructures import Headers
from starlette.types import ASGIApp, Message, Receive, Scope, Send

# Containers libsndfile handles natively. Anything outside this list is refused
# before a byte reaches a C parser -- a whitelist, not a blacklist.
_MAGIC: tuple[tuple[bytes, str, str], ...] = (
    (b"RIFF", "wav", "audio/wav"),          # RIFF....WAVE, checked further below
    (b"fLaC", "flac", "audio/flac"),
    (b"OggS", "ogg", "audio/ogg"),
    (b"FORM", "aiff", "audio/aiff"),
    (b"ID3", "mp3", "audio/mpeg"),
    (b"\xff\xfb", "mp3", "audio/mpeg"),
    (b"\xff\xf3", "mp3", "audio/mpeg"),
    (b"\xff\xf2", "mp3", "audio/mpeg"),
    (b"\xff\xfa", "mp3", "audio/mpeg"),
    (b"\xff\xe3", "mp3", "audio/mpeg"),
    (b"caff", "caf", "audio/x-caf"),
)

SUPPORTED_FORMATS = ["wav", "flac", "ogg", "aiff", "mp3", "caf"]


def sniff_format(data: bytes) -> tuple[str, str] | None:
    """(format, media_type) from magic bytes, or None if unrecognised."""
    if len(data) < 4:
        return None
    if data[:4] == b"RIFF":
        # RIFF is a container for many things; only WAVE is audio we want.
        if len(data) >= 12 and data[8:12] in (b"WAVE", b"wave"):
            return "wav", "audio/wav"
        return None
    if data[:4] == b"FORM":
        if len(data) >= 12 and data[8:12] in (b"AIFF", b"AIFC"):
            return "aiff", "audio/aiff"
        return None
    for magic, fmt, media in _MAGIC:
        if magic in (b"RIFF", b"FORM"):
            continue
        if data.startswith(magic):
            return fmt, media
    return None


class PayloadTooLarge(Exception):
    def __init__(self, max_bytes: int) -> None:
        super().__init__(f"body exceeded {max_bytes} bytes")
        self.max_bytes = max_bytes


class BodySizeLimitMiddleware:
    """Abort any request whose body exceeds `max_bytes`, as it streams."""

    def __init__(self, app: ASGIApp, *, max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = Headers(scope=scope)
        declared = headers.get("content-length")
        if declared is not None:
            try:
                if int(declared) > self.max_bytes:
                    await self._reject(scope, send)
                    return
            except ValueError:
                pass  # malformed header; the byte counter below still applies

        received = 0
        too_large = False

        async def counting_receive() -> Message:
            nonlocal received, too_large
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_bytes:
                    too_large = True
                    # Truncate rather than hand the oversized chunk onward.
                    return {"type": "http.request", "body": b"", "more_body": False}
            return message

        sent_start = False

        async def guarded_send(message: Message) -> None:
            nonlocal sent_start
            if message["type"] == "http.response.start":
                sent_start = True
            await send(message)

        if too_large and not sent_start:
            await self._reject(scope, send)
            return

        try:
            await self.app(scope, counting_receive, guarded_send)
        finally:
            pass

        if too_large and not sent_start:
            await self._reject(scope, send)

    async def _reject(self, scope: Scope, send: Send) -> None:
        import json

        body = json.dumps({
            "type": "https://docs.musicintel.dev/problems/payload-too-large",
            "title": "Payload Too Large",
            "status": 413,
            "detail": f"Request body exceeds the {self.max_bytes} byte limit.",
            "instance": scope.get("path", ""),
            "max_bytes": self.max_bytes,
        }).encode()
        await send({
            "type": "http.response.start",
            "status": 413,
            "headers": [
                (b"content-type", b"application/problem+json"),
                (b"content-length", str(len(body)).encode()),
            ],
        })
        await send({"type": "http.response.body", "body": body})


__all__ = [
    "BodySizeLimitMiddleware", "PayloadTooLarge", "SUPPORTED_FORMATS",
    "sniff_format",
]
