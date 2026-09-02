"""HTTP API for Music Intelligence (Stage 3).

The recognition core is frozen. Everything in this package is transport,
validation, isolation and safety around it -- nothing here changes how audio is
fingerprinted, matched or decided.

`create_app` is resolved lazily. The sandboxed decode worker
(`_decoder.py`) runs as a bare script and must not drag FastAPI into the
process that parses untrusted audio; keeping this module import-free at the top
level is what makes that possible.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from musicintel.api.app import create_app

__all__ = ["create_app"]


def __getattr__(name: str):
    if name == "create_app":
        from musicintel.api.app import create_app as _create_app
        return _create_app
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
