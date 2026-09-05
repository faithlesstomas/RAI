"""Replaceable, optional agent backend adapters."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .antigravity import AntigravityBackend

__all__ = ["AntigravityBackend"]


def __getattr__(name: str) -> Any:  # noqa: ANN401
    """Load optional backend SDKs only when their adapter is requested."""
    if name != "AntigravityBackend":
        raise AttributeError(name)
    from .antigravity import AntigravityBackend  # noqa: PLC0415

    return AntigravityBackend
