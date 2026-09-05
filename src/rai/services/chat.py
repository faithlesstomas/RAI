"""Compatibility facade for callers migrating to the AgentBackend port."""

from __future__ import annotations

try:
    from rai.backends.antigravity import AntigravityBackend
except ImportError as import_error:
    _ANTIGRAVITY_IMPORT_ERROR = import_error

    class ChatService:  # pylint: disable=too-few-public-methods
        """Fail clearly when the transitional compatibility extra is absent."""

        def __init__(self, *_args: object, **_kwargs: object) -> None:
            raise RuntimeError(
                "The transitional Antigravity chat backend is not installed. "
                "Install 'rich-ai[antigravity]' or use the provider-neutral "
                "capability runtime."
            ) from _ANTIGRAVITY_IMPORT_ERROR

else:
    class ChatService(AntigravityBackend):
        """Deprecated name preserving the pre-Stage-1 chat API."""
