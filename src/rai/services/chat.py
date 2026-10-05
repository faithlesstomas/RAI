"""Tombstone for the removed ChatService compatibility facade."""

from __future__ import annotations


class ChatService:
    """ChatService was removed in Issue #43. Use AssistantService."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise RuntimeError(
            "ChatService was removed in #43. Use AssistantService via "
            "POST /api/v1/assistant/turn or 'rai assistant chat'."
        )
