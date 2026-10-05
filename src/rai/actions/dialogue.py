"""Bounded previous action context; no provider-owned conversation state."""
from __future__ import annotations

from returns.result import Failure

from rai.assistant.ports import MemoryGraphStore
from rai.assistant.records import ConversationTurn
from rai.kernel.records import DataClass

PRIVACY_RANK = {DataClass.PUBLIC: 0, DataClass.LOCAL: 1, DataClass.PRIVATE: 2}


async def previous_action(store: MemoryGraphStore | None, turn: ConversationTurn, profile: str) -> dict:
    """Read only the last eligible assistant reply from this session/profile."""
    if store is None:
        return {}
    fetched = await store.get_recent_reply_chain(session_id=turn.session_id, limit=3)
    if isinstance(fetched, Failure):
        return {}
    for item in reversed(fetched.unwrap()):
        if item.record_id == turn.record_id or item.role != "assistant":
            continue
        if item.metadata.get("profile_scope") != profile:
            return {}
        if PRIVACY_RANK.get(item.data_class, 99) > PRIVACY_RANK.get(turn.data_class, -1):
            return {}
        metadata = item.metadata.get("action", {})
        return dict(metadata) if isinstance(metadata, dict) else {}
    return {}
