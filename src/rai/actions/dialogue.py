"""Bounded previous action context; no provider-owned conversation state."""
from __future__ import annotations

from returns.result import Failure

from rai.assistant.ports import MemoryGraphStore
from rai.assistant.records import ConversationTurn
from rai.kernel.records import DataClass

PRIVACY_RANK = {DataClass.PUBLIC: 0, DataClass.LOCAL: 1, DataClass.PRIVATE: 2}


async def previous_action(  # noqa: PLR0911, PLR0912
    store: MemoryGraphStore | None, turn: ConversationTurn, profile: str
) -> dict:
    """Read only the last eligible assistant reply from this session/profile."""
    in_turn = turn.metadata.get("in_turn_action")
    if isinstance(in_turn, dict) and in_turn:
        return dict(in_turn)
    if store is None:
        return {}
    fetched = await store.get_recent_reply_chain(session_id=turn.session_id, limit=5)
    if isinstance(fetched, Failure):
        return {}
    latest_action: dict = {}
    fallback_choices: dict = {}
    fallback_frame: dict = {}
    for item in reversed(fetched.unwrap()):
        if item.record_id == turn.record_id or item.role != "assistant":
            continue
        if (
            item.metadata.get("profile_scope") != profile
            or PRIVACY_RANK.get(item.data_class, 99) > PRIVACY_RANK.get(turn.data_class, -1)
        ):
            return {}
        metadata = item.metadata.get("action", {})
        item_meta = item.metadata if isinstance(item.metadata, dict) else {}
        action_dict = dict(metadata) if isinstance(metadata, dict) else {}
        if not latest_action and action_dict:
            latest_action = action_dict
        if not fallback_choices:
            if action_dict.get("action_choices"):
                fallback_choices = action_dict["action_choices"]
            elif item_meta.get("action_choices"):
                fallback_choices = item_meta["action_choices"]
        if not fallback_frame:
            if action_dict.get("active_choice_frame"):
                fallback_frame = action_dict["active_choice_frame"]
            elif item_meta.get("active_choice_frame"):
                fallback_frame = item_meta["active_choice_frame"]
        if latest_action.get("action_choices"):
            return latest_action
    if latest_action:
        res = dict(latest_action)
        if not res.get("action_choices") and fallback_choices:
            res["action_choices"] = fallback_choices
        if not res.get("active_choice_frame") and fallback_frame:
            res["active_choice_frame"] = fallback_frame
        return res
    if fallback_choices or fallback_frame:
        res = {}
        if fallback_choices:
            res["action_choices"] = fallback_choices
        if fallback_frame:
            res["active_choice_frame"] = fallback_frame
        return res
    return {}

