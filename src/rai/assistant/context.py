"""Context construction and manifest assembly for assistant inference."""

from __future__ import annotations

import json
from datetime import datetime, timezone

from returns.result import Failure, Result, Success

from rai.kernel.records import (
    ActionFailure,
    DataClass,
    ProducerIdentity,
    _new_id,
    _utc_now,
)

from .ports import MemoryGraphStore
from .query import MemoryQueryResolver
from .records import (
    AssistantContextManifest,
    AssistantContextManifestItem,
    AssistantContextPackage,
    ConversationTurn,
    make_assistant_failure,
)

DEFAULT_SYSTEM_INSTRUCTION = (
    "You are Rich AI (RAI), an intelligent and secure local assistant for the GNU/Linux desktop. "
    "You respect user preferences stored in durable memory and provide concise, accurate answers."
)


class AssistantContextBuilder:
    """Selects recent dialogue and durable memories independently to build a ContextPackage."""

    def __init__(  # noqa: PLR0913
        self,
        store: MemoryGraphStore,
        query_resolver: MemoryQueryResolver | None = None,
        system_instruction: str = DEFAULT_SYSTEM_INSTRUCTION,
        max_recent_turns: int = 10,
        max_memories: int = 5,
        max_context_characters: int = 8_000,
        producer: ProducerIdentity | None = None,
    ) -> None:
        self.store = store
        self.query_resolver = query_resolver or MemoryQueryResolver()
        self.system_instruction = system_instruction
        self.max_recent_turns = max_recent_turns
        self.max_memories = max_memories
        self.max_context_characters = max_context_characters
        self.producer = producer or ProducerIdentity(
            producer_id="assistant-context-builder", kind="service", version="1.0.0"
        )

    async def build_context(  # noqa: PLR0915
        self, turn: ConversationTurn
    ) -> Result[AssistantContextPackage, ActionFailure]:
        """Assemble a bounded, inspectable AssistantContextPackage."""
        query = self.query_resolver.resolve(turn.text)

        recent_res = await self.store.get_recent_reply_chain(
            session_id=turn.session_id,
            limit=self.max_recent_turns,
            before_turn_id=turn.record_id,
        )
        if isinstance(recent_res, Failure):
            return Failure(recent_res.failure())
        recent_turns = recent_res.unwrap()

        mem_res = await self.store.retrieve_relevant_memories(
            profile_scope="default",
            query=query,
            data_classes=query.data_classes,
            limit=self.max_memories,
        )
        if isinstance(mem_res, Failure):
            return Failure(mem_res.failure())
        memories_with_reasons = mem_res.unwrap()

        items: list[AssistantContextManifestItem] = []
        recent_ids: list[str] = []
        for rt in recent_turns:
            recent_ids.append(rt.record_id)
            items.append(
                AssistantContextManifestItem(
                    source_id=rt.record_id,
                    source_type="conversation_turn",
                    layer="recent_conversation",
                    data_class=rt.data_class,
                    ranking_reason="reply-chain chronological window",
                    fields=("text", "role"),
                )
            )

        durable_ids: list[str] = []
        ranking_reasons: dict[str, str] = {}
        durable_content: list[dict[str, object]] = []
        exclusions: list[str] = []
        now = datetime.now(timezone.utc)
        for mem, reason in memories_with_reasons:
            data_class = (
                mem.data_class
                if isinstance(mem.data_class, DataClass)
                else DataClass(mem.data_class)
            )
            valid_source = await self.store.get_turn(mem.source_turn_id)
            invalid_reasons = []
            if mem.profile_scope != query.profile_scope:
                invalid_reasons.append("profile_scope")
            if data_class not in query.data_classes:
                invalid_reasons.append("privacy_class")
            if mem.valid_from > now or (
                mem.valid_until is not None and mem.valid_until <= now
            ):
                invalid_reasons.append("temporal_validity")
            if isinstance(valid_source, Failure) or valid_source.unwrap() is None:
                invalid_reasons.append("missing_provenance_source")
            if invalid_reasons:
                exclusions.append(f"{mem.record_id}:{','.join(invalid_reasons)}")
                continue
            durable_ids.append(mem.record_id)
            ranking_reasons[mem.record_id] = reason
            durable_content.append(
                {
                    "record_id": mem.record_id,
                    "kind": mem.kind,
                    "topic": mem.topic,
                    "content": mem.content,
                }
            )
            items.append(
                AssistantContextManifestItem(
                    source_id=mem.record_id,
                    source_type="memory_record",
                    layer="durable_memory",
                    data_class=mem.data_class,
                    ranking_reason=reason,
                    fields=("topic", "content"),
                )
            )

        recent_content = [
            {"record_id": t.record_id, "role": t.role, "text": t.text}
            for t in recent_turns
        ]
        content: dict[str, object] = {
            "system_instruction": self.system_instruction,
            "current_turn": {
                "record_id": turn.record_id,
                "role": turn.role,
                "text": turn.text,
            },
            "recent_turns": recent_content,
            "durable_memories": durable_content,
        }

        serialized = json.dumps(content, ensure_ascii=False)
        while len(serialized) > self.max_context_characters and recent_content:
            removed = recent_content.pop(0)
            removed_id = str(removed["record_id"])
            recent_ids.remove(removed_id)
            items = [item for item in items if item.source_id != removed_id]
            exclusions.append(f"{removed_id}:character_budget")
            serialized = json.dumps(content, ensure_ascii=False)
        while len(serialized) > self.max_context_characters and durable_content:
            removed = durable_content.pop()
            removed_id = str(removed["record_id"])
            durable_ids.remove(removed_id)
            ranking_reasons.pop(removed_id, None)
            items = [item for item in items if item.source_id != removed_id]
            exclusions.append(f"{removed_id}:character_budget")
            serialized = json.dumps(content, ensure_ascii=False)
        if len(serialized) > self.max_context_characters:
            return Failure(
                make_assistant_failure(
                    code="CONTEXT_BUDGET_EXCEEDED",
                    message=(
                        "the current turn and required instructions exceed the "
                        f"{self.max_context_characters}-character context budget"
                    ),
                    request_id=turn.record_id,
                )
            )
        total_chars = len(serialized)
        est_tokens = max(1, total_chars // 4)

        manifest = AssistantContextManifest(
            record_id=_new_id(),
            timestamp=_utc_now(),
            producer=self.producer,
            session_id=turn.session_id,
            turn_id=turn.record_id,
            recent_turn_ids=tuple(recent_ids),
            durable_memory_ids=tuple(durable_ids),
            ranking_reasons=ranking_reasons,
            exclusions=tuple(exclusions),
            redactions=(),
            retriever_version="1.0.0",
            policy_version="1.0.0",
            actual_tokens=est_tokens,
            actual_characters=total_chars,
            items=tuple(items),
        )

        package = AssistantContextPackage(
            record_id=_new_id(),
            timestamp=_utc_now(),
            producer=self.producer,
            session_id=turn.session_id,
            turn_id=turn.record_id,
            manifest=manifest,
            content=content,
        )

        return Success(package)
