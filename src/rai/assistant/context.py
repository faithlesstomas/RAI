"""Context construction and manifest assembly for assistant inference."""

from __future__ import annotations

import json

from returns.result import Failure, Result, Success

from rai.kernel.records import ActionFailure, DataClass, ProducerIdentity, _new_id, _utc_now

from .ports import MemoryGraphStore
from .query import MemoryQueryResolver
from .records import (
    AssistantContextManifest,
    AssistantContextManifestItem,
    AssistantContextPackage,
    ConversationTurn,
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
        producer: ProducerIdentity | None = None,
    ) -> None:
        self.store = store
        self.query_resolver = query_resolver or MemoryQueryResolver()
        self.system_instruction = system_instruction
        self.max_recent_turns = max_recent_turns
        self.max_memories = max_memories
        self.producer = producer or ProducerIdentity(
            producer_id="assistant-context-builder", kind="service", version="1.0.0"
        )

    async def build_context(
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
        for mem, reason in memories_with_reasons:
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

        content: dict[str, object] = {
            "system_instruction": self.system_instruction,
            "current_turn": {
                "record_id": turn.record_id,
                "role": turn.role,
                "text": turn.text,
            },
            "recent_turns": [
                {"record_id": t.record_id, "role": t.role, "text": t.text}
                for t in recent_turns
            ],
            "durable_memories": durable_content,
        }

        serialized = json.dumps(content, ensure_ascii=False)
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
            exclusions=(),
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
