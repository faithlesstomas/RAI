"""Bounded interaction service orchestrating turns, context, and memory."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from datetime import timedelta
import time
from typing import Any, AsyncIterator

from returns.result import Failure, Result, Success

from rai.kernel.ports import CancellationToken
from rai.kernel.records import (
    ActionFailure,
    InferenceBudget,
    ProducerIdentity,
    ProvenanceReference,
    _new_id,
    _utc_now,
)

from .audit import AssistantAuditEntry, AssistantAuditLedger, InMemoryAssistantAuditLedger
from .backends.deterministic import DeterministicAssistantBackend
from .context import AssistantContextBuilder
from .ports import AssistantModelBackend, MemoryGraphStore
from .records import (
    AssistantCandidate,
    AssistantResponse,
    ConversationTurn,
    InferenceRequest,
    MemoryRecord,
    MemoryRelation,
    MemoryRelationKind,
)


class AssistantService:
    """One container-owned application boundary for assistant turns."""

    def __init__(
        self,
        store: MemoryGraphStore,
        backend: AssistantModelBackend | None = None,
        context_builder: AssistantContextBuilder | None = None,
        audit_ledger: AssistantAuditLedger | None = None,
        producer: ProducerIdentity | None = None,
    ) -> None:
        self.store = store
        self.backend = backend or DeterministicAssistantBackend()
        self.context_builder = context_builder or AssistantContextBuilder(store=store)
        self.audit_ledger = audit_ledger or InMemoryAssistantAuditLedger()
        self.producer = producer or ProducerIdentity(
            producer_id="assistant-service", kind="service", version="1.0.0"
        )

    async def accept_turn(
        self,
        turn: ConversationTurn,
        cancellation: CancellationToken | None = None,
        request_id: str | None = None,
    ) -> Result[AssistantResponse, ActionFailure]:
        """Execute the one-turn assistant pipeline with exactly-once terminal delivery."""
        token = cancellation or CancellationToken()
        req_id = request_id or turn.record_id

        existing_res = await self.store.get_response_by_request_id(req_id)
        if isinstance(existing_res, Success) and existing_res.unwrap() is not None:
            return Success(existing_res.unwrap())  # type: ignore[arg-type]

        accepted_res = await self.store.accept_turn(turn)
        if isinstance(accepted_res, Failure):
            return Failure(accepted_res.failure())

        ctx_res = await self.context_builder.build_context(turn)
        if isinstance(ctx_res, Failure):
            return Failure(ctx_res.failure())
        context_package = ctx_res.unwrap()
        manifest = context_package.manifest

        now = _utc_now()
        budget = InferenceBudget(
            record_id=_new_id(),
            timestamp=now,
            producer=self.producer,
            max_input_tokens=2048,
            max_output_tokens=512,
            max_agent_turns=1,
            max_tool_calls=0,
            max_images=0,
            max_audio_seconds=0,
            max_latency_seconds=30,
            max_provider_cost=0,
            max_ram_bytes=1024 * 1024 * 1024,
            max_vram_bytes=0,
            cancellation_deadline=now + timedelta(seconds=30),
        )

        inference_req = InferenceRequest(
            record_id=_new_id(),
            timestamp=now,
            producer=self.producer,
            session_id=turn.session_id,
            turn_id=turn.record_id,
            request_id=req_id,
            context=context_package,
            budget=budget,
            strategy="DIRECT",
        )

        start_time = time.perf_counter()
        backend_res = await self.backend.generate(inference_req, token)
        latency_ms = (time.perf_counter() - start_time) * 1000

        if isinstance(backend_res, Failure):
            err = backend_res.failure()
            status = "CANCELLED" if getattr(err, "code", "") == "CANCELLED" else "FAILED"
            fail_resp = AssistantResponse(
                record_id=_new_id(),
                timestamp=_utc_now(),
                producer=self.producer,
                session_id=turn.session_id,
                turn_id=turn.record_id,
                user_turn_id=turn.record_id,
                request_id=req_id,
                manifest_id=manifest.record_id,
                text="",
                status=status,
                error_message=err.message,
            )
            await self.store.commit_terminal(
                response=fail_resp,
                manifest=manifest,
                assistant_turn=None,
                memories=(),
                relations=(),
            )
            await self.audit_ledger.append(
                AssistantAuditEntry(
                    session_id=turn.session_id,
                    turn_id=turn.record_id,
                    request_id=req_id,
                    manifest_id=manifest.record_id,
                    status=status,
                    latency_ms=latency_ms,
                )
            )
            return Failure(err)

        candidate = backend_res.unwrap()
        asst_turn_id = _new_id()
        asst_turn = ConversationTurn(
            record_id=asst_turn_id,
            timestamp=_utc_now(),
            producer=self.producer,
            session_id=turn.session_id,
            role="assistant",
            text=candidate.text,
            reply_to_turn_id=turn.record_id,
            data_class=turn.data_class,
            status="COMPLETED",
        )

        admitted_memories: list[MemoryRecord] = []
        admitted_ids: list[str] = []
        relations: list[MemoryRelation] = [
            MemoryRelation(
                relation_id=_new_id(),
                source_id=asst_turn_id,
                target_id=turn.record_id,
                kind=MemoryRelationKind.REPLIES_TO,
            )
        ]

        for proposal in candidate.proposals:
            mem_id = _new_id()
            memory = MemoryRecord(
                record_id=mem_id,
                timestamp=_utc_now(),
                producer=self.producer,
                kind=proposal.kind,
                topic=proposal.topic,
                content=proposal.content,
                source_turn_id=turn.record_id,
                data_class=turn.data_class,
                profile_scope="default",
                valid_from=_utc_now(),
                valid_until=None,
                provenance=(
                    ProvenanceReference(
                        source_id=turn.record_id,
                        source_type="conversation_turn",
                        source_version="1.0.0",
                        relation="DERIVED_FROM",
                        producer=turn.producer,
                    ),
                ),
            )
            admitted_memories.append(memory)
            admitted_ids.append(mem_id)

        completed_resp = AssistantResponse(
            record_id=_new_id(),
            timestamp=_utc_now(),
            producer=self.producer,
            session_id=turn.session_id,
            turn_id=asst_turn_id,
            user_turn_id=turn.record_id,
            request_id=req_id,
            manifest_id=manifest.record_id,
            text=candidate.text,
            status="COMPLETED",
            error_message=None,
            admitted_memory_ids=tuple(admitted_ids),
        )

        commit_res = await self.store.commit_terminal(
            response=completed_resp,
            manifest=manifest,
            assistant_turn=asst_turn,
            memories=tuple(admitted_memories),
            relations=tuple(relations),
        )
        if isinstance(commit_res, Failure):
            return Failure(commit_res.failure())

        await self.audit_ledger.append(
            AssistantAuditEntry(
                session_id=turn.session_id,
                turn_id=turn.record_id,
                request_id=req_id,
                manifest_id=manifest.record_id,
                status="COMPLETED",
                latency_ms=latency_ms,
                tokens={"input": candidate.tokens_in, "output": candidate.tokens_out},
                admitted_memories=tuple(admitted_ids),
            )
        )

        return Success(completed_resp)

    async def stream_turn(
        self,
        turn: ConversationTurn,
        on_chunk: Callable[[str], Any],
        cancellation: CancellationToken | None = None,
        request_id: str | None = None,
    ) -> Result[AssistantResponse, ActionFailure]:
        """Stream assistant output chunks as transport events before committing terminal state."""
        token = cancellation or CancellationToken()
        req_id = request_id or turn.record_id

        existing_res = await self.store.get_response_by_request_id(req_id)
        if isinstance(existing_res, Success):
            resp = existing_res.unwrap()
            if resp is not None:
                on_chunk(resp.text)
                return Success(resp)

        accepted_res = await self.store.accept_turn(turn)
        if isinstance(accepted_res, Failure):
            return Failure(accepted_res.failure())

        ctx_res = await self.context_builder.build_context(turn)
        if isinstance(ctx_res, Failure):
            return Failure(ctx_res.failure())
        context_package = ctx_res.unwrap()
        manifest = context_package.manifest

        now = _utc_now()
        budget = InferenceBudget(
            record_id=_new_id(),
            timestamp=now,
            producer=self.producer,
            max_input_tokens=2048,
            max_output_tokens=512,
            max_agent_turns=1,
            max_tool_calls=0,
            max_images=0,
            max_audio_seconds=0,
            max_latency_seconds=30,
            max_provider_cost=0,
            max_ram_bytes=1024 * 1024 * 1024,
            max_vram_bytes=0,
            cancellation_deadline=now + timedelta(seconds=30),
        )

        inference_req = InferenceRequest(
            record_id=_new_id(),
            timestamp=now,
            producer=self.producer,
            session_id=turn.session_id,
            turn_id=turn.record_id,
            request_id=req_id,
            context=context_package,
            budget=budget,
            strategy="DIRECT",
        )

        accumulated: list[str] = []
        stream_err: ActionFailure | None = None

        async for chunk_res in self.backend.stream(inference_req, token):
            if isinstance(chunk_res, Failure):
                stream_err = chunk_res.failure()
                break
            chunk = chunk_res.unwrap()
            accumulated.append(chunk)
            on_chunk(chunk)

        if stream_err is not None:
            status = "CANCELLED" if getattr(stream_err, "code", "") == "CANCELLED" else "FAILED"
            fail_resp = AssistantResponse(
                record_id=_new_id(),
                timestamp=_utc_now(),
                producer=self.producer,
                session_id=turn.session_id,
                turn_id=turn.record_id,
                user_turn_id=turn.record_id,
                request_id=req_id,
                manifest_id=manifest.record_id,
                text="".join(accumulated),
                status=status,
                error_message=stream_err.message,
            )
            await self.store.commit_terminal(
                response=fail_resp,
                manifest=manifest,
                assistant_turn=None,
                memories=(),
                relations=(),
            )
            return Failure(stream_err)

        full_text = "".join(accumulated)
        asst_turn_id = _new_id()
        asst_turn = ConversationTurn(
            record_id=asst_turn_id,
            timestamp=_utc_now(),
            producer=self.producer,
            session_id=turn.session_id,
            role="assistant",
            text=full_text or "Rozumiem.",
            reply_to_turn_id=turn.record_id,
            data_class=turn.data_class,
            status="COMPLETED",
        )

        completed_resp = AssistantResponse(
            record_id=_new_id(),
            timestamp=_utc_now(),
            producer=self.producer,
            session_id=turn.session_id,
            turn_id=asst_turn_id,
            user_turn_id=turn.record_id,
            request_id=req_id,
            manifest_id=manifest.record_id,
            text=full_text or "Rozumiem.",
            status="COMPLETED",
            error_message=None,
        )

        await self.store.commit_terminal(
            response=completed_resp,
            manifest=manifest,
            assistant_turn=asst_turn,
            memories=(),
            relations=(
                MemoryRelation(
                    relation_id=_new_id(),
                    source_id=asst_turn_id,
                    target_id=turn.record_id,
                    kind=MemoryRelationKind.REPLIES_TO,
                ),
            ),
        )
        return Success(completed_resp)

    async def get_recent_turns(
        self, session_id: str, limit: int = 10
    ) -> Result[tuple[ConversationTurn, ...], ActionFailure]:
        return await self.store.get_recent_reply_chain(session_id=session_id, limit=limit)

    async def delete_turn(self, turn_id: str) -> Result[int, ActionFailure]:
        return await self.store.delete_turn(turn_id)

    async def accept_turn_stream(
        self,
        turn: ConversationTurn,
        cancellation: CancellationToken | None = None,
        request_id: str | None = None,
    ) -> AsyncIterator[Result[str, ActionFailure]]:
        """Yield chunks as streaming transport events."""
        queue: asyncio.Queue[Result[str, ActionFailure] | None] = asyncio.Queue()

        def _on_chunk(chunk: str) -> None:
            queue.put_nowait(Success(chunk))

        async def _run() -> None:
            res = await self.stream_turn(turn, _on_chunk, cancellation, request_id)
            if isinstance(res, Failure):
                queue.put_nowait(Failure(res.failure()))
            queue.put_nowait(None)

        task = asyncio.create_task(_run())
        try:
            while True:
                item = await queue.get()
                if item is None:
                    break
                yield item
        finally:
            await task
