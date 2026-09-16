"""Bounded interaction service orchestrating turns, context, and memory."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from datetime import timedelta
import json
import time
from typing import Any, AsyncIterator

from returns.result import Failure, Result, Success

from rai.kernel.ports import CancellationToken, LifecycleState
from rai.kernel.records import (
    ActionFailure,
    DataClass,
    InferenceBudget,
    ProducerIdentity,
    ProvenanceReference,
    _new_id,
    _utc_now,
)

from .audit import (
    AssistantAuditEntry,
    AssistantAuditLedger,
    InMemoryAssistantAuditLedger,
)
from .backends.deterministic import DeterministicAssistantBackend
from .context import AssistantContextBuilder
from .ports import (
    AssistantModelBackend,
    MemoryGraphStore,
    MemoryProposalExtractor,
    MemoryQuery,
)
from .query import MemoryQueryResolver
from .records import (
    AssistantCandidate,
    AssistantContextManifest,
    AssistantContextPackage,
    AssistantResponse,
    ConversationTurn,
    InferenceRequest,
    MemoryOperation,
    MemoryOperationKind,
    MemoryProposal,
    MemoryRecord,
    MemoryRelation,
    MemoryRelationKind,
    make_assistant_failure,
)

_DATA_CLASS_RANK = {
    DataClass.PUBLIC.value: 0,
    DataClass.LOCAL.value: 1,
    DataClass.PRIVATE.value: 2,
}
_AUTO_ADMISSION_CONFIDENCE = 0.75
_MEMORY_ABSTENTION_TEXT = (
    "Nie mam wystarczających, dopuszczonych dowodów w pamięci, aby odpowiedzieć."
)


def _data_class_value(value: DataClass | str) -> str:
    return value.value if isinstance(value, DataClass) else str(value)


def _memory_operation_value(value: MemoryOperationKind | str) -> str:
    return value.value if isinstance(value, MemoryOperationKind) else str(value)


def _claim_payload(content: dict[str, Any]) -> str:
    """Return a stable semantic payload while ignoring extraction trace metadata."""
    semantic_keys = (
        "subject",
        "predicate",
        "value",
        "preference",
        "plan",
        "fact",
        "attribute",
    )
    semantic = {key: content[key] for key in semantic_keys if key in content}
    return json.dumps(semantic or content, sort_keys=True, ensure_ascii=False)


class AssistantService:
    """One container-owned application boundary for assistant turns."""

    def __init__(  # noqa: PLR0913
        self,
        store: MemoryGraphStore,
        backend: AssistantModelBackend | None = None,
        context_builder: AssistantContextBuilder | None = None,
        memory_extractor: MemoryProposalExtractor | None = None,
        audit_ledger: AssistantAuditLedger | None = None,
        producer: ProducerIdentity | None = None,
        profile_scope: str = "default",
    ) -> None:
        self.store = store
        self.backend = backend or DeterministicAssistantBackend()
        self.context_builder = context_builder or AssistantContextBuilder(
            store=store,
            profile_scope=profile_scope,
        )
        self.memory_extractor = memory_extractor
        self.audit_ledger = audit_ledger or InMemoryAssistantAuditLedger()
        self.producer = producer or ProducerIdentity(
            producer_id="assistant-service", kind="service", version="1.0.0"
        )
        self.profile_scope = profile_scope
        self._state = LifecycleState.CREATED
        self._request_locks: dict[str, asyncio.Lock] = {}

    @staticmethod
    def _proposal_rejection_reason(  # noqa: PLR0911
        proposal: MemoryProposal, turn: ConversationTurn
    ) -> str | None:
        """Apply deterministic admission rules to untrusted memory proposals."""
        if proposal.source_span is None:
            return "missing exact source span"
        if proposal.span_start is None or proposal.span_end is None:
            return "missing source span offsets"
        if turn.text[proposal.span_start : proposal.span_end] != proposal.source_span:
            return "source span does not match the accepted turn"
        if proposal.modality in {"quoted", "hearsay"}:
            return f"{proposal.modality} content requires explicit review"
        if (
            proposal.modality == "hedged"
            or proposal.confidence < _AUTO_ADMISSION_CONFIDENCE
        ):
            return "uncertain content requires explicit review"
        if proposal.negated and proposal.statement_type != "request":
            return "implicit negated content is not auto-admitted"
        return None

    def _operation_evidence(
        self, turn: ConversationTurn
    ) -> tuple[ProvenanceReference, ...]:
        return (
            ProvenanceReference(
                source_id=turn.record_id,
                source_type="conversation_turn",
                source_version=turn.schema_version,
                relation="TRIGGERED_BY",
                producer=turn.producer,
            ),
        )

    @staticmethod
    def _proposal_trace(proposal: MemoryProposal) -> dict[str, object]:
        """Retain the exact untrusted extraction result on its operation trace."""
        return {
            "proposal_id": proposal.record_id,
            "source_span": proposal.source_span,
            "span_start": proposal.span_start,
            "span_end": proposal.span_end,
            "modality": proposal.modality,
            "confidence": proposal.confidence,
        }

    async def _prepare_memory_changes(
        self,
        proposals: tuple[MemoryProposal, ...],
        turn: ConversationTurn,
    ) -> tuple[list[MemoryRecord], list[MemoryOperation], str | None]:
        """Validate proposals and prepare atomic records for the store."""
        admitted: list[MemoryRecord] = []
        operations: list[MemoryOperation] = []
        forget_message: str | None = None
        evidence = self._operation_evidence(turn)

        for proposal in proposals:
            rejection = self._proposal_rejection_reason(proposal, turn)
            operation_value = _memory_operation_value(proposal.operation)
            if rejection:
                operations.append(
                    MemoryOperation(
                        record_id=_new_id(),
                        timestamp=_utc_now(),
                        producer=self.producer,
                        operation=MemoryOperationKind(operation_value),
                        trigger="conversation_turn",
                        trigger_id=turn.record_id,
                        profile_scope=self.profile_scope,
                        policy_outcome="DENY",
                        status="REJECTED",
                        stage="ADMISSION",
                        reason=rejection,
                        evidence=evidence,
                        **self._proposal_trace(proposal),
                    )
                )
                continue

            if operation_value == MemoryOperationKind.FORGET.value:
                query_text = str(proposal.content.get("query", "")).strip()
                if proposal.target_topic == "*" or query_text == "*":
                    query = None
                else:
                    query = MemoryQueryResolver.resolve(
                        query_text, profile_scope=self.profile_scope
                    )
                target_res = await self.store.retrieve_relevant_memories(
                    profile_scope=self.profile_scope,
                    query=query,
                    data_classes=(
                        DataClass.PUBLIC,
                        DataClass.LOCAL,
                        DataClass.PRIVATE,
                    ),
                    limit=100,
                )
                if isinstance(target_res, Failure):
                    operations.append(
                        MemoryOperation(
                            record_id=_new_id(),
                            timestamp=_utc_now(),
                            producer=self.producer,
                            operation=MemoryOperationKind.FORGET,
                            trigger="conversation_turn",
                            trigger_id=turn.record_id,
                            profile_scope=self.profile_scope,
                            policy_outcome="DENY",
                            status="REJECTED",
                            stage="RETRIEVAL",
                            reason=f"target retrieval failed: {target_res.failure().code}",
                            evidence=evidence,
                            **self._proposal_trace(proposal),
                        )
                    )
                    forget_message = "Nie mogłem bezpiecznie sprawdzić pamięci; niczego nie usunąłem."
                    continue
                targets = tuple(
                    memory.record_id for memory, _reason in target_res.unwrap()
                )
                status = "APPLIED" if targets else "NOOP"
                operations.append(
                    MemoryOperation(
                        record_id=_new_id(),
                        timestamp=_utc_now(),
                        producer=self.producer,
                        operation=MemoryOperationKind.FORGET,
                        trigger="conversation_turn",
                        trigger_id=turn.record_id,
                        profile_scope=self.profile_scope,
                        target_memory_ids=targets,
                        active_memory_ids_before=targets,
                        active_memory_ids_after=(),
                        preconditions=("source_turn_exists", "policy_allowed"),
                        policy_outcome="ALLOW",
                        status=status,
                        stage="DELETION",
                        reason=(
                            f"matched memory query: {query_text}"
                            if targets
                            else f"no active memory matched: {query_text}"
                        ),
                        evidence=evidence,
                        **self._proposal_trace(proposal),
                    )
                )
                forget_message = (
                    f"Usunąłem {len(targets)} pasującą informację z pamięci."
                    if len(targets) == 1
                    else f"Usunąłem {len(targets)} pasujących informacji z pamięci."
                    if targets
                    else "Nie znalazłem w aktywnej pamięci pasującej informacji."
                )
                continue

            existing_res = await self.store.retrieve_relevant_memories(
                profile_scope=self.profile_scope,
                query=MemoryQuery(
                    topic=proposal.topic,
                    profile_scope=self.profile_scope,
                    domain_scopes=(
                        "global",
                        proposal.domain_scope.split(":", maxsplit=1)[0],
                    ),
                    purpose=proposal.purpose,
                    data_classes=(
                        DataClass.PUBLIC,
                        DataClass.LOCAL,
                        DataClass.PRIVATE,
                    ),
                ),
                data_classes=(
                    DataClass.PUBLIC,
                    DataClass.LOCAL,
                    DataClass.PRIVATE,
                ),
                limit=10,
            )
            if isinstance(existing_res, Failure):
                operations.append(
                    MemoryOperation(
                        record_id=_new_id(),
                        timestamp=_utc_now(),
                        producer=self.producer,
                        operation=MemoryOperationKind(operation_value),
                        trigger="conversation_turn",
                        trigger_id=turn.record_id,
                        profile_scope=self.profile_scope,
                        policy_outcome="DENY",
                        status="REJECTED",
                        stage="ADMISSION",
                        reason="active-claim lookup failed; no memory was changed",
                        evidence=evidence,
                        **self._proposal_trace(proposal),
                    )
                )
                continue
            existing = next(
                (
                    memory
                    for memory, _reason in existing_res.unwrap()
                    if memory.kind == proposal.kind and memory.topic == proposal.topic
                ),
                None,
            )
            if existing is not None:
                same_claim = _claim_payload(existing.content) == _claim_payload(
                    proposal.content
                )
                if same_claim:
                    operations.append(
                        MemoryOperation(
                            record_id=_new_id(),
                            timestamp=_utc_now(),
                            producer=self.producer,
                            operation=MemoryOperationKind.REMEMBER,
                            trigger="conversation_turn",
                            trigger_id=turn.record_id,
                            profile_scope=self.profile_scope,
                            target_memory_ids=(existing.record_id,),
                            active_memory_ids_before=(existing.record_id,),
                            active_memory_ids_after=(existing.record_id,),
                            policy_outcome="ALLOW",
                            status="NOOP",
                            stage="ADMISSION",
                            reason="equivalent active claim already exists",
                            evidence=evidence,
                            **self._proposal_trace(proposal),
                        )
                    )
                    continue
                mutable_personal_claim = (
                    proposal.scope == "personal"
                    and proposal.modality == "direct"
                    and any(
                        key in proposal.content for key in ("attribute", "preference")
                    )
                )
                if (
                    proposal.statement_type != "correction"
                    and not mutable_personal_claim
                ):
                    operations.append(
                        MemoryOperation(
                            record_id=_new_id(),
                            timestamp=_utc_now(),
                            producer=self.producer,
                            operation=MemoryOperationKind.UPDATE,
                            trigger="conversation_turn",
                            trigger_id=turn.record_id,
                            profile_scope=self.profile_scope,
                            target_memory_ids=(existing.record_id,),
                            active_memory_ids_before=(existing.record_id,),
                            active_memory_ids_after=(existing.record_id,),
                            policy_outcome="DENY",
                            status="REJECTED",
                            stage="ADMISSION",
                            reason=(
                                "candidate conflicts with an active claim; "
                                "an explicit correction is required"
                            ),
                            evidence=evidence,
                            **self._proposal_trace(proposal),
                        )
                    )
                    continue

            proposed_class = _data_class_value(proposal.privacy_class)
            turn_class = _data_class_value(turn.data_class)
            memory_class = (
                proposed_class
                if _DATA_CLASS_RANK[proposed_class] >= _DATA_CLASS_RANK[turn_class]
                else turn_class
            )
            admitted.append(
                MemoryRecord(
                    record_id=_new_id(),
                    timestamp=_utc_now(),
                    producer=self.producer,
                    kind=proposal.kind,
                    topic=proposal.topic,
                    content={
                        **proposal.content,
                        "statement_type": proposal.statement_type,
                        "modality": proposal.modality,
                        "source_span": proposal.source_span,
                        "span_start": proposal.span_start,
                        "span_end": proposal.span_end,
                        "confidence": proposal.confidence,
                        "proposal_id": proposal.record_id,
                        "scope": proposal.scope,
                        "source_type": proposal.source_type,
                    },
                    source_turn_id=turn.record_id,
                    source_type=proposal.source_type,
                    data_class=DataClass(memory_class),
                    profile_scope=self.profile_scope,
                    domain_scope=proposal.domain_scope,
                    purpose=proposal.purpose,
                    epistemic_status=(
                        "observed"
                        if proposal.source_type
                        in {"rich_history_episode", "system_observation"}
                        else "asserted"
                    ),
                    valid_from=proposal.valid_from or _utc_now(),
                    valid_until=proposal.valid_until,
                    provenance=(
                        ProvenanceReference(
                            source_id=turn.record_id,
                            source_type="conversation_turn",
                            source_version=turn.schema_version,
                            relation="DERIVED_FROM",
                            producer=turn.producer,
                        ),
                    ),
                )
            )
        return admitted, operations, forget_message

    async def _extract_proposals(
        self,
        turn: ConversationTurn,
        candidate_proposals: tuple[MemoryProposal, ...],
        cancellation: CancellationToken,
    ) -> tuple[tuple[MemoryProposal, ...], list[MemoryOperation]]:
        """Merge bounded control proposals with schema-constrained model proposals."""
        if self.memory_extractor is None:
            return candidate_proposals, []

        evidence = self._operation_evidence(turn)
        extraction_id = _new_id()
        extracted = await self.memory_extractor.extract(turn, cancellation)
        if isinstance(extracted, Failure):
            failure = extracted.failure()
            operation = MemoryOperation(
                record_id=_new_id(),
                timestamp=_utc_now(),
                producer=self.producer,
                operation=MemoryOperationKind.REFLECT,
                trigger="conversation_turn",
                trigger_id=turn.record_id,
                profile_scope=self.profile_scope,
                proposal_id=extraction_id,
                source_span=turn.text,
                span_start=0,
                span_end=len(turn.text),
                policy_outcome="DENY",
                status="REJECTED",
                stage="EXTRACTION",
                reason=f"{failure.code}: {failure.message}",
                evidence=evidence,
            )
            return candidate_proposals, [operation]

        model_proposals = extracted.unwrap()
        merged: list[MemoryProposal] = list(candidate_proposals)
        seen = {
            (proposal.topic, proposal.span_start, proposal.span_end)
            for proposal in candidate_proposals
        }
        for proposal in model_proposals:
            identity = (proposal.topic, proposal.span_start, proposal.span_end)
            if identity not in seen:
                merged.append(proposal)
                seen.add(identity)

        operation = MemoryOperation(
            record_id=_new_id(),
            timestamp=_utc_now(),
            producer=self.producer,
            operation=MemoryOperationKind.REFLECT,
            trigger="conversation_turn",
            trigger_id=turn.record_id,
            profile_scope=self.profile_scope,
            proposal_id=extraction_id,
            source_span=turn.text,
            span_start=0,
            span_end=len(turn.text),
            policy_outcome="ALLOW",
            status="APPLIED" if model_proposals else "NOOP",
            stage="EXTRACTION",
            reason=f"schema-valid candidates: {len(model_proposals)}",
            evidence=evidence,
        )
        return tuple(merged), [operation]

    @property
    def state(self) -> LifecycleState:
        return self._state

    async def start(self) -> Result[LifecycleState, ActionFailure]:
        """Start durable storage and the selected model backend."""
        store_res = await self.store.start()
        if isinstance(store_res, Failure):
            self._state = LifecycleState.FAILED
            return Failure(store_res.failure())
        backend_res = await self.backend.start()
        if isinstance(backend_res, Failure):
            await self.store.stop()
            self._state = LifecycleState.FAILED
            return Failure(backend_res.failure())
        self._state = LifecycleState.RUNNING
        return Success(self._state)

    async def stop(self) -> Result[LifecycleState, ActionFailure]:
        """Stop model and storage, preserving the first lifecycle failure."""
        backend_res = await self.backend.stop()
        store_res = await self.store.stop()
        if isinstance(backend_res, Failure):
            self._state = LifecycleState.FAILED
            return Failure(backend_res.failure())
        if isinstance(store_res, Failure):
            self._state = LifecycleState.FAILED
            return Failure(store_res.failure())
        self._state = LifecycleState.STOPPED
        return Success(self._state)

    def _budget(self) -> InferenceBudget:
        now = _utc_now()
        return InferenceBudget(
            record_id=_new_id(),
            timestamp=now,
            producer=self.producer,
            max_input_tokens=2048,
            max_output_tokens=int(getattr(self.backend, "max_output_tokens", 512)),
            max_agent_turns=1,
            max_tool_calls=0,
            max_images=0,
            max_audio_seconds=0,
            max_latency_seconds=60,
            max_provider_cost=0,
            max_ram_bytes=2 * 1024 * 1024 * 1024,
            max_vram_bytes=0,
            cancellation_deadline=now + timedelta(seconds=60),
        )

    def _attach_backend_metadata(
        self, context_package: AssistantContextPackage
    ) -> AssistantContextPackage:
        manifest = context_package.manifest.model_copy(
            update={
                "backend_name": getattr(self.backend, "backend_name", "deterministic"),
                "model_name": getattr(
                    self.backend, "model_name", "deterministic-conformance"
                ),
                "model_artifact_version": getattr(
                    self.backend, "model_artifact_version", None
                ),
                "prompt_template_version": getattr(
                    self.backend, "prompt_template_version", "deterministic-v1"
                ),
            }
        )
        return context_package.model_copy(update={"manifest": manifest})

    async def _commit_failure(
        self,
        *,
        turn: ConversationTurn,
        request_id: str,
        manifest: AssistantContextManifest,
        error: ActionFailure,
        latency_ms: float,
    ) -> Failure[AssistantResponse, ActionFailure]:
        status = "CANCELLED" if error.code == "CANCELLED" else "FAILED"
        response = AssistantResponse(
            record_id=_new_id(),
            timestamp=_utc_now(),
            producer=self.producer,
            session_id=turn.session_id,
            turn_id=turn.record_id,
            user_turn_id=turn.record_id,
            request_id=request_id,
            manifest_id=manifest.record_id,
            text="",
            status=status,
            error_message=error.message,
        )
        await self.store.commit_terminal(
            response=response,
            manifest=manifest,
            assistant_turn=None,
            memories=(),
            relations=(),
        )
        await self.audit_ledger.append(
            AssistantAuditEntry(
                session_id=turn.session_id,
                turn_id=turn.record_id,
                request_id=request_id,
                manifest_id=manifest.record_id,
                model_name=str(getattr(self.backend, "model_name", "deterministic")),
                backend_name=str(
                    getattr(self.backend, "backend_name", "deterministic")
                ),
                status=status,
                latency_ms=latency_ms,
            )
        )
        return Failure(error)

    async def accept_turn(
        self,
        turn: ConversationTurn,
        cancellation: CancellationToken | None = None,
        request_id: str | None = None,
    ) -> Result[AssistantResponse, ActionFailure]:
        """Execute the one-turn pipeline with in-process exactly-once inference."""
        req_id = request_id or turn.record_id
        lock = self._request_locks.setdefault(req_id, asyncio.Lock())
        try:
            async with lock:
                return await self._accept_turn_locked(
                    turn, cancellation or CancellationToken(), req_id
                )
        finally:
            if not lock.locked():
                self._request_locks.pop(req_id, None)

    async def _accept_turn_locked(  # noqa: PLR0911
        self,
        turn: ConversationTurn,
        token: CancellationToken,
        request_id: str,
    ) -> Result[AssistantResponse, ActionFailure]:
        turn_query = MemoryQueryResolver.resolve(
            turn.text, profile_scope=self.profile_scope
        )
        turn_domain = next(
            (
                domain
                for domain in turn_query.domain_scopes
                if domain != "global"
            ),
            "global",
        )
        turn = turn.model_copy(
            update={
                "metadata": {**turn.metadata, "profile_scope": self.profile_scope},
                "domain_scope": turn_domain,
                "purpose": "assistant",
            }
        )
        existing_res = await self.store.get_response_by_request_id(request_id)
        if isinstance(existing_res, Failure):
            return Failure(existing_res.failure())
        existing = existing_res.unwrap()
        if existing is not None:
            return Success(existing)

        accepted_res = await self.store.accept_turn(turn)
        if isinstance(accepted_res, Failure):
            return Failure(accepted_res.failure())

        ctx_res = await self.context_builder.build_context(turn)
        if isinstance(ctx_res, Failure):
            manifest = AssistantContextManifest(
                record_id=_new_id(),
                timestamp=_utc_now(),
                producer=self.producer,
                session_id=turn.session_id,
                turn_id=turn.record_id,
                exclusions=("context_build_failed",),
                backend_name=str(
                    getattr(self.backend, "backend_name", "deterministic")
                ),
                model_name=str(getattr(self.backend, "model_name", "deterministic")),
                model_artifact_version=getattr(
                    self.backend, "model_artifact_version", None
                ),
                prompt_template_version=str(
                    getattr(self.backend, "prompt_template_version", "deterministic-v1")
                ),
            )
            return await self._commit_failure(
                turn=turn,
                request_id=request_id,
                manifest=manifest,
                error=ctx_res.failure(),
                latency_ms=0,
            )
        context_package = self._attach_backend_metadata(ctx_res.unwrap())
        manifest = context_package.manifest

        inference_req = InferenceRequest(
            record_id=_new_id(),
            timestamp=_utc_now(),
            producer=self.producer,
            session_id=turn.session_id,
            turn_id=turn.record_id,
            request_id=request_id,
            context=context_package,
            budget=self._budget(),
            strategy="DIRECT",
            model_name=str(getattr(self.backend, "model_name", "deterministic")),
        )

        if manifest.evidence_required and manifest.routing_decision == "no_evidence":
            candidate = AssistantCandidate(
                text=_MEMORY_ABSTENTION_TEXT,
                metadata={
                    "deterministic_abstention": True,
                    "reason": "memory_evidence_required_but_unavailable",
                },
            )
            latency_ms = 0.0
        else:
            start_time = time.perf_counter()
            backend_res = await self.backend.generate(inference_req, token)
            latency_ms = (time.perf_counter() - start_time) * 1000
            if isinstance(backend_res, Failure):
                return await self._commit_failure(
                    turn=turn,
                    request_id=request_id,
                    manifest=manifest,
                    error=backend_res.failure(),
                    latency_ms=latency_ms,
                )
            candidate = backend_res.unwrap()
        if not isinstance(candidate, AssistantCandidate):
            error = make_assistant_failure(
                code="INVALID_OUTPUT",
                message="assistant backend returned an invalid candidate type",
                request_id=request_id,
            )
            return await self._commit_failure(
                turn=turn,
                request_id=request_id,
                manifest=manifest,
                error=error,
                latency_ms=latency_ms,
            )
        proposals, extraction_operations = await self._extract_proposals(
            turn, candidate.proposals, token
        )
        if any(proposal.source_turn_id != turn.record_id for proposal in proposals):
            error = make_assistant_failure(
                code="INVALID_OUTPUT",
                message="memory proposal source does not match the current user turn",
                request_id=request_id,
            )
            return await self._commit_failure(
                turn=turn,
                request_id=request_id,
                manifest=manifest,
                error=error,
                latency_ms=latency_ms,
            )

        (
            admitted_memories,
            memory_operations,
            forget_message,
        ) = await self._prepare_memory_changes(proposals, turn)
        memory_operations = [*extraction_operations, *memory_operations]
        rejected_operations = [
            operation
            for operation in memory_operations
            if operation.status == "REJECTED"
        ]
        delivered_text = forget_message or candidate.text
        if proposals and not admitted_memories and rejected_operations:
            delivered_text = (
                "Nie zapisałem tej informacji automatycznie: "
                f"{rejected_operations[0].reason}."
            )

        assistant_turn_id = _new_id()
        assistant_turn = ConversationTurn(
            record_id=assistant_turn_id,
            timestamp=_utc_now(),
            producer=self.producer,
            session_id=turn.session_id,
            role="assistant",
            text=delivered_text,
            reply_to_turn_id=turn.record_id,
            data_class=turn.data_class,
            domain_scope=turn.domain_scope,
            purpose=turn.purpose,
            status="COMPLETED",
            metadata={"profile_scope": self.profile_scope},
        )

        admitted_ids = [memory.record_id for memory in admitted_memories]
        relations = [
            MemoryRelation(
                relation_id=_new_id(),
                source_id=assistant_turn_id,
                target_id=turn.record_id,
                kind=MemoryRelationKind.REPLIES_TO,
            )
        ]
        response = AssistantResponse(
            record_id=_new_id(),
            timestamp=_utc_now(),
            producer=self.producer,
            session_id=turn.session_id,
            turn_id=assistant_turn_id,
            user_turn_id=turn.record_id,
            request_id=request_id,
            manifest_id=manifest.record_id,
            text=delivered_text,
            status="COMPLETED",
            admitted_memory_ids=tuple(admitted_ids),
            memory_operation_ids=tuple(
                operation.record_id for operation in memory_operations
            ),
        )
        commit_res = await self.store.commit_terminal(
            response=response,
            manifest=manifest,
            assistant_turn=assistant_turn,
            memories=tuple(admitted_memories),
            relations=tuple(relations),
            operations=tuple(memory_operations),
            context=context_package,
        )
        if isinstance(commit_res, Failure):
            return Failure(commit_res.failure())

        response = commit_res.unwrap()
        await self.audit_ledger.append(
            AssistantAuditEntry(
                session_id=turn.session_id,
                turn_id=turn.record_id,
                request_id=request_id,
                manifest_id=manifest.record_id,
                model_name=str(getattr(self.backend, "model_name", "deterministic")),
                backend_name=str(
                    getattr(self.backend, "backend_name", "deterministic")
                ),
                model_artifact_version=getattr(
                    self.backend, "model_artifact_version", None
                ),
                prompt_template_version=str(
                    getattr(self.backend, "prompt_template_version", "deterministic-v1")
                ),
                status="COMPLETED",
                latency_ms=latency_ms,
                tokens={"input": candidate.tokens_in, "output": candidate.tokens_out},
                generation_metadata={
                    key: value
                    for key, value in candidate.metadata.items()
                    if key != "raw_model_output"
                },
                admitted_memories=tuple(admitted_ids),
                memory_operations=response.memory_operation_ids,
            )
        )
        return Success(response)

    async def stream_turn(
        self,
        turn: ConversationTurn,
        on_chunk: Callable[[str], Any],
        cancellation: CancellationToken | None = None,
        request_id: str | None = None,
    ) -> Result[AssistantResponse, ActionFailure]:
        """Deliver a committed response as a transport chunk.

        The MVP deliberately commits the validated candidate and its memory proposals
        through the same path as non-streaming requests. Backend token streaming can be
        added later without creating a second persistence pipeline.
        """
        result = await self.accept_turn(turn, cancellation, request_id)
        if isinstance(result, Success):
            on_chunk(result.unwrap().text)
        return result

    async def get_recent_turns(
        self, session_id: str, limit: int = 10
    ) -> Result[tuple[ConversationTurn, ...], ActionFailure]:
        return await self.store.get_recent_reply_chain(
            session_id=session_id, limit=limit
        )

    async def delete_turn(self, turn_id: str) -> Result[int, ActionFailure]:
        return await self.store.delete_turn(turn_id)

    async def accept_turn_stream(
        self,
        turn: ConversationTurn,
        cancellation: CancellationToken | None = None,
        request_id: str | None = None,
    ) -> AsyncIterator[Result[str, ActionFailure]]:
        """Yield a validated response while retaining one persistence pipeline."""
        result = await self.accept_turn(turn, cancellation, request_id)
        if isinstance(result, Failure):
            yield Failure(result.failure())
            return
        yield Success(result.unwrap().text)
