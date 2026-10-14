"""Bounded interaction service orchestrating turns, context, and memory."""

from __future__ import annotations


import asyncio
from collections.abc import Awaitable, Callable
from datetime import timedelta
import json
import time
from typing import TYPE_CHECKING, Any, AsyncIterator

from rai.diagnostics import trace, trace_payload

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

if TYPE_CHECKING:
    from rai.actions.workflow import ResearchWorkflow, TaskStepController
    from rai.kernel.service import CapabilityService

from .audit import (
    AssistantAuditEntry,
    AssistantAuditLedger,
    InMemoryAssistantAuditLedger,
)
from .backends.deterministic import DeterministicAssistantBackend
from .context import AssistantContextBuilder
from .ports import (
    AssistantModelBackend,
    AssistantActionHandler,
    MemoryGraphStore,
    MemoryProposalExtractor,
    MemoryQuery,
)
from .query import MemoryQueryResolver
from .records import (
    MAX_PROPOSALS_PER_TURN,
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
    ToolObservation,
    make_assistant_failure,
)
from .tool_calling import extract_model_tool_call, intent_from_tool_call

_DATA_CLASS_RANK = {
    DataClass.PUBLIC.value: 0,
    DataClass.LOCAL.value: 1,
    DataClass.PRIVATE.value: 2,
}
_AUTO_ADMISSION_CONFIDENCE = 0.75
_MEMORY_ABSTENTION_TEXT = (
    "Nie mam wystarczających, dopuszczonych dowodów w pamięci, aby odpowiedzieć."
)
_MAX_SYNTHESIS_TOOL_OBSERVATIONS: int = 3


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
        actions: AssistantActionHandler | None = None,
        step_controller: TaskStepController | None = None,
        context_approver: Callable[
            [AssistantContextPackage, CancellationToken],
            Awaitable[Result[AssistantContextPackage, ActionFailure]],
        ]
        | None = None,
    ) -> None:
        self.store = store
        self.actions = actions
        self.step_controller = step_controller
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
        self.context_approver = context_approver
        self._state = LifecycleState.CREATED
        self._turn_locks: dict[str, asyncio.Lock] = {}
        self._turn_lock_references: dict[str, int] = {}
        self._turn_locks_guard = asyncio.Lock()

    async def _retain_turn_lock(self, turn_id: str) -> asyncio.Lock:
        """Keep one lock alive while owners hold it or wait in its queue."""
        async with self._turn_locks_guard:
            lock = self._turn_locks.setdefault(turn_id, asyncio.Lock())
            self._turn_lock_references[turn_id] = (
                self._turn_lock_references.get(turn_id, 0) + 1
            )
            return lock

    async def _release_turn_lock(self, turn_id: str) -> None:
        async with self._turn_locks_guard:
            remaining = self._turn_lock_references.get(turn_id, 1) - 1
            if remaining <= 0:
                self._turn_lock_references.pop(turn_id, None)
                self._turn_locks.pop(turn_id, None)
            else:
                self._turn_lock_references[turn_id] = remaining

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
                dependent_res = await self.store.get_dependent_memory_ids(
                    targets, profile_scope=self.profile_scope
                )
                if isinstance(dependent_res, Failure):
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
                            reason=(
                                "dependent-memory lookup failed: "
                                f"{dependent_res.failure().code}"
                            ),
                            evidence=evidence,
                            **self._proposal_trace(proposal),
                        )
                    )
                    forget_message = (
                        "Nie mogłem bezpiecznie sprawdzić zależności pamięci; "
                        "niczego nie usunąłem."
                    )
                    continue
                targets = tuple(dict.fromkeys((*targets, *dependent_res.unwrap())))
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
                            f"matched memory query with dependent cascade: {query_text}"
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
            if identity not in seen and len(merged) < MAX_PROPOSALS_PER_TURN:
                merged.append(proposal)
                seen.add(identity)

        dropped_by_limit = max(
            0,
            len(
                {
                    (proposal.topic, proposal.span_start, proposal.span_end)
                    for proposal in (*candidate_proposals, *model_proposals)
                }
            )
            - len(merged),
        )

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
            reason=(
                f"schema-valid candidates: {len(model_proposals)}; "
                f"merged accepted: {len(merged)}; "
                f"dropped by shared limit: {dropped_by_limit}"
            ),
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
        lock = await self._retain_turn_lock(turn.record_id)
        try:
            async with lock:
                return await self._accept_turn_locked(
                    turn, cancellation or CancellationToken(), req_id
                )
        finally:
            await self._release_turn_lock(turn.record_id)

    async def _synthesize_action_candidate(  # noqa: PLR0913
        self,
        turn: ConversationTurn,
        request_id: str,
        context_package: AssistantContextPackage,
        manifest: AssistantContextManifest,
        inference_req: InferenceRequest,
        action_candidate: AssistantCandidate,
        token: CancellationToken,
        preamble: str | None = None,
        depth: int = 0,
    ) -> Result[tuple[AssistantCandidate, AssistantContextManifest, float], ActionFailure]:
        action_result = action_candidate.metadata.get("action_result") or {}
        action_proposal = action_candidate.metadata.get("action_proposal") or {}
        output = (
            action_result.get("output")
            if isinstance(action_result, dict)
            else {}
        )
        capability = (
            (action_proposal.get("capability") if isinstance(action_proposal, dict) else None)
            or (action_result.get("capability") if isinstance(action_result, dict) else None)
            or "tool"
        )
        summary = (
            (output.get("title") if isinstance(output, dict) else None)
            or (action_candidate.text.splitlines()[0] if action_candidate.text else "")
            or str(capability)
        )
        obs_content = dict(output) if isinstance(output, dict) else {"raw_text": str(output)}
        if action_candidate.text and "formatted_text" not in obs_content:
            obs_content["formatted_text"] = action_candidate.text
        obs = ToolObservation(
            source_id=(
                (action_result.get("request_id") if isinstance(action_result, dict) else None)
                or _new_id()
            ),
            source_turn_id=turn.record_id,
            capability=str(capability),
            status="SUCCESS",
            summary=str(summary),
            untrusted_content=bool(action_candidate.metadata.get("untrusted_content", False)),
            content=obs_content,
        )
        existing_obs = list(context_package.content.get("tool_observations", []))
        existing_obs.append(obs.model_dump(mode="json"))
        if len(existing_obs) > _MAX_SYNTHESIS_TOOL_OBSERVATIONS:
            existing_obs = existing_obs[-_MAX_SYNTHESIS_TOOL_OBSERVATIONS:]
        updated_content = dict(context_package.content)
        updated_content["tool_observations"] = existing_obs

        max_chars = getattr(self.context_builder, "max_context_characters", 8_000)
        synth_serialized = json.dumps(updated_content, ensure_ascii=False)
        if len(synth_serialized) > max_chars:
            synth_ext = list(updated_content.get("external_evidence", []))
            while len(synth_serialized) > max_chars and synth_ext:
                synth_ext.pop()
                updated_content["external_evidence"] = synth_ext
                synth_serialized = json.dumps(updated_content, ensure_ascii=False)
            while len(synth_serialized) > max_chars and len(existing_obs) > 1:
                existing_obs.pop(0)
                updated_content["tool_observations"] = existing_obs
                synth_serialized = json.dumps(updated_content, ensure_ascii=False)

        updated_manifest = manifest.model_copy(
            update={"tool_observation_ids": (*manifest.tool_observation_ids, obs.source_id)}
        )
        synth_context = context_package.model_copy(
            update={"content": updated_content, "manifest": updated_manifest}
        )
        synth_inference_req = inference_req.model_copy(update={"context": synth_context})
        synth_total_chars = len(json.dumps(updated_content, ensure_ascii=False))
        trace(
            "assistant.model.synthesize_budget",
            turn_id=turn.record_id,
            total_characters=synth_total_chars,
            estimated_tokens=int(synth_total_chars / 3.5),
        )
        trace_payload(
            "assistant.model.synthesize_context",
            updated_content,
            turn_id=turn.record_id,
            capability=str(capability),
        )

        trace("assistant.model.synthesize_tool_start", turn_id=turn.record_id)
        start_time = time.perf_counter()
        backend_res = await self.backend.generate(synth_inference_req, token)
        latency_ms = (time.perf_counter() - start_time) * 1000
        trace(
            "assistant.model.synthesize_tool_end",
            turn_id=turn.record_id,
            elapsed_ms=round(latency_ms),
            status="failure" if isinstance(backend_res, Failure) else "success",
        )
        if isinstance(backend_res, Failure):
            return Failure(backend_res.failure())

        synth_candidate = backend_res.unwrap()

        if (
            depth < 1
            and self.actions is not None
            and (next_tool := extract_model_tool_call(synth_candidate.text))
        ):
            _, next_cap, next_args = next_tool
            next_intent = intent_from_tool_call(turn, next_cap, next_args)
            if next_intent is not None:
                next_action_cand = await self.actions.dispatch_intent(turn, next_intent, token)
                if next_action_cand is not None:
                    return await self._synthesize_action_candidate(
                        turn=turn,
                        request_id=request_id,
                        context_package=synth_context,
                        manifest=updated_manifest,
                        inference_req=synth_inference_req,
                        action_candidate=next_action_cand,
                        token=token,
                        preamble=preamble,
                        depth=depth + 1,
                    )

        delivered_text = synth_candidate.text
        if preamble and not delivered_text.startswith(preamble[:30]):
            delivered_text = f"{preamble}\n\n{delivered_text}"

        candidate = AssistantCandidate(
            text=delivered_text,
            proposals=synth_candidate.proposals,
            tokens_in=synth_candidate.tokens_in,
            tokens_out=synth_candidate.tokens_out,
            metadata={
                **action_candidate.metadata,
                **synth_candidate.metadata,
                "synthesized_from_action": True,
            },
        )
        return Success((candidate, updated_manifest, latency_ms))

    async def _accept_turn_locked(  # noqa: PLR0911, PLR0912, PLR0915
        self,
        turn: ConversationTurn,
        token: CancellationToken,
        request_id: str,
    ) -> Result[AssistantResponse, ActionFailure]:
        turn_query = MemoryQueryResolver.resolve(
            turn.text, profile_scope=self.profile_scope
        )
        turn_domain = next(
            (domain for domain in turn_query.domain_scopes if domain != "global"),
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
            if existing.user_turn_id != turn.record_id:
                return Failure(
                    make_assistant_failure(
                        code="ID_CONFLICT",
                        message=(f"request id {request_id} belongs to another turn"),
                        request_id=request_id,
                    )
                )
            return Success(existing)

        turn_response_res = await self.store.get_response_by_turn_id(turn.record_id)
        if isinstance(turn_response_res, Failure):
            return Failure(turn_response_res.failure())
        turn_response = turn_response_res.unwrap()
        if turn_response is not None:
            return Success(turn_response)

        stored_turn_res = await self.store.get_turn(turn.record_id)
        if isinstance(stored_turn_res, Failure):
            return Failure(stored_turn_res.failure())
        stored_turn = stored_turn_res.unwrap()
        if stored_turn is not None:
            immutable_input_matches = (
                stored_turn.session_id == turn.session_id
                and stored_turn.role == turn.role
                and stored_turn.text == turn.text
                and stored_turn.data_class == turn.data_class
                and stored_turn.timestamp == turn.timestamp
                and stored_turn.producer == turn.producer
                and stored_turn.correlation_id == turn.correlation_id
                and stored_turn.domain_scope == turn.domain_scope
                and stored_turn.purpose == turn.purpose
                and stored_turn.metadata == turn.metadata
                and (
                    turn.reply_to_turn_id is None
                    or stored_turn.reply_to_turn_id == turn.reply_to_turn_id
                )
            )
            if not immutable_input_matches:
                return Failure(
                    make_assistant_failure(
                        code="ID_CONFLICT",
                        message=(
                            f"turn id {turn.record_id} exists with differing content"
                        ),
                        request_id=turn.record_id,
                    )
                )
            turn = stored_turn

        if stored_turn is None and turn.reply_to_turn_id is None:
            chain_res = await self.store.get_recent_reply_chain(
                session_id=turn.session_id, limit=1
            )
            if isinstance(chain_res, Failure):
                return Failure(chain_res.failure())
            chain = chain_res.unwrap()
            if chain:
                turn = turn.model_copy(update={"reply_to_turn_id": chain[-1].record_id})
        elif stored_turn is None:
            parent_res = await self.store.get_turn(turn.reply_to_turn_id)
            if isinstance(parent_res, Failure):
                return Failure(parent_res.failure())
            parent = parent_res.unwrap()
            if parent is None or parent.session_id != turn.session_id:
                return Failure(
                    make_assistant_failure(
                        code="INVALID_REPLY_TARGET",
                        message=(
                            "reply_to_turn_id must identify a turn in the same session"
                        ),
                        request_id=turn.record_id,
                    )
                )

        accepted_res = await self.store.accept_turn(turn)
        if isinstance(accepted_res, Failure):
            return Failure(accepted_res.failure())

        trace("assistant.context.start", turn_id=turn.record_id)
        ctx_res = await self.context_builder.build_context(
            turn, is_remote=self.backend.is_remote
        )
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
        if self.context_approver is not None:
            approved = await self.context_approver(context_package, token)
            if isinstance(approved, Failure):
                return await self._commit_failure(
                    turn=turn,
                    request_id=request_id,
                    manifest=context_package.manifest,
                    error=approved.failure(),
                    latency_ms=0,
                )
            context_package = approved.unwrap()
        manifest = context_package.manifest
        total_ctx_chars = len(json.dumps(context_package.content, ensure_ascii=False))
        trace("assistant.context.end", turn_id=turn.record_id,
              external_evidence=len(context_package.content.get("external_evidence", ())),
              routing=manifest.routing_decision,
              total_characters=total_ctx_chars,
              estimated_tokens=int(total_ctx_chars / 3.5))
        trace_payload(
            "assistant.context.package",
            {
                "manifest": manifest.model_dump(mode="json"),
                "summary": {
                    "total_characters": total_ctx_chars,
                    "estimated_tokens": int(total_ctx_chars / 3.5),
                    "recent_turns_count": len(context_package.content.get("recent_turns", ())),
                    "external_evidence_count": len(context_package.content.get("external_evidence", ())),
                    "tool_observations_count": len(context_package.content.get("tool_observations", ())),
                    "durable_memories_count": len(context_package.content.get("durable_memories", ())),
                },
                "content": context_package.content,
            },
            turn_id=turn.record_id,
        )

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
            system_instruction=self.context_builder.system_instruction,
            model_name=str(getattr(self.backend, "model_name", "deterministic")),
        )

        action_candidate = (
            await self.actions.handle(turn, token) if self.actions is not None else None
        )
        requires_synthesis = (
            action_candidate is not None
            and (
                action_candidate.metadata.get("requires_synthesis") is True
                or (
                    action_candidate.metadata.get("untrusted_content") is True
                    and not action_candidate.metadata.get("active_choice_frame")
                )
            )
        )
        if requires_synthesis and action_candidate is not None:
            synth_res = await self._synthesize_action_candidate(
                turn=turn,
                request_id=request_id,
                context_package=context_package,
                manifest=manifest,
                inference_req=inference_req,
                action_candidate=action_candidate,
                token=token,
            )
            if isinstance(synth_res, Failure):
                return await self._commit_failure(
                    turn=turn,
                    request_id=request_id,
                    manifest=manifest,
                    error=synth_res.failure(),
                    latency_ms=0.0,
                )
            candidate, manifest, latency_ms = synth_res.unwrap()
        elif action_candidate is not None:
            candidate = action_candidate
            latency_ms = 0.0
        elif manifest.evidence_required and manifest.routing_decision == "no_evidence":
            candidate = AssistantCandidate(
                text=_MEMORY_ABSTENTION_TEXT,
                metadata={
                    "deterministic_abstention": True,
                    "reason": "memory_evidence_required_but_unavailable",
                },
            )
            latency_ms = 0.0
        else:
            trace("assistant.model.start", turn_id=turn.record_id)
            start_time = time.perf_counter()
            backend_res = await self.backend.generate(inference_req, token)
            latency_ms = (time.perf_counter() - start_time) * 1000
            trace(
                "assistant.model.end",
                turn_id=turn.record_id,
                elapsed_ms=round(latency_ms),
                status="failure" if isinstance(backend_res, Failure) else "success",
            )
            if isinstance(backend_res, Failure):
                return await self._commit_failure(
                    turn=turn,
                    request_id=request_id,
                    manifest=manifest,
                    error=backend_res.failure(),
                    latency_ms=latency_ms,
                )
            candidate = backend_res.unwrap()
            if self.actions is not None and (tool_call := extract_model_tool_call(candidate.text)):
                preamble, cap_name, tool_args = tool_call
                tool_intent = intent_from_tool_call(turn, cap_name, tool_args)
                if tool_intent is not None:
                    tool_cand = await self.actions.dispatch_intent(turn, tool_intent, token)
                    if tool_cand is not None:
                        synth_res = await self._synthesize_action_candidate(
                            turn=turn,
                            request_id=request_id,
                            context_package=context_package,
                            manifest=manifest,
                            inference_req=inference_req,
                            action_candidate=tool_cand,
                            token=token,
                            preamble=preamble,
                        )
                        if isinstance(synth_res, Success):
                            candidate, manifest, synth_lat = synth_res.unwrap()
                            latency_ms += synth_lat
                            action_candidate = tool_cand
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
        assistant_turn_metadata: dict[str, object] = {"profile_scope": self.profile_scope}
        if action_candidate is not None:
            assistant_turn_metadata["action"] = candidate.metadata
        if reasoning_content := candidate.metadata.get("reasoning_content"):
            assistant_turn_metadata["reasoning_content"] = reasoning_content

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
            metadata=assistant_turn_metadata,
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
            reasoning_content=candidate.metadata.get("reasoning_content"),
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

    async def run_research_workflow(
        self,
        query: str,
        session_id: str,
        capabilities: CapabilityService,
        cancellation: CancellationToken | None = None,
        task_id: str | None = None,
    ) -> Result[ResearchWorkflow, ActionFailure]:
        """Run an end-to-end bounded research task using step_controller and backend synthesis."""
        if self.step_controller is None:
            return Failure(
                ActionFailure(
                    request_id=f"research:{session_id}",
                    capability="assistant.research",
                    code="CONTROLLER_NOT_CONFIGURED",
                    message="TaskStepController is not configured on AssistantService",
                    producer=self.producer,
                )
            )
        token = cancellation or CancellationToken()
        t_id = task_id or f"task:{session_id}:{_new_id()}"

        async def _synthesizer(
            wf: ResearchWorkflow, ct: CancellationToken
        ) -> Result[str, ActionFailure]:
            obs_list = [
                ToolObservation(
                    source_id=f"{wf.task_id}:obs:{idx}",
                    source_turn_id=session_id,
                    capability="browser.read_page",
                    status="SUCCESS",
                    summary=page.get("title", f"Page {idx}"),
                    untrusted_content=True,
                    content=page,
                ).model_dump(mode="json")
                for idx, page in enumerate(wf.read_pages, 1)
            ]
            manifest = AssistantContextManifest(
                record_id=_new_id(),
                timestamp=_utc_now(),
                producer=self.producer,
                session_id=session_id,
                turn_id=session_id,
                tool_observation_ids=tuple(
                    f"{wf.task_id}:obs:{idx}" for idx in range(1, len(obs_list) + 1)
                ),
            )
            context = AssistantContextPackage(
                record_id=_new_id(),
                timestamp=_utc_now(),
                producer=self.producer,
                session_id=session_id,
                turn_id=session_id,
                manifest=manifest,
                content={
                    "current_turn": {"text": query},
                    "tool_observations": obs_list,
                },
            )
            req = InferenceRequest(
                record_id=_new_id(),
                timestamp=_utc_now(),
                producer=self.producer,
                session_id=session_id,
                turn_id=session_id,
                request_id=f"{t_id}:synthesis",
                context=context,
                budget=self._budget(),
                strategy="DIRECT",
            )
            gen_res = await self.backend.generate(req, ct)
            if isinstance(gen_res, Failure):
                return Failure(gen_res.failure())
            return Success(gen_res.unwrap().text)

        return await self.step_controller.run_bounded_research(
            task_id=t_id,
            session_id=session_id,
            query=query,
            capabilities=capabilities,
            synthesizer=_synthesizer,
            cancellation=token,
            actor=self.producer,
        )
