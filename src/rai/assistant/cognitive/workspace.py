"""WorkspaceEngine implementing Global Workspace Theory (GWT) cognitive deliberation."""

from __future__ import annotations

import asyncio
import time
from typing import Any

from returns.result import Failure, Result, Success

from rai.actions.workflow import (
    _compact_observation_content,
    _should_synthesize_action_candidate,
)
from rai.assistant.ports import (
    AssistantActionHandler,
    AssistantModelBackend,
    InteractionController,
)
from rai.assistant.records import (
    AssistantCandidate,
    AssistantContextManifest,
    AssistantContextPackage,
    ConversationTurn,
    MemoryProposal,
)
from rai.assistant.tool_calling import intent_from_tool_call
from rai.diagnostics import trace
from rai.kernel.ports import CancellationToken
from rai.kernel.records import (
    ActionFailure,
    InferenceBudget,
    ProducerIdentity,
    _new_id,
    _utc_now,
)
from rai.kernel.service import CapabilityService
from rai.kernel.transport import normalize_request
from rai.skills.models import SkillRegistry

from .arbiter import SalienceArbiter
from .models import WorkspaceContribution, WorkspaceContributionKind, WorkspaceState
from .probe import JLensProbe
from .specialists import (
    _MIN_REASONABLE_INPUT_TOKENS,
    ActionReasonerSpecialist,
    AnswerDraftSpecialist,
    CognitiveSpecialist,
    GroundingEvaluationSpecialist,
    RetrieverSpecialist,
)

PRODUCER = ProducerIdentity(producer_id="workspace-engine", kind="controller", version="1.0.0")
_DEFAULT_MAX_CYCLES: int = 4
_MAX_CONSECUTIVE_CRITIQUES: int = 2
_MEMORY_ABSTENTION_TEXT: str = (
    "Nie mam wystarczających, dopuszczonych dowodów w pamięci, aby odpowiedzieć."
)


class WorkspaceEngine(InteractionController):
    """Global Workspace Theory deliberative interaction controller."""

    def __init__(  # noqa: PLR0913
        self,
        specialists: tuple[CognitiveSpecialist, ...] | None = None,
        arbiter: SalienceArbiter | None = None,
        capabilities: CapabilityService | None = None,
        actions: AssistantActionHandler | None = None,
        backend: AssistantModelBackend | None = None,
        skills: SkillRegistry | None = None,
        probe: JLensProbe | None = None,
        context_builder: object = None,
        max_cycles: int | None = _DEFAULT_MAX_CYCLES,
        store: object = None,
        observation_retention: str = "session",
    ) -> None:
        self.capabilities = capabilities
        self.actions = actions
        self.backend = backend
        self.skills = skills
        self.probe = probe or JLensProbe()
        self.context_builder = context_builder
        self.max_cycles = max_cycles
        self.arbiter = arbiter or SalienceArbiter()
        self.store = store or getattr(context_builder, "store", None)
        self.observation_retention = observation_retention

        if specialists is not None:
            self.specialists = specialists
        else:
            sys_inst = getattr(self.context_builder, "system_instruction", None)
            self.specialists = (
                RetrieverSpecialist(),
                ActionReasonerSpecialist(
                    backend=backend,
                    capabilities=capabilities,
                    skills=skills,
                ),
                AnswerDraftSpecialist(
                    backend=backend,
                    system_instruction=sys_inst,
                ),
                GroundingEvaluationSpecialist(),
            )

    async def _execute_tool_proposal(  # noqa: PLR0913
        self,
        turn: ConversationTurn,
        workspace: WorkspaceState,
        winner: WorkspaceContribution,
        token: CancellationToken,
        obs_ids: list[str],
        action_metadata: dict[str, Any],
        unexecuted_proposals: list[dict[str, Any]],
    ) -> WorkspaceState:
        """Execute or record a winning TOOL_CALL proposal outside specialist modules."""
        cap_name = str(winner.payload.get("capability", ""))
        args = dict(winner.payload.get("arguments", {}))
        req_id = f"{workspace.turn_id}:workspace_step:{workspace.cycle + 1}"

        # 1. Try high-level AssistantActionHandler (handles ActiveChoiceFrame & handle resolution)
        if self.actions is not None and hasattr(self.actions, "dispatch_intent"):
            tool_intent = intent_from_tool_call(turn, cap_name, args)
            if tool_intent is not None:
                step_turn = (
                    turn.model_copy(
                        update={
                            "metadata": {
                                **turn.metadata,
                                "in_turn_action": dict(action_metadata),
                            }
                        }
                    )
                    if action_metadata
                    else turn
                )
                tool_cand = await self.actions.dispatch_intent(
                    step_turn, tool_intent, token
                )
                if tool_cand is not None:
                    action_metadata.update(tool_cand.metadata)
                    act_res_dict = tool_cand.metadata.get("action_result") or {}
                    raw_out = (
                        act_res_dict.get("output")
                        if isinstance(act_res_dict, dict)
                        else {"raw_text": tool_cand.text}
                    )
                    from rai.assistant.observation_budget import persist_observation_artifact  # noqa: PLC0415
                    artifact_id = (
                        act_res_dict.get("_artifact_id")
                        if isinstance(act_res_dict, dict)
                        else None
                    )
                    if not artifact_id and self.store is not None:
                        artifact_id = await persist_observation_artifact(
                            store=self.store,
                            session_id=turn.session_id,
                            turn_id=turn.turn_id,
                            capability=cap_name,
                            request_id=str(req_id),
                            data_class=turn.data_class,
                            output=raw_out,
                            producer=PRODUCER,
                            retention=self.observation_retention,
                        )
                    if artifact_id and isinstance(raw_out, dict):
                        raw_out = {**raw_out, "_artifact_id": artifact_id}
                    obs_out = _compact_observation_content(
                        raw_out,
                        fallback_text=tool_cand.text,
                        capability=cap_name,
                    )
                    obs_id = (
                        (act_res_dict.get("request_id") if isinstance(act_res_dict, dict) else None)
                        or req_id
                    )
                    obs_ids.append(str(obs_id))
                    return workspace.model_copy(
                        update={
                            "observations": (*workspace.observations, obs_out),
                            "tool_calls_count": workspace.tool_calls_count + 1,
                        }
                    )

        # 2. Try direct CapabilityService invocation
        if self.capabilities is not None:
            descriptor = self.capabilities.registry.descriptor(cap_name)
            if descriptor is not None:
                call_args = dict(args)
                if (
                    turn.session_id
                    and isinstance(descriptor.input_schema, dict)
                    and "session_id" in descriptor.input_schema.get("properties", {})
                ):
                    call_args["session_id"] = turn.session_id
                request = normalize_request(
                    descriptor,
                    call_args,
                    actor=turn.producer,
                    request_id=req_id,
                    data_class=turn.data_class,
                )
                decision, executed = await self.capabilities.invoke(request, token)
                obs_ids.append(req_id)
                if isinstance(executed, Success):
                    act_res = executed.unwrap()
                    from rai.assistant.observation_budget import persist_observation_artifact  # noqa: PLC0415
                    from rai.kernel.records import max_data_class  # noqa: PLC0415

                    effective_dc = max_data_class(
                        turn.data_class,
                        getattr(decision, "data_class", None),
                        getattr(act_res, "data_class", None),
                    )
                    derived_from = tuple(
                        p.source_id for p in act_res.provenance if hasattr(p, "source_id")
                    )
                    artifact_id = await persist_observation_artifact(
                        store=self.store,
                        session_id=turn.session_id,
                        turn_id=getattr(turn, "turn_id", getattr(turn, "record_id", "")),
                        capability=cap_name,
                        request_id=act_res.request_id,
                        data_class=effective_dc,
                        output=act_res.output,
                        producer=act_res.producer,
                        derived_from=derived_from,
                        retention=self.observation_retention,
                    )
                    out_for_compact = act_res.output
                    if artifact_id and isinstance(out_for_compact, dict):
                        out_for_compact = {**out_for_compact, "_artifact_id": artifact_id}
                    compacted = _compact_observation_content(
                        out_for_compact,
                        capability=cap_name,
                    )
                    return workspace.model_copy(
                        update={
                            "observations": (*workspace.observations, compacted),
                            "tool_calls_count": workspace.tool_calls_count + 1,
                        }
                    )
                err = executed.failure()
                return workspace.model_copy(
                    update={
                        "observations": (
                            *workspace.observations,
                            {
                                "error": err.message,
                                "code": err.code,
                                "capability": cap_name,
                            },
                        ),
                        "tool_calls_count": workspace.tool_calls_count + 1,
                    }
                )

        # 3. Fallback: emit unexecuted proposal record when no execution boundary is attached
        unexecuted_proposals.append(
            {
                "record_id": _new_id(),
                "record_type": "workspace_action_proposal",
                "source_turn_id": turn.record_id,
                "task_id": f"workspace:{turn.record_id}",
                "capability": cap_name or "unknown",
                "arguments": args,
                "data_class": str(turn.data_class),
            }
        )
        return workspace.model_copy(
            update={
                "observations": (
                    *workspace.observations,
                    {
                        "status": "PROPOSED_NOT_EXECUTED",
                        "capability": cap_name,
                        "arguments": args,
                    },
                ),
                "tool_calls_count": workspace.tool_calls_count + 1,
            }
        )

    async def process_turn(  # noqa: PLR0911, PLR0912, PLR0915
        self,
        turn: ConversationTurn,
        context: AssistantContextPackage,
        budget: InferenceBudget,
        token: CancellationToken,
    ) -> Result[AssistantCandidate, ActionFailure]:
        """Deliberate over multiple specialist competition cycles and produce final candidate."""
        manifest = context.manifest
        initial_observations: list[Any] = []
        obs_ids: list[str] = list(manifest.tool_observation_ids)
        action_metadata: dict[str, Any] = {}
        unexecuted_proposals: list[dict[str, Any]] = []

        if self.actions is not None:
            action_candidate = await self.actions.handle(turn, token)
            if action_candidate is not None:
                requires_synthesis = _should_synthesize_action_candidate(action_candidate)
                if not requires_synthesis:
                    return Success(
                        action_candidate.model_copy(
                            update={
                                "metadata": {
                                    **action_candidate.metadata,
                                    "updated_manifest": manifest,
                                    "latency_ms": 0.0,
                                }
                            }
                        )
                    )
                action_metadata = dict(action_candidate.metadata)
                action_result = action_candidate.metadata.get("action_result") or {}
                act_prop = action_candidate.metadata.get("action_proposal") or {}
                output = (
                    action_result.get("output") if isinstance(action_result, dict) else {}
                )
                capability = (
                    (act_prop.get("capability") if isinstance(act_prop, dict) else None)
                    or (action_result.get("capability") if isinstance(action_result, dict) else None)
                    or "tool"
                )
                obs_content = _compact_observation_content(
                    output,
                    fallback_text=action_candidate.text,
                    capability=str(capability),
                )
                obs_id = (
                    (
                        action_result.get("request_id")
                        if isinstance(action_result, dict)
                        else None
                    )
                    or _new_id()
                )
                obs_ids.append(str(obs_id))
                initial_observations.append(obs_content)

        if (
            not initial_observations
            and manifest.evidence_required
            and manifest.routing_decision == "no_evidence"
        ):
            return Success(
                AssistantCandidate(
                    text=_MEMORY_ABSTENTION_TEXT,
                    metadata={
                        "deterministic_abstention": True,
                        "reason": "memory_evidence_required_but_unavailable",
                        "updated_manifest": manifest,
                        "latency_ms": 0.0,
                    },
                )
            )

        start_time = time.perf_counter()
        if self.max_cycles is not None and budget.max_agent_turns > 0:
            effective_max_cycles = min(self.max_cycles, budget.max_agent_turns)
        elif self.max_cycles is not None:
            effective_max_cycles = self.max_cycles
        elif budget.max_agent_turns > 0:
            effective_max_cycles = budget.max_agent_turns
        else:
            effective_max_cycles = _DEFAULT_MAX_CYCLES
        workspace = WorkspaceState(
            cycle=0,
            max_cycles=effective_max_cycles,
            budget=budget,
            goal=turn.text,
            session_id=turn.session_id,
            turn_id=turn.record_id,
            context=context,
            observations=tuple(initial_observations),
            metadata={
                "evidence_required": manifest.evidence_required,
                "routing_decision": manifest.routing_decision,
            },
        )

        total_tokens_in = 0
        total_tokens_out = 0
        last_failure: ActionFailure | None = None
        consecutive_critiques = 0
        last_draft_text: str | None = None

        while not workspace.converged and (
            workspace.max_cycles is not None and workspace.cycle < workspace.max_cycles
        ):
            if token.cancelled:
                return Failure(
                    ActionFailure(
                        record_id=_new_id(),
                        timestamp=_utc_now(),
                        producer=PRODUCER,
                        request_id=turn.record_id,
                        capability="workspace_engine",
                        code="CANCELLED",
                        message="Deliberation cancelled by token",
                    )
                )

            elapsed_sec = time.perf_counter() - start_time
            if (budget.max_latency_seconds > 0 and elapsed_sec >= budget.max_latency_seconds) or (
                budget.cancellation_deadline is not None and _utc_now() >= budget.cancellation_deadline
            ):
                if any(
                    c.kind == WorkspaceContributionKind.ANSWER_DRAFT and bool(c.content.strip())
                    for c in workspace.history
                ):
                    break
                return Failure(
                    ActionFailure(
                        record_id=_new_id(),
                        timestamp=_utc_now(),
                        producer=PRODUCER,
                        request_id=turn.record_id,
                        capability="workspace_engine",
                        code="DEADLINE_EXCEEDED",
                        message=f"Deliberation exceeded latency budget ({elapsed_sec:.2f}s >= {budget.max_latency_seconds}s)",
                    )
                )

            if (budget.max_input_tokens > 0 and total_tokens_in >= budget.max_input_tokens) or (
                budget.max_output_tokens > 0 and total_tokens_out >= budget.max_output_tokens
            ):
                if any(
                    c.kind == WorkspaceContributionKind.ANSWER_DRAFT and bool(c.content.strip())
                    for c in workspace.history
                ):
                    break
                return Failure(
                    ActionFailure(
                        record_id=_new_id(),
                        timestamp=_utc_now(),
                        producer=PRODUCER,
                        request_id=turn.record_id,
                        capability="workspace_engine",
                        code="BUDGET_EXCEEDED",
                        message=f"Deliberation exceeded token budget (in={total_tokens_in}/{budget.max_input_tokens}, out={total_tokens_out}/{budget.max_output_tokens})",
                    )
                )

            rem_in = max(0, budget.max_input_tokens - total_tokens_in) if budget.max_input_tokens > 0 else 0
            rem_out = max(0, budget.max_output_tokens - total_tokens_out) if budget.max_output_tokens > 0 else 0
            rem_lat = max(1, int(budget.max_latency_seconds - elapsed_sec)) if budget.max_latency_seconds > 0 else 30
            cycle_budget = budget.model_copy(
                update={
                    "max_input_tokens": rem_in,
                    "max_output_tokens": rem_out,
                    "max_latency_seconds": rem_lat,
                }
            )
            workspace = workspace.model_copy(update={"budget": cycle_budget})

            trace(
                "assistant.workspace.cycle",
                turn_id=turn.record_id,
                cycle=workspace.cycle + 1,
                observations=len(workspace.observations),
            )

            # 1. Evaluate non-draft specialists first so we do not waste an LLM draft call
            # in a cycle where ActionReasonerSpecialist proposes a high-salience TOOL_CALL.
            non_draft_specs = [
                s for s in self.specialists if not isinstance(s, AnswerDraftSpecialist)
            ]
            draft_specs = [
                s for s in self.specialists if isinstance(s, AnswerDraftSpecialist)
            ]

            proposals: list[WorkspaceContribution] = []
            if non_draft_specs:
                non_draft_results = await asyncio.gather(
                    *(s.evaluate(workspace, token) for s in non_draft_specs)
                )
                for res in non_draft_results:
                    if isinstance(res, Failure):
                        last_failure = res.failure()
                        continue
                    contrib = res.unwrap()
                    if contrib is not None:
                        proposals.append(contrib)
                        total_tokens_in += contrib.tokens_in
                        total_tokens_out += contrib.tokens_out

            has_winning_tool_proposal = any(
                p.kind == WorkspaceContributionKind.TOOL_CALL
                and p.salience >= self.arbiter.convergence_threshold
                for p in proposals
            )

            if not has_winning_tool_proposal and draft_specs:
                draft_results = await asyncio.gather(
                    *(s.evaluate(workspace, token) for s in draft_specs)
                )
                for res in draft_results:
                    if isinstance(res, Failure):
                        last_failure = res.failure()
                        continue
                    contrib = res.unwrap()
                    if contrib is not None:
                        proposals.append(contrib)
                        total_tokens_in += contrib.tokens_in
                        total_tokens_out += contrib.tokens_out

            # 2. Select winner via salience arbiter
            winner = self.arbiter.select_winner(proposals, workspace)
            if winner is None:
                if last_failure is not None and not workspace.history:
                    return Failure(last_failure)
                break

            if self.probe.is_available:
                drift_res = await self.probe.evaluate_drift(workspace, winner)
                if isinstance(drift_res, Success) and self.probe.is_divergent(drift_res.unwrap()):
                    latency_ms = (time.perf_counter() - start_time) * 1000
                    updated_manifest: AssistantContextManifest = manifest.model_copy(
                        update={"tool_observation_ids": tuple(obs_ids)}
                    )
                    return Success(
                        AssistantCandidate(
                            text=(
                                "I detected conceptual drift in evaluating your request. "
                                "Could you clarify your intent?"
                            ),
                            metadata={
                                "latent_drift_detected": True,
                                "drift_score": drift_res.unwrap(),
                                "workspace_cycles": workspace.cycle + 1,
                                "updated_manifest": updated_manifest,
                                "latency_ms": latency_ms,
                            },
                        )
                    )

            # 3. Record any auxiliary retrieval notes alongside the winning broadcast in history
            aux_retrievals = tuple(
                p
                for p in proposals
                if p is not winner and p.kind == WorkspaceContributionKind.RETRIEVAL
            )
            workspace = workspace.model_copy(
                update={
                    "broadcast": winner,
                    "history": (*workspace.history, *aux_retrievals, winner),
                }
            )

            # 4. Handle winning action
            if winner.kind == WorkspaceContributionKind.TOOL_CALL:
                workspace = await self._execute_tool_proposal(
                    turn=turn,
                    workspace=workspace,
                    winner=winner,
                    token=token,
                    obs_ids=obs_ids,
                    action_metadata=action_metadata,
                    unexecuted_proposals=unexecuted_proposals,
                )

            elif winner.kind == WorkspaceContributionKind.ANSWER_DRAFT:
                if not winner.content.strip():
                    workspace = workspace.model_copy(update={"cycle": workspace.cycle + 1})
                    continue

                # Allow grounding specialists to inspect the broadcast draft before converging
                grounding_critique: WorkspaceContribution | None = None
                for spec in self.specialists:
                    if isinstance(spec, GroundingEvaluationSpecialist):
                        g_res = await spec.evaluate(workspace, token)
                        if isinstance(g_res, Success) and g_res.unwrap() is not None:
                            grounding_critique = g_res.unwrap()
                            break

                if grounding_critique is not None:
                    consecutive_critiques += 1
                    if (
                        winner.content == last_draft_text
                        or consecutive_critiques > _MAX_CONSECUTIVE_CRITIQUES
                    ):
                        grounding_critique = None

                if grounding_critique is not None:
                    last_draft_text = winner.content
                    workspace = workspace.model_copy(
                        update={
                            "broadcast": grounding_critique,
                            "history": (*workspace.history, grounding_critique),
                        }
                    )
                else:
                    is_last_cycle = (
                        workspace.max_cycles is not None
                        and workspace.cycle >= workspace.max_cycles - 1
                    )
                    if winner.salience >= self.arbiter.convergence_threshold or is_last_cycle:
                        latency_ms = (time.perf_counter() - start_time) * 1000
                        raw_cand = winner.payload.get("candidate")
                        cand_proposals: tuple[MemoryProposal, ...] = ()
                        cand_meta: dict[str, Any] = {}
                        if isinstance(raw_cand, dict):
                            if raw_cand.get("proposals"):
                                cand_proposals = tuple(
                                    MemoryProposal.model_validate(p)
                                    for p in raw_cand["proposals"]
                                )
                            if isinstance(raw_cand.get("metadata"), dict):
                                cand_meta = dict(raw_cand["metadata"])
                        updated_manifest = manifest.model_copy(
                            update={"tool_observation_ids": tuple(obs_ids)}
                        )
                        extra_meta: dict[str, Any] = {}
                        if unexecuted_proposals:
                            extra_meta["action_proposals"] = list(unexecuted_proposals)
                        if initial_observations or workspace.tool_calls_count > 0:
                            extra_meta["synthesized_from_action"] = True
                        final_candidate = AssistantCandidate(
                            text=winner.content,
                            proposals=cand_proposals,
                            tokens_in=total_tokens_in,
                            tokens_out=total_tokens_out,
                            metadata={
                                **action_metadata,
                                **cand_meta,
                                **extra_meta,
                                "workspace_cycles": workspace.cycle + 1,
                                "workspace_converged": True,
                                "winning_specialist": winner.specialist_name,
                                "tool_calls_executed": workspace.tool_calls_count,
                                "observations_count": len(workspace.observations),
                                "updated_manifest": updated_manifest,
                                "latency_ms": latency_ms,
                            },
                        )
                        return Success(final_candidate)

            workspace = workspace.model_copy(update={"cycle": workspace.cycle + 1})

        # Fallback if loop exited without explicit answer draft convergence
        if last_failure is not None and not workspace.history:
            return Failure(last_failure)

        draft_candidate = next(
            (
                c
                for c in reversed(workspace.history)
                if c.kind == WorkspaceContributionKind.ANSWER_DRAFT and bool(c.content.strip())
            ),
            None,
        )
        if draft_candidate is None:
            if budget.max_input_tokens > 0 and (
                total_tokens_in >= budget.max_input_tokens
                or budget.max_input_tokens < _MIN_REASONABLE_INPUT_TOKENS
            ):
                return Failure(
                    ActionFailure(
                        record_id=_new_id(),
                        timestamp=_utc_now(),
                        producer=PRODUCER,
                        request_id=turn.record_id,
                        capability="workspace_engine",
                        code="BUDGET_EXCEEDED",
                        message="Token budget exhausted before candidate generation",
                    )
                )
            if (
                (
                    budget.max_latency_seconds > 0
                    and (time.perf_counter() - start_time) >= budget.max_latency_seconds
                )
                or (
                    budget.cancellation_deadline is not None
                    and _utc_now() >= budget.cancellation_deadline
                )
            ):
                return Failure(
                    ActionFailure(
                        record_id=_new_id(),
                        timestamp=_utc_now(),
                        producer=PRODUCER,
                        request_id=turn.record_id,
                        capability="workspace_engine",
                        code="DEADLINE_EXCEEDED",
                        message="Deliberation deadline exceeded without candidate draft",
                    )
                )

        latency_ms = (time.perf_counter() - start_time) * 1000
        final_text = (
            draft_candidate.content
            if draft_candidate
            else f"I have processed your request: {turn.text}"
        )
        updated_manifest = manifest.model_copy(
            update={"tool_observation_ids": tuple(obs_ids)}
        )
        extra_meta = {}
        if unexecuted_proposals:
            extra_meta["action_proposals"] = list(unexecuted_proposals)

        return Success(
            AssistantCandidate(
                text=final_text,
                tokens_in=total_tokens_in,
                tokens_out=total_tokens_out,
                metadata={
                    **action_metadata,
                    **extra_meta,
                    "workspace_cycles": workspace.cycle,
                    "workspace_converged": False,
                    "tool_calls_executed": workspace.tool_calls_count,
                    "observations_count": len(workspace.observations),
                    "updated_manifest": updated_manifest,
                    "latency_ms": latency_ms,
                },
            )
        )
