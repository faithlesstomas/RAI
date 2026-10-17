"""WorkspaceEngine implementing Global Workspace Theory (GWT) cognitive deliberation."""

from __future__ import annotations

import asyncio
import time
from typing import Any

from returns.result import Failure, Result, Success

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
    ActionReasonerSpecialist,
    AnswerDraftSpecialist,
    CognitiveSpecialist,
    GroundingEvaluationSpecialist,
    RetrieverSpecialist,
)

PRODUCER = ProducerIdentity(producer_id="workspace-engine", kind="controller", version="1.0.0")
_DEFAULT_MAX_CYCLES: int = 4
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
        max_cycles: int = _DEFAULT_MAX_CYCLES,
    ) -> None:
        self.capabilities = capabilities
        self.actions = actions
        self.backend = backend
        self.skills = skills
        self.probe = probe or JLensProbe()
        self.context_builder = context_builder
        self.max_cycles = max_cycles
        self.arbiter = arbiter or SalienceArbiter()

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
                tool_cand = await self.actions.dispatch_intent(turn, tool_intent, token)
                if tool_cand is not None:
                    action_metadata.update(tool_cand.metadata)
                    act_res_dict = tool_cand.metadata.get("action_result") or {}
                    obs_out = (
                        dict(act_res_dict.get("output") or {})
                        if isinstance(act_res_dict, dict)
                        else {"raw_text": tool_cand.text}
                    )
                    if tool_cand.text and "formatted_text" not in obs_out:
                        obs_out["formatted_text"] = tool_cand.text
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
                request = normalize_request(
                    descriptor,
                    args,
                    actor=turn.producer,
                    request_id=req_id,
                    data_class=turn.data_class,
                )
                _, executed = await self.capabilities.invoke(request, token)
                obs_ids.append(req_id)
                if isinstance(executed, Success):
                    act_res = executed.unwrap()
                    return workspace.model_copy(
                        update={
                            "observations": (*workspace.observations, act_res.output),
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
                requires_synthesis = action_candidate.metadata.get(
                    "requires_synthesis"
                ) is True or (
                    action_candidate.metadata.get("untrusted_content") is True
                    and not action_candidate.metadata.get("active_choice_frame")
                )
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
                output = (
                    action_result.get("output") if isinstance(action_result, dict) else {}
                )
                obs_content = (
                    dict(output) if isinstance(output, dict) else {"raw_text": str(output)}
                )
                if action_candidate.text and "formatted_text" not in obs_content:
                    obs_content["formatted_text"] = action_candidate.text
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
        effective_max_cycles = min(self.max_cycles, budget.max_agent_turns or self.max_cycles)
        workspace = WorkspaceState(
            cycle=0,
            max_cycles=effective_max_cycles,
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

        while not workspace.converged and workspace.cycle < workspace.max_cycles:
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
                # Allow grounding specialists to inspect the broadcast draft before converging
                grounding_critique: WorkspaceContribution | None = None
                for spec in self.specialists:
                    if isinstance(spec, GroundingEvaluationSpecialist):
                        g_res = await spec.evaluate(workspace, token)
                        if isinstance(g_res, Success) and g_res.unwrap() is not None:
                            grounding_critique = g_res.unwrap()
                            break

                if grounding_critique is not None:
                    workspace = workspace.model_copy(
                        update={
                            "broadcast": grounding_critique,
                            "history": (*workspace.history, grounding_critique),
                        }
                    )
                else:
                    is_last_cycle = workspace.cycle >= workspace.max_cycles - 1
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

        latency_ms = (time.perf_counter() - start_time) * 1000
        draft_candidate = next(
            (
                c
                for c in reversed(workspace.history)
                if c.kind == WorkspaceContributionKind.ANSWER_DRAFT
            ),
            None,
        )
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
