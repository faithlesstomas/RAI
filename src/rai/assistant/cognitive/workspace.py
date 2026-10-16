"""WorkspaceEngine implementing Global Workspace Theory (GWT) cognitive deliberation."""

from __future__ import annotations

import asyncio
import time

from returns.result import Failure, Result, Success

from rai.assistant.ports import AssistantModelBackend, InteractionController
from rai.assistant.records import (
    AssistantCandidate,
    AssistantContextPackage,
    ConversationTurn,
    MemoryProposal,
)
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
        backend: AssistantModelBackend | None = None,
        skills: SkillRegistry | None = None,
        probe: JLensProbe | None = None,
        max_cycles: int = _DEFAULT_MAX_CYCLES,
    ) -> None:
        self.capabilities = capabilities
        self.backend = backend
        self.skills = skills
        self.probe = probe or JLensProbe()
        self.max_cycles = max_cycles
        self.arbiter = arbiter or SalienceArbiter()

        if specialists is not None:
            self.specialists = specialists
        else:
            self.specialists = (
                RetrieverSpecialist(),
                ActionReasonerSpecialist(backend=backend, capabilities=capabilities, skills=skills),
                AnswerDraftSpecialist(backend=backend),
                GroundingEvaluationSpecialist(),
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
        if manifest.evidence_required and manifest.routing_decision == "no_evidence":
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

            # 1. Gather proposals from all specialists concurrently
            tasks = [s.evaluate(workspace, token) for s in self.specialists]
            eval_results = await asyncio.gather(*tasks)

            proposals: list[WorkspaceContribution] = []
            for res in eval_results:
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
                                "updated_manifest": manifest,
                                "latency_ms": latency_ms,
                            },
                        )
                    )

            # 3. Update blackboard history and broadcast
            workspace = workspace.model_copy(
                update={
                    "broadcast": winner,
                    "history": (*workspace.history, winner),
                }
            )

            # 4. Handle winning action
            if (
                winner.kind == WorkspaceContributionKind.TOOL_CALL
                and self.capabilities is not None
            ):
                cap_name = str(winner.payload.get("capability", ""))
                args = dict(winner.payload.get("arguments", {}))
                descriptor = self.capabilities.registry.descriptor(cap_name)
                if descriptor is not None:
                    req_id = f"{workspace.turn_id}:workspace_step:{workspace.cycle + 1}"
                    request = normalize_request(
                        descriptor,
                        args,
                        actor=turn.producer,
                        request_id=req_id,
                        data_class=turn.data_class,
                    )
                    _, executed = await self.capabilities.invoke(request, token)
                    if isinstance(executed, Success):
                        act_res = executed.unwrap()
                        workspace = workspace.model_copy(
                            update={
                                "observations": (*workspace.observations, act_res.output),
                                "tool_calls_count": workspace.tool_calls_count + 1,
                            }
                        )
                    else:
                        err = executed.failure()
                        workspace = workspace.model_copy(
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
                        if isinstance(raw_cand, dict) and raw_cand.get("proposals"):
                            cand_proposals = tuple(
                                MemoryProposal.model_validate(p) for p in raw_cand["proposals"]
                            )
                        final_candidate = AssistantCandidate(
                            text=winner.content,
                            proposals=cand_proposals,
                            tokens_in=total_tokens_in,
                            tokens_out=total_tokens_out,
                            metadata={
                                "workspace_cycles": workspace.cycle + 1,
                                "workspace_converged": True,
                                "winning_specialist": winner.specialist_name,
                                "tool_calls_executed": workspace.tool_calls_count,
                                "observations_count": len(workspace.observations),
                                "updated_manifest": manifest,
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

        return Success(
            AssistantCandidate(
                text=final_text,
                tokens_in=total_tokens_in,
                tokens_out=total_tokens_out,
                metadata={
                    "workspace_cycles": workspace.cycle,
                    "workspace_converged": False,
                    "tool_calls_executed": workspace.tool_calls_count,
                    "observations_count": len(workspace.observations),
                    "updated_manifest": manifest,
                    "latency_ms": latency_ms,
                },
            )
        )
