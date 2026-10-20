"""Specialist agents competing for the Global Workspace broadcast."""

from __future__ import annotations

from datetime import timedelta
import json
from typing import Protocol, runtime_checkable

from returns.result import Failure, Result, Success

from rai.assistant.ports import AssistantModelBackend
from rai.assistant.records import (
    AssistantCandidate,
    AssistantContextPackage,
    InferenceRequest,
    ToolObservation,
)
from rai.assistant.tool_calling import extract_model_tool_call
from rai.kernel.ports import CancellationToken
from rai.kernel.records import ActionFailure, InferenceBudget, ProducerIdentity, _new_id, _utc_now
from rai.kernel.service import CapabilityService
from rai.skills.models import SkillRegistry

from .models import WorkspaceContribution, WorkspaceContributionKind, WorkspaceState

PRODUCER = ProducerIdentity(producer_id="workspace-specialist", kind="specialist", version="1.0.0")


def _make_specialist_budget(max_input_tokens: int, max_output_tokens: int) -> InferenceBudget:
    now = _utc_now()
    return InferenceBudget(
        record_id=_new_id(),
        timestamp=now,
        producer=PRODUCER,
        max_input_tokens=max_input_tokens,
        max_output_tokens=max_output_tokens,
        max_agent_turns=1,
        max_tool_calls=1,
        max_images=0,
        max_audio_seconds=0,
        max_latency_seconds=30,
        max_provider_cost=0,
        max_ram_bytes=2 * 1024 * 1024 * 1024,
        max_vram_bytes=0,
        cancellation_deadline=now + timedelta(seconds=30),
    )


@runtime_checkable
class CognitiveSpecialist(Protocol):
    """Protocol for specialized cognitive agents proposing contributions to the blackboard."""

    @property
    def name(self) -> str:
        """Unique specialist identifier."""
        ...

    async def evaluate(
        self,
        workspace: WorkspaceState,
        cancellation: CancellationToken,
    ) -> Result[WorkspaceContribution | None, ActionFailure]:
        """Inspect current blackboard working memory and propose candidate contribution with salience."""
        ...


class RetrieverSpecialist:
    """Specialist evaluating context sufficiency and summarizing available memory/knowledge evidence."""

    def __init__(self, name: str = "retriever") -> None:
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    async def evaluate(
        self,
        workspace: WorkspaceState,
        cancellation: CancellationToken,
    ) -> Result[WorkspaceContribution | None, ActionFailure]:
        del cancellation
        # If memory evidence was already evaluated in history or tool observations exist, skip
        has_retrieval = any(
            c.kind == WorkspaceContributionKind.RETRIEVAL for c in workspace.history
        )
        if has_retrieval or workspace.observations:
            return Success(None)

        evidence_req = bool(workspace.metadata.get("evidence_required", False))
        durable_count = 0
        episodic_count = 0
        summary_count = 0
        external_count = 0
        if isinstance(workspace.context, AssistantContextPackage):
            content = workspace.context.content
            durable_count = len(content.get("durable_memories", ()))
            episodic_count = len(content.get("episodic_evidence", ()))
            summary_count = len(content.get("grounded_summaries", ()))
            external_count = len(content.get("external_evidence", ()))

        total_evidence = durable_count + episodic_count + summary_count + external_count
        if evidence_req and total_evidence == 0:
            salience = 0.75
            summary_text = (
                f"Evidence is required for goal '{workspace.goal}', "
                "but no admitted memory or external evidence is present in context."
            )
        else:
            salience = 0.35
            summary_text = (
                f"Context evidence summary for goal '{workspace.goal}': "
                f"{durable_count} durable memories, {episodic_count} episodic turns, "
                f"{summary_count} summaries, {external_count} external items."
            )

        return Success(
            WorkspaceContribution(
                cycle=workspace.cycle,
                specialist_name=self.name,
                kind=WorkspaceContributionKind.RETRIEVAL,
                content=summary_text,
                payload={
                    "evidence_required": evidence_req,
                    "durable_memories_count": durable_count,
                    "episodic_evidence_count": episodic_count,
                    "grounded_summaries_count": summary_count,
                    "external_evidence_count": external_count,
                    "total_evidence_count": total_evidence,
                },
                salience=salience,
            )
        )


class ActionReasonerSpecialist:
    """Specialist reasoning about required capabilities/skills and proposing tool executions."""

    def __init__(  # noqa: PLR0913
        self,
        backend: AssistantModelBackend | None = None,
        capabilities: CapabilityService | None = None,
        skills: SkillRegistry | None = None,
        name: str = "action_reasoner",
        max_tool_calls: int = 1,
        capability_names: tuple[str, ...] | None = None,
    ) -> None:
        self._name = name
        self.backend = backend
        self.capabilities = capabilities
        self.skills = skills
        self.max_tool_calls = max_tool_calls
        self.capability_names = capability_names

    @property
    def name(self) -> str:
        return self._name

    def _available_capability_names(self) -> tuple[str, ...]:
        if self.capability_names is not None:
            return self.capability_names
        if self.capabilities is not None:
            return tuple(d.name for d in self.capabilities.registry.descriptors())
        return ()

    async def evaluate(  # noqa: PLR0911
        self,
        workspace: WorkspaceState,
        cancellation: CancellationToken,
    ) -> Result[WorkspaceContribution | None, ActionFailure]:
        avail_caps = self._available_capability_names()
        if not avail_caps or not self.backend:
            return Success(None)

        if (
            len(workspace.observations) >= self.max_tool_calls
            or workspace.tool_calls_count >= self.max_tool_calls
        ):
            return Success(None)

        skills_prompt = ""
        if self.skills is not None:
            skills_prompt = self.skills.format_skills_instruction(avail_caps)

        caps_list = ", ".join(avail_caps)
        broadcast_note = ""
        if workspace.broadcast is not None:
            broadcast_note = (
                f"\nPrevious Blackboard Broadcast ({workspace.broadcast.specialist_name} / "
                f"{workspace.broadcast.kind}): {workspace.broadcast.content}\n"
            )
        obs_note = ""
        if workspace.observations:
            obs_lines = [json.dumps(obs, ensure_ascii=False) for obs in workspace.observations]
            obs_note = "\nCollected Observations:\n" + "\n".join(obs_lines) + "\n"

        prompt = (
            f"You are the Action Reasoning Specialist in Rich AI.\n"
            f"User Goal: {workspace.goal}\n"
            f"Available Capabilities: {caps_list}\n"
            f"{broadcast_note}"
            f"{obs_note}"
            f"{skills_prompt}\n\n"
            f"If an action is needed, emit exactly `<tool_call>call: capability_name{{...}}</tool_call>`.\n"
            f"If no action is needed, reply with 'NO_ACTION'."
        )

        if workspace.context is None:
            return Success(None)

        budget = _make_specialist_budget(1024, 256)
        inference_req = InferenceRequest(
            record_id=_new_id(),
            timestamp=_utc_now(),
            producer=PRODUCER,
            session_id=workspace.session_id,
            turn_id=workspace.turn_id,
            request_id=_new_id(),
            context=workspace.context,
            budget=budget,
            system_instruction=prompt,
        )

        gen_res = await self.backend.generate(inference_req, cancellation)
        if isinstance(gen_res, Failure):
            return Failure(gen_res.failure())

        candidate = gen_res.unwrap()
        if not isinstance(candidate, AssistantCandidate):
            return Success(None)

        tool_call = extract_model_tool_call(candidate.text)
        if tool_call:
            preamble, cap_name, tool_args = tool_call
            salience = 0.90 if not workspace.observations else 0.40
            return Success(
                WorkspaceContribution(
                    cycle=workspace.cycle,
                    specialist_name=self.name,
                    kind=WorkspaceContributionKind.TOOL_CALL,
                    content=f"Proposed tool execution: {cap_name}",
                    payload={
                        "capability": cap_name,
                        "arguments": tool_args,
                        "preamble": preamble,
                    },
                    salience=salience,
                    tokens_in=candidate.tokens_in,
                    tokens_out=candidate.tokens_out,
                )
            )

        return Success(None)


class AnswerDraftSpecialist:
    """Specialist generating answer drafts from blackboard state, memory, and observations."""

    def __init__(
        self,
        backend: AssistantModelBackend | None = None,
        name: str = "answer_draft",
        system_instruction: str | None = None,
    ) -> None:
        self._name = name
        self.backend = backend
        self.system_instruction = system_instruction

    @property
    def name(self) -> str:
        return self._name

    async def evaluate(  # noqa: PLR0912
        self,
        workspace: WorkspaceState,
        cancellation: CancellationToken,
    ) -> Result[WorkspaceContribution | None, ActionFailure]:
        if not self.backend:
            return Success(None)

        obs_text = "None."
        if workspace.observations:
            obs_lines = [json.dumps(obs, ensure_ascii=False) for obs in workspace.observations]
            obs_text = "\n".join(obs_lines)

        blackboard_notes: list[str] = []
        if workspace.broadcast is not None:
            blackboard_notes.append(
                f"Active Broadcast ({workspace.broadcast.specialist_name} / "
                f"{workspace.broadcast.kind}): {workspace.broadcast.content}"
            )
        for item in workspace.history:
            if item.kind in {
                WorkspaceContributionKind.RETRIEVAL,
                WorkspaceContributionKind.CRITIQUE,
            }:
                note = f"[{item.specialist_name}:{item.kind}] {item.content}"
                if note not in blackboard_notes:
                    blackboard_notes.append(note)

        notes_section = (
            "Blackboard Notes:\n" + "\n".join(blackboard_notes) + "\n\n"
            if blackboard_notes
            else ""
        )
        base_prefix = f"{self.system_instruction}\n\n" if self.system_instruction else ""
        prompt = (
            f"{base_prefix}"
            f"Goal: {workspace.goal}\n"
            f"{notes_section}"
            f"Observations:\n{obs_text}\n\n"
            f"Provide a helpful, concise answer to the user grounded in the context and observations."
        )

        if workspace.context is None:
            return Success(None)

        req_context = workspace.context
        if isinstance(req_context, AssistantContextPackage) and workspace.observations:
            existing_obs = list(req_context.content.get("tool_observations", []))
            for idx, obs in enumerate(workspace.observations, 1):
                existing_obs.append(
                    ToolObservation(
                        source_id=f"{workspace.turn_id}:ws_obs:{idx}",
                        source_turn_id=workspace.turn_id,
                        capability="workspace_tool",
                        status="SUCCESS",
                        summary=json.dumps(obs, ensure_ascii=False)[:120],
                        untrusted_content=True,
                        content=obs if isinstance(obs, dict) else {"value": obs},
                    ).model_dump(mode="json")
                )
            req_context = req_context.model_copy(
                update={
                    "content": {
                        **req_context.content,
                        "tool_observations": existing_obs,
                    }
                }
            )

        budget = _make_specialist_budget(1500, 512)
        inference_req = InferenceRequest(
            record_id=_new_id(),
            timestamp=_utc_now(),
            producer=PRODUCER,
            session_id=workspace.session_id,
            turn_id=workspace.turn_id,
            request_id=_new_id(),
            context=req_context,
            budget=budget,
            system_instruction=prompt,
        )

        gen_res = await self.backend.generate(inference_req, cancellation)
        if isinstance(gen_res, Failure):
            return Failure(gen_res.failure())

        candidate = gen_res.unwrap()
        if not isinstance(candidate, AssistantCandidate):
            return Success(None)

        # If observations exist or this is the final cycle, answer salience is 0.95;
        # otherwise 0.85 (>= 0.80 convergence threshold so conversational turns converge
        # in 1 cycle, while still < 0.90 of ActionReasonerSpecialist's TOOL_CALL).
        if (
            workspace.observations
            or (workspace.max_cycles is not None and workspace.cycle >= workspace.max_cycles - 1)
        ):
            salience = 0.95
        else:
            salience = 0.85

        return Success(
            WorkspaceContribution(
                cycle=workspace.cycle,
                specialist_name=self.name,
                kind=WorkspaceContributionKind.ANSWER_DRAFT,
                content=candidate.text,
                payload={"candidate": candidate.model_dump()},
                salience=salience,
                tokens_in=candidate.tokens_in,
                tokens_out=candidate.tokens_out,
            )
        )


class GroundingEvaluationSpecialist:
    """Specialist verifying that proposed answers are grounded in observations."""

    def __init__(self, name: str = "grounding_eval") -> None:
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    async def evaluate(
        self,
        workspace: WorkspaceState,
        cancellation: CancellationToken,
    ) -> Result[WorkspaceContribution | None, ActionFailure]:
        del cancellation
        # Only evaluate if the current broadcast is an answer draft
        if not workspace.broadcast or workspace.broadcast.kind != WorkspaceContributionKind.ANSWER_DRAFT:
            return Success(None)

        # If answer is ungrounded (e.g. empty when observations exist)
        answer_text = workspace.broadcast.content
        if not answer_text.strip():
            return Success(
                WorkspaceContribution(
                    cycle=workspace.cycle,
                    specialist_name=self.name,
                    kind=WorkspaceContributionKind.CRITIQUE,
                    content="Draft answer is empty.",
                    payload={"target_specialist": workspace.broadcast.specialist_name},
                    salience=0.85,
                )
            )

        return Success(None)
