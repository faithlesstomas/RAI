"""Direct interaction controller for single-turn conversational exchanges and tool execution."""

from __future__ import annotations

import json
import time
from typing import Any

from returns.result import Failure, Result, Success

from rai.diagnostics import trace, trace_payload
from rai.kernel.ports import CancellationToken
from rai.kernel.records import (
    ActionFailure,
    InferenceBudget,
    ProducerIdentity,
    _new_id,
    _utc_now,
)

from ..backends.deterministic import DeterministicAssistantBackend
from ..ports import (
    AssistantActionHandler,
    AssistantModelBackend,
    InteractionController,
)
from ..records import (
    AssistantCandidate,
    AssistantContextManifest,
    AssistantContextPackage,
    ConversationTurn,
    InferenceRequest,
    ToolObservation,
    make_assistant_failure,
)
from ..tool_calling import extract_model_tool_call, intent_from_tool_call

_MAX_SYNTHESIS_TOOL_OBSERVATIONS: int = 3
_MEMORY_ABSTENTION_TEXT: str = (
    "Nie mam wystarczających, dopuszczonych dowodów w pamięci, aby odpowiedzieć."
)


class DirectController(InteractionController):
    """Single-turn direct generation / conversational exchange controller."""

    def __init__(
        self,
        backend: AssistantModelBackend | None = None,
        actions: AssistantActionHandler | None = None,
        context_builder: object = None,
        producer: ProducerIdentity | None = None,
    ) -> None:
        self.backend = backend or DeterministicAssistantBackend()
        self.actions = actions
        self.context_builder = context_builder
        self.producer = producer or ProducerIdentity(
            producer_id="direct-controller", kind="controller", version="1.0.0"
        )

    async def process_turn(  # noqa: PLR0911
        self,
        turn: ConversationTurn,
        context: AssistantContextPackage,
        budget: InferenceBudget,
        token: CancellationToken,
    ) -> Result[AssistantCandidate, ActionFailure]:
        """Process one conversational turn directly via actions, evidence checks, or backend inference."""
        manifest = context.manifest
        request_id = _new_id()
        system_instruction = getattr(self.context_builder, "system_instruction", None)
        model_name = str(getattr(self.backend, "model_name", "deterministic"))

        inference_req = InferenceRequest(
            record_id=_new_id(),
            timestamp=_utc_now(),
            producer=self.producer,
            session_id=turn.session_id,
            turn_id=turn.record_id,
            request_id=request_id,
            context=context,
            budget=budget,
            strategy="DIRECT",
            system_instruction=system_instruction,
            model_name=model_name,
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
                context_package=context,
                manifest=manifest,
                inference_req=inference_req,
                action_candidate=action_candidate,
                token=token,
            )
            if isinstance(synth_res, Failure):
                return Failure(synth_res.failure())
            candidate, updated_manifest, latency_ms = synth_res.unwrap()
            return Success(
                candidate.model_copy(
                    update={
                        "metadata": {
                            **candidate.metadata,
                            "updated_manifest": updated_manifest,
                            "latency_ms": latency_ms,
                        }
                    }
                )
            )

        if action_candidate is not None:
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
            return Failure(backend_res.failure())

        candidate = backend_res.unwrap()
        if not isinstance(candidate, AssistantCandidate):
            return Failure(
                make_assistant_failure(
                    code="INVALID_OUTPUT",
                    message="Model backend produced invalid candidate payload",
                    producer=self.producer,
                )
            )
        updated_manifest = manifest

        if self.actions is not None and (tool_call := extract_model_tool_call(candidate.text)):
            preamble, cap_name, tool_args = tool_call
            tool_intent = intent_from_tool_call(turn, cap_name, tool_args)
            if tool_intent is not None and hasattr(self.actions, "dispatch_intent"):
                tool_cand = await self.actions.dispatch_intent(turn, tool_intent, token)
                if tool_cand is not None:
                    synth_res = await self._synthesize_action_candidate(
                        turn=turn,
                        request_id=request_id,
                        context_package=context,
                        manifest=manifest,
                        inference_req=inference_req,
                        action_candidate=tool_cand,
                        token=token,
                        preamble=preamble,
                    )
                    if isinstance(synth_res, Success):
                        candidate, updated_manifest, synth_lat = synth_res.unwrap()
                        latency_ms += synth_lat

        return Success(
            candidate.model_copy(
                update={
                    "metadata": {
                        **candidate.metadata,
                        "updated_manifest": updated_manifest,
                        "latency_ms": latency_ms,
                    }
                }
            )
        )

    async def _synthesize_action_candidate(  # noqa: PLR0913, PLR0915
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
        """Synthesize action results or observational data into natural language."""
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
            if next_intent is not None and hasattr(self.actions, "dispatch_intent"):
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
