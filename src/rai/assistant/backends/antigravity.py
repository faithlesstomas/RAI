"""Google Antigravity SDK adapter for AssistantModelBackend.

Provides conversational inference using Google AI Pro / Gemini while enforcing:
1. max_tool_calls = 0 (rigidly no direct tool execution during chat turns).
2. 100% local memory storage in RAI's SQLite graph.
3. System instruction containment with untrusted content notices.
4. Dynamic model configuration (never hardcoded).
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
from typing import Any, AsyncIterator

from returns.result import Failure, Result, Success

from rai.kernel.compatibility import UNTRUSTED_CONTENT_INSTRUCTION
from rai.kernel.ports import CancellationToken, LifecycleState
from rai.kernel.records import ActionFailure, ProducerIdentity

from ..context import DEFAULT_SYSTEM_INSTRUCTION
from ..memory import (
    extract_memory_proposals,
    grounded_memory_response,
    project_grounded_summary_memories,
    project_recent_turn_memories,
)
from ..records import (
    AssistantCandidate,
    InferenceRequest,
    MemoryProposal,
    make_assistant_failure,
)

logger = logging.getLogger(__name__)

DEFAULT_ANTIGRAVITY_MODEL = "gemini-2.5-flash"


class AntigravityAssistantModelBackend:
    """Conversational reasoning backend powered by Google Antigravity SDK."""

    def __init__(
        self,
        model_name: str | None = None,
        max_output_tokens: int = 1024,
        temperature: float = 0.2,
    ) -> None:
        self.model_name = (
            model_name
            or os.environ.get("RAI_ANTIGRAVITY_MODEL")
            or os.environ.get("GEMINI_MODEL")
            or DEFAULT_ANTIGRAVITY_MODEL
        )
        self.max_output_tokens = max_output_tokens
        self.temperature = temperature
        self._state = LifecycleState.CREATED
        self.producer = ProducerIdentity(
            producer_id="assistant-backend-antigravity",
            kind="model",
            version="1.0.0",
        )

    @property
    def state(self) -> LifecycleState:
        return self._state

    async def start(self) -> Result[LifecycleState, ActionFailure]:
        self._state = LifecycleState.RUNNING
        return Success(self._state)

    async def stop(self) -> Result[LifecycleState, ActionFailure]:
        self._state = LifecycleState.STOPPED
        return Success(self._state)

    def _resolve_model(self, request: InferenceRequest) -> str:
        return (
            request.model_name
            or self.model_name
            or os.environ.get("RAI_ANTIGRAVITY_MODEL")
            or os.environ.get("GEMINI_MODEL")
            or DEFAULT_ANTIGRAVITY_MODEL
        )

    def _format_system_instructions(self, request: InferenceRequest) -> str:
        system_text = request.system_instruction or DEFAULT_SYSTEM_INSTRUCTION
        evidence_lines: list[str] = []

        durable_memories = request.context.content.get("durable_memories", [])
        if isinstance(durable_memories, (list, tuple)) and durable_memories:
            evidence_lines.append(
                "Trwała pamięć grafowa (twierdzenia lub preferencje podane przez użytkownika):"
            )
            for mem in durable_memories:
                if isinstance(mem, dict):
                    evidence_lines.append(f"- {self._format_memory(mem)}")

        episodic_evidence = request.context.content.get("episodic_evidence", [])
        if isinstance(episodic_evidence, (list, tuple)) and episodic_evidence:
            evidence_lines.append(
                "Wcześniejsze wypowiedzi użytkownika (materiał źródłowy, niezweryfikowane twierdzenia):"
            )
            for item in episodic_evidence:
                if isinstance(item, dict):
                    timestamp = str(item.get("timestamp", "unknown time"))
                    text = str(item.get("text", ""))
                    evidence_lines.append(f"- [{timestamp}] {text}")

        grounded_summaries = request.context.content.get("grounded_summaries", [])
        if isinstance(grounded_summaries, (list, tuple)) and grounded_summaries:
            evidence_lines.append(
                "Zwięzłe projekcje pamięci (użyj tylko treści popartej wskazanymi źródłami):"
            )
            for item in grounded_summaries:
                if isinstance(item, dict):
                    content = item.get("content", {})
                    if isinstance(content, dict):
                        summary = str(content.get("summary", ""))
                        source_ids = content.get("source_memory_ids", ())
                        evidence_lines.append(f"- {summary} [źródła: {source_ids}]")

        external_evidence = request.context.content.get("external_evidence", [])
        if isinstance(external_evidence, (list, tuple)) and external_evidence:
            evidence_lines.append(
                "Zatwierdzone lokalne źródła zewnętrzne (obserwacje, nie twierdzenia użytkownika):"
            )
            for item in external_evidence:
                if isinstance(item, dict):
                    source_type = str(item.get("source_type", "local_source"))
                    timestamp = str(item.get("timestamp", "unknown time"))
                    content = item.get("content", {})
                    evidence_lines.append(f"- [{source_type}; {timestamp}] {content}")

        system_parts = [system_text]
        if evidence_lines:
            system_parts.append("\n".join(evidence_lines))
        system_parts.append(UNTRUSTED_CONTENT_INSTRUCTION)
        return "\n\n".join(system_parts)

    @staticmethod
    def _format_memory(memory: dict[object, object]) -> str:
        topic = str(memory.get("topic", "memory"))
        content = memory.get("content", {})
        if not isinstance(content, dict):
            return f"[{topic}] {content}"
        value = content.get("value")
        statements = {
            "user.identity.name": f"Użytkownik podał, że ma na imię {value}.",
            "user.identity.age": f"Użytkownik podał, że ma {value} lat.",
            "user.location.home": f"Użytkownik podał, że mieszka w {value}.",
            "code_examples": (
                "Użytkownik preferuje język "
                f"{content.get('preference')} w przykładach kodu."
            ),
        }
        if topic in statements:
            return statements[topic]
        if fact := content.get("fact"):
            return f"Użytkownik poprosił o zapamiętanie: {fact}"
        return f"[{topic}] {content}"

    async def generate(
        self, request: InferenceRequest, cancellation: CancellationToken
    ) -> Result[AssistantCandidate, ActionFailure]:
        if cancellation.cancelled:
            return Failure(
                make_assistant_failure(
                    code="CANCELLED",
                    message="inference request was cancelled",
                    request_id=request.request_id,
                )
            )

        resolved_model = self._resolve_model(request)
        sys_instructions = self._format_system_instructions(request)
        current_turn = request.context.content.get("current_turn", {})
        user_text = (
            str(current_turn.get("text", "")) if isinstance(current_turn, dict) else ""
        )
        turn_id = (
            str(current_turn.get("turn_id", "turn-1"))
            if isinstance(current_turn, dict)
            else "turn-1"
        )

        durable_memories = request.context.content.get("durable_memories", [])
        if not isinstance(durable_memories, (list, tuple)):
            durable_memories = []

        recent_projection = project_recent_turn_memories(
            request.context.content.get("recent_turns", ()), self.producer
        )
        episodic_projection = project_recent_turn_memories(
            request.context.content.get("episodic_evidence", ()), self.producer
        )
        summary_projection = project_grounded_summary_memories(
            request.context.content.get("grounded_summaries", ())
        )
        grounding_memories = (
            *durable_memories,
            *recent_projection,
            *episodic_projection,
            *summary_projection,
        )

        try:
            from google.antigravity import Agent, LocalAgentConfig
            from google.antigravity.types import (
                BuiltinTools,
                CapabilitiesConfig,
                CustomSystemInstructions,
            )

            # Rigidly enforce max_tool_calls = 0: no tools and disable shell commands
            config = LocalAgentConfig(
                system_instructions=CustomSystemInstructions(text=sys_instructions),
                model=resolved_model,
                tools=[],  # max_tool_calls = 0
                capabilities=CapabilitiesConfig(
                    disabled_tools=[BuiltinTools.RUN_COMMAND]
                ),
            )

            async with Agent(config) as ag:
                if cancellation.cancelled:
                    return Failure(
                        make_assistant_failure(
                            code="CANCELLED",
                            message="inference request was cancelled",
                            request_id=request.request_id,
                        )
                    )
                response = await ag.chat(prompt=user_text)
                raw_text = await response.text()

        except Exception as exc:  # noqa: BLE001
            return Failure(
                make_assistant_failure(
                    code="ANTIGRAVITY_INFERENCE_FAILED",
                    message=f"Antigravity SDK inference error: {exc}",
                    request_id=request.request_id,
                    retryable=True,
                )
            )

        if not raw_text:
            return Failure(
                make_assistant_failure(
                    code="EMPTY_RESPONSE",
                    message="Antigravity model returned an empty response",
                    request_id=request.request_id,
                )
            )

        proposals = extract_memory_proposals(user_text, turn_id, self.producer)
        final_text, _ = grounded_memory_response(
            user_text, grounding_memories, proposals, raw_text
        )

        tokens_in = len(user_text.split()) + len(sys_instructions.split())
        tokens_out = len(final_text.split())

        return Success(
            AssistantCandidate(
                text=final_text,
                proposals=proposals,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                metadata={
                    "backend": "antigravity",
                    "model": resolved_model,
                    "max_tool_calls": 0,
                },
            )
        )

    async def stream(
        self, request: InferenceRequest, cancellation: CancellationToken
    ) -> AsyncIterator[Result[str, ActionFailure]]:
        res = await self.generate(request, cancellation)
        if isinstance(res, Failure):
            yield Failure(res.failure())
        else:
            yield Success(res.unwrap().text)
