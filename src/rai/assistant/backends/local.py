"""Direct local AssistantModelBackend adapter wrapping LocalTextEngine.

Resolves issue #2 by bypassing broken chat templating paths and using inspectable
plain-text completion prompts.
"""

from __future__ import annotations

import asyncio
import re
from typing import AsyncIterator

from returns.result import Failure, Result, Success

from rai.kernel.ports import CancellationToken, LifecycleState
from rai.kernel.records import ActionFailure, ProducerIdentity

from ..context import DEFAULT_SYSTEM_INSTRUCTION
from ..memory import extract_memory_proposals, grounded_memory_response
from ..records import (
    AssistantCandidate,
    InferenceRequest,
    MemoryProposal,
    make_assistant_failure,
)


class LocalAssistantBackend:
    """Direct local model backend adapter for AssistantModelBackend protocol."""

    def __init__(  # noqa: PLR0913
        self,
        engine: object | None = None,
        model_name: str | None = None,
        backend_name: str = "ollama",
        max_output_tokens: int = 256,
        temperature: float = 0.2,
        model_artifact_version: str | None = None,
    ) -> None:
        self.engine = engine
        self.model_name = model_name or "local-model"
        self.backend_name = backend_name
        self.max_output_tokens = max_output_tokens
        self.temperature = temperature
        self.model_artifact_version = model_artifact_version
        self.prompt_template_version = "rai-assistant-plain-text-v3"
        self._state = LifecycleState.CREATED
        self.producer = ProducerIdentity(
            producer_id=f"local-assistant-{self.backend_name}",
            kind="model",
            version="1.0.0",
        )

    @property
    def state(self) -> LifecycleState:
        return self._state

    async def start(self) -> Result[LifecycleState, ActionFailure]:
        if self.engine is None:
            self._state = LifecycleState.FAILED
            return Failure(
                make_assistant_failure(
                    code="ENGINE_NOT_CONFIGURED",
                    message="no local text engine is configured",
                )
            )
        if self.engine is not None and hasattr(self.engine, "load"):
            try:
                load_res = await self.engine.load()
                if isinstance(load_res, Failure):
                    self._state = LifecycleState.FAILED
                    return Failure(
                        make_assistant_failure(
                            code="ENGINE_LOAD_FAILED",
                            message=f"failed to load engine: {load_res.failure()}",
                        )
                    )
            except Exception as exc:  # noqa: BLE001
                self._state = LifecycleState.FAILED
                return Failure(
                    make_assistant_failure(
                        code="ENGINE_LOAD_FAILED",
                        message=f"exception loading engine: {exc}",
                    )
                )
        self._state = LifecycleState.RUNNING
        return Success(self._state)

    async def stop(self) -> Result[LifecycleState, ActionFailure]:
        if self.engine is not None and hasattr(self.engine, "unload"):
            try:
                await self.engine.unload()
            except Exception:  # noqa: BLE001, S110
                pass
        self._state = LifecycleState.STOPPED
        return Success(self._state)

    def _format_prompt(self, request: InferenceRequest) -> str:  # noqa: PLR0912
        """Format an inspectable plain-text prompt bypassing chat template issues."""
        system_text = request.system_instruction or DEFAULT_SYSTEM_INSTRUCTION
        lines = [
            f"System: {system_text}",
            "Use only relevant conversation and memory evidence below. Durable memory contains "
            "user-stated claims or preferences, not verified world facts. If evidence is missing "
            "or conflicting, say that you do not know. Never invent missing details.",
            "",
        ]

        durable_memories = request.context.content.get("durable_memories", [])
        if isinstance(durable_memories, (list, tuple)) and durable_memories:
            lines.append(
                "Trwała pamięć grafowa (twierdzenia lub preferencje podane przez użytkownika):"
            )
            for mem in durable_memories:
                if isinstance(mem, dict):
                    lines.append(f"- {self._format_memory(mem)}")
            lines.append("")

        episodic_evidence = request.context.content.get("episodic_evidence", [])
        if isinstance(episodic_evidence, (list, tuple)) and episodic_evidence:
            lines.append(
                "Wcześniejsze wypowiedzi użytkownika (materiał źródłowy, niezweryfikowane twierdzenia):"
            )
            for item in episodic_evidence:
                if isinstance(item, dict):
                    timestamp = str(item.get("timestamp", "unknown time"))
                    text = str(item.get("text", ""))
                    lines.append(f"- [{timestamp}] {text}")
            lines.append("")

        grounded_summaries = request.context.content.get("grounded_summaries", [])
        if isinstance(grounded_summaries, (list, tuple)) and grounded_summaries:
            lines.append(
                "Zwięzłe projekcje pamięci (użyj tylko treści popartej wskazanymi źródłami):"
            )
            for item in grounded_summaries:
                if not isinstance(item, dict):
                    continue
                content = item.get("content", {})
                if isinstance(content, dict):
                    summary = str(content.get("summary", ""))
                    source_ids = content.get("source_memory_ids", ())
                    lines.append(f"- {summary} [źródła: {source_ids}]")
            lines.append("")

        external_evidence = request.context.content.get("external_evidence", [])
        if isinstance(external_evidence, (list, tuple)) and external_evidence:
            lines.append(
                "Zatwierdzone lokalne źródła zewnętrzne (obserwacje, nie twierdzenia użytkownika):"
            )
            for item in external_evidence:
                if isinstance(item, dict):
                    source_type = str(item.get("source_type", "local_source"))
                    timestamp = str(item.get("timestamp", "unknown time"))
                    content = item.get("content", {})
                    lines.append(f"- [{source_type}; {timestamp}] {content}")
            lines.append("")

        recent_turns = request.context.content.get("recent_turns", [])
        if isinstance(recent_turns, (list, tuple)) and recent_turns:
            lines.append("Historia bieżącej rozmowy:")
            for turn in recent_turns:
                if isinstance(turn, dict):
                    role = str(turn.get("role", "")).capitalize()
                    text = str(turn.get("text", ""))
                    lines.append(f"{role}: {text}")
            lines.append("")

        current_turn = request.context.content.get("current_turn", {})
        user_text = (
            str(current_turn.get("text", "")) if isinstance(current_turn, dict) else ""
        )
        lines.append(f"Użytkownik: {user_text}")
        lines.append("Asystent:")
        return "\n".join(lines)

    @staticmethod
    def _format_memory(memory: dict[object, object]) -> str:
        """Render structured memory as a natural-language evidence statement."""
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

    def _extract_proposals(
        self, user_text: str, turn_id: str
    ) -> tuple[MemoryProposal, ...]:
        """Deterministically extract bounded memory proposals from user statements."""
        return extract_memory_proposals(user_text, turn_id, self.producer)

    @staticmethod
    def _truncate_repetition(text: str) -> tuple[str, bool]:
        """Stop exact sentence loops commonly emitted by very small local models."""
        parts = re.split(r"(?<=[.!?])\s+", text.strip())
        seen: set[str] = set()
        kept: list[str] = []
        for part in parts:
            normalized = re.sub(r"\s+", " ", part).strip().casefold()
            if normalized and normalized in seen:
                return " ".join(kept).strip(), True
            if normalized:
                seen.add(normalized)
                kept.append(part.strip())
        return text.strip(), False

    @staticmethod
    def _ground_memory_response(
        user_text: str,
        durable_memories: tuple[object, ...] | list[object],
        proposals: tuple[MemoryProposal, ...],
        model_text: str,
    ) -> tuple[str, bool]:
        """Apply deterministic grounding for persisted personal facts."""
        return grounded_memory_response(
            user_text, durable_memories, proposals, model_text
        )

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

        prompt = self._format_prompt(request)
        current_turn = request.context.content.get("current_turn", {})
        user_text = (
            str(current_turn.get("text", "")) if isinstance(current_turn, dict) else ""
        )
        durable_memories = request.context.content.get("durable_memories", [])
        if not isinstance(durable_memories, (list, tuple)):
            durable_memories = []

        if self.engine is None or not hasattr(self.engine, "generate"):
            return Failure(
                make_assistant_failure(
                    code="ENGINE_NOT_CONFIGURED",
                    message="no local text engine is configured",
                    request_id=request.request_id,
                )
            )
        try:
            gen_res = await self.engine.generate(
                prompt=prompt,
                stop=["\nUżytkownik:", "\nSystem:", "\nAsystent:"],
                max_tokens=self.max_output_tokens,
                temperature=self.temperature,
            )
            if isinstance(gen_res, Failure):
                return Failure(
                    make_assistant_failure(
                        code="ENGINE_ERROR",
                        message=f"local engine failed: {gen_res.failure()}",
                        request_id=request.request_id,
                        retryable=True,
                    )
                )
            result_obj = gen_res.unwrap()
            response_text = result_obj.text.strip()
        except Exception as exc:  # noqa: BLE001
            return Failure(
                make_assistant_failure(
                    code="ENGINE_ERROR",
                    message=f"exception during local engine generation: {exc}",
                    request_id=request.request_id,
                    retryable=True,
                )
            )

        if not response_text:
            return Failure(
                make_assistant_failure(
                    code="INVALID_OUTPUT",
                    message="local engine returned an empty response",
                    request_id=request.request_id,
                )
            )

        response_text, repetition_truncated = self._truncate_repetition(response_text)
        proposals = self._extract_proposals(user_text, request.turn_id)
        grounded_text, grounding_override = self._ground_memory_response(
            user_text, durable_memories, proposals, response_text
        )
        candidate = AssistantCandidate(
            text=grounded_text,
            proposals=proposals,
            tokens_in=(
                result_obj.stats.input_tokens
                if result_obj.stats
                else len(prompt.split())
            ),
            tokens_out=(
                result_obj.stats.output_tokens
                if result_obj.stats
                else len(response_text.split())
            ),
            metadata={
                "backend": self.backend_name,
                "model": self.model_name,
                "model_artifact_version": self.model_artifact_version,
                "prompt_template_version": self.prompt_template_version,
                "finish_reason": result_obj.finish_reason,
                "grounding_override": grounding_override,
                "repetition_truncated": repetition_truncated,
                "delivered_words": len(grounded_text.split()),
                "raw_model_output": response_text,
            },
        )
        return Success(candidate)

    async def stream(
        self, request: InferenceRequest, cancellation: CancellationToken
    ) -> AsyncIterator[Result[str, ActionFailure]]:
        res = await self.generate(request, cancellation)
        if isinstance(res, Failure):
            yield Failure(res.failure())
            return

        candidate = res.unwrap()
        words = re.split(r"(\s+)", candidate.text)
        for word in words:
            if cancellation.cancelled:
                yield Failure(
                    make_assistant_failure(
                        code="CANCELLED",
                        message="streaming was cancelled",
                        request_id=request.request_id,
                    )
                )
                return
            if word:
                yield Success(word)
                await asyncio.sleep(0.005)
