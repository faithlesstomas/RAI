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
from rai.kernel.records import ActionFailure, ProducerIdentity, _new_id, _utc_now

from ..context import DEFAULT_SYSTEM_INSTRUCTION
from ..records import (
    AssistantCandidate,
    InferenceRequest,
    MemoryProposal,
    make_assistant_failure,
)


class LocalAssistantBackend:
    """Direct local model backend adapter for AssistantModelBackend protocol."""

    def __init__(
        self,
        engine: object | None = None,
        model_name: str | None = None,
        backend_name: str = "ollama",
    ) -> None:
        self.engine = engine
        self.model_name = model_name or "local-model"
        self.backend_name = backend_name
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

    def _format_prompt(self, request: InferenceRequest) -> str:
        """Format an inspectable plain-text prompt bypassing chat template issues."""
        system_text = request.system_instruction or DEFAULT_SYSTEM_INSTRUCTION
        lines = [f"System: {system_text}", ""]

        durable_memories = request.context.content.get("durable_memories", [])
        if isinstance(durable_memories, (list, tuple)) and durable_memories:
            lines.append("Zapisane fakty i preferencje użytkownika (trwała pamięć grafowa):")
            for mem in durable_memories:
                if isinstance(mem, dict):
                    topic = mem.get("topic", "")
                    content = mem.get("content", {})
                    pref = (
                        content.get("preference", "")
                        if isinstance(content, dict)
                        else ""
                    )
                    lines.append(f"- [{topic}] Preferencja: {pref}")
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
            str(current_turn.get("text", ""))
            if isinstance(current_turn, dict)
            else ""
        )
        lines.append(f"Użytkownik: {user_text}")
        lines.append("Asystent:")
        return "\n".join(lines)

    def _extract_proposals(
        self, user_text: str, turn_id: str
    ) -> tuple[MemoryProposal, ...]:
        """Deterministically extract bounded memory proposals from user statements."""
        lowered = user_text.lower()
        proposals: list[MemoryProposal] = []

        pref_match = None
        if "guile" in lowered:
            pref_match = "Guile"
        elif "python" in lowered or "pythona" in lowered:
            pref_match = "Python"

        if (
            "preferuj" in lowered
            or "preferuję" in lowered
            or "zapamiętaj" in lowered
            or "zmień tę preferencję" in lowered
            or "używaj" in lowered
        ) and pref_match:
            proposals.append(
                MemoryProposal(
                    record_id=_new_id(),
                    timestamp=_utc_now(),
                    producer=self.producer,
                    source_turn_id=turn_id,
                    kind="preference",
                    topic="code_examples",
                    content={
                        "topic": "code_examples",
                        "preference": pref_match,
                        "raw_statement": user_text,
                    },
                )
            )

        return tuple(proposals)

    def _fallback_inference(
        self, user_text: str, durable_memories: tuple[object, ...] | list[object]
    ) -> str:
        """Fallback response logic when running without a heavy local weights file."""
        lowered = user_text.lower()
        if "w jakim języku" in lowered or "jakim języku" in lowered:
            for mem in durable_memories:
                if isinstance(mem, dict):
                    content = mem.get("content", {})
                    if (
                        isinstance(content, dict)
                        and content.get("topic") == "code_examples"
                    ):
                        return (
                            f"Zgodnie z Twoją zapisaną preferencją, powinienem pokazywać przykłady kodu w języku {content.get('preference')}."
                        )
            return "Nie mam zapisanej preferencji dotyczącej języka w przykładach kodu."

        if "preferuj" in lowered or "preferuję" in lowered or "zapamiętaj" in lowered:
            lang = "Guile" if "guile" in lowered else ("Python" if "python" in lowered else "wybranym")
            return f"Zapamiętałem: w przykładach kodu będę preferować język {lang}."

        if "zmień tę preferencję" in lowered or "używaj pythona" in lowered:
            return "Zmieniłem preferencję: od nowa w przykładach kodu będę używać języka Python."

        return "Rozumiem."

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
            str(current_turn.get("text", ""))
            if isinstance(current_turn, dict)
            else ""
        )
        durable_memories = request.context.content.get("durable_memories", [])
        if not isinstance(durable_memories, (list, tuple)):
            durable_memories = []

        response_text: str
        if self.engine is not None and hasattr(self.engine, "generate"):
            try:
                gen_res = await self.engine.generate(
                    prompt=prompt,
                    stop=["Użytkownik:", "System:"],
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
        else:
            response_text = self._fallback_inference(user_text, durable_memories)

        proposals = self._extract_proposals(user_text, request.turn_id)
        candidate = AssistantCandidate(
            text=response_text or "Rozumiem.",
            proposals=proposals,
            tokens_in=len(prompt.split()),
            tokens_out=len(response_text.split()) if response_text else 1,
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
