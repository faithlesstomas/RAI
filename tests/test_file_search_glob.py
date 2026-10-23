"""Tests for glob pattern matching, directory scoping, and multi-step request ID uniqueness in file operations."""
from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from typing import Any, NoReturn

import pytest
from returns.result import Result, Success

from rai.actions.assistant import AssistantActions
from rai.actions.execution import SQLiteExecutionStore
from rai.actions.files import (
    Document,
    _matches_document_query,
    inspect_document,
    register_file_capabilities,
    search_documents,
)
from rai.actions.handles import SQLiteHandleStore
from rai.actions.intent import ActionIntent
from rai.actions.records import ActiveChoiceFrame, ChoiceEntry, ChoiceState, ResourceHandle
from rai.actions.service import ActionCapabilityService
from rai.actions.workflow import BoundedStepController
from rai.assistant.ports import AssistantModelBackend
from rai.assistant.records import (
    AssistantCandidate,
    AssistantContextManifest,
    AssistantContextPackage,
    ConversationTurn,
    InferenceRequest,
)
from rai.assistant.store import SQLiteMemoryGraphStore
from rai.assistant.tool_calling import intent_from_tool_call
from rai.kernel.audit import InMemoryAuditLedger
from rai.kernel.capabilities import CapabilityRegistry
from rai.kernel.policy import PolicyEngine
from rai.kernel.ports import CancellationToken, LifecycleState
from rai.kernel.records import (
    ActionFailure,
    DataClass,
    InferenceBudget,
    ProducerIdentity,
    _new_id,
    _utc_now,
)
from rai.kernel.synthetic import SyntheticApprovalBroker
from rai.kernel.transport import normalize_request

ACTOR = ProducerIdentity(producer_id="user", kind="user", version="1.0.0")


def test_matches_document_query_glob_and_extensions() -> None:
    # Glob wildcard matches
    assert _matches_document_query("1103858973.pdf", "*.pdf")
    assert _matches_document_query("1103858973.PDF", "*.pdf")
    assert _matches_document_query("report_final.pdf", "*final*.pdf")
    assert _matches_document_query("doc1.txt", "doc?.txt")
    assert not _matches_document_query("readme.txt", "*.pdf")

    # Extension dot matches
    assert _matches_document_query("1103858973.pdf", ".pdf")
    assert _matches_document_query("file.tar.gz", ".gz")
    assert not _matches_document_query("notes.txt", ".pdf")

    # Substring matches
    assert _matches_document_query("1103858973.pdf", "1103")
    assert _matches_document_query("1103858973.pdf", "pdf")
    assert _matches_document_query("annual_report.txt", "report")
    assert not _matches_document_query("notes.txt", "missing")


def test_search_documents_wildcards_and_scoping(tmp_path: Path) -> None:
    root1 = tmp_path / "workspace"
    root1.mkdir()
    sub_dir = root1 / "reports"
    sub_dir.mkdir()
    root2 = tmp_path / "tmp_dir"
    root2.mkdir()

    # Create files
    pdf1 = root1 / "1103858973.pdf"
    pdf1.write_text("dummy pdf 1")
    pdf2 = sub_dir / "annual_report.pdf"
    pdf2.write_text("dummy pdf 2")
    txt1 = root1 / "notes.txt"
    txt1.write_text("some notes")
    pdf3 = root2 / "temp_document.pdf"
    pdf3.write_text("dummy pdf 3")

    roots = (root1, root2)

    # 1. Search with *.pdf across all roots
    found_all = search_documents(roots, "*.pdf")
    names = [doc.path.name for doc in found_all]
    assert "1103858973.pdf" in names
    assert "annual_report.pdf" in names
    assert "temp_document.pdf" in names
    assert "notes.txt" not in names

    # 2. Search scoped to sub_dir using directory argument
    found_sub = search_documents(roots, "*.pdf", directory=sub_dir)
    assert len(found_sub) == 1
    assert found_sub[0].path.name == "annual_report.pdf"

    # 3. Search with absolute path containing wildcards
    found_abs_wildcard = search_documents(roots, str(root2 / "*.pdf"))
    assert len(found_abs_wildcard) == 1
    assert found_abs_wildcard[0].path.name == "temp_document.pdf"

    # 4. Search with numeric substring
    found_num = search_documents(roots, "1103")
    assert len(found_num) == 1
    assert found_num[0].path.name == "1103858973.pdf"


async def test_file_search_capability_with_directory_and_total_count(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    sub_folder = workspace / "sub"
    sub_folder.mkdir()

    (workspace / "1103858973.pdf").write_text("content")
    (sub_folder / "sub_doc.pdf").write_text("content sub")

    class DummyBackend:
        async def open(self, document: Document, cancellation: CancellationToken) -> Result[dict[str, Any], str]:
            return Success({"status": "OK"})

    registry = CapabilityRegistry()
    handles = SQLiteHandleStore(tmp_path / "handles.db")
    register_file_capabilities(registry, (workspace,), handles, DummyBackend())

    service = ActionCapabilityService(registry, PolicyEngine(), InMemoryAuditLedger(), SyntheticApprovalBroker())
    service.handles = handles
    service.executions = SQLiteExecutionStore(tmp_path / "executions.db")

    descriptor = registry.descriptor("file.search")
    assert descriptor is not None
    assert "directory" in descriptor.input_schema["properties"]

    # Search with directory restriction
    req = normalize_request(
        descriptor,
        {"task_id": "task-1", "query": "*.pdf", "directory": str(sub_folder)},
        actor=ACTOR,
        request_id="req-1",
    )
    decision, result_res = await service.invoke(req)
    assert isinstance(result_res, Success)
    output = result_res.unwrap().output
    assert output["total_count"] == 1
    assert len(output["documents"]) == 1
    assert output["documents"][0]["name"] == "sub_doc.pdf"


def test_tool_calling_extracts_query_and_directory() -> None:
    turn = ConversationTurn(producer=ACTOR, session_id="s1", role="user", text="szukaj")
    args = {"query": "*.pdf", "directory": "/home/user/docs"}
    intent = intent_from_tool_call(turn, "file.search", args)
    assert intent is not None
    assert intent.outcome == "file.search"
    assert intent.query == "*.pdf"
    assert intent.directory == "/home/user/docs"


async def test_multi_step_request_id_uniqueness_avoids_collision(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "file1.txt").write_text("file 1")

    class DummyBackend:
        async def open(self, document: Document, cancellation: CancellationToken) -> Result[dict[str, Any], str]:
            return Success({})

    registry = CapabilityRegistry()
    handles = SQLiteHandleStore(tmp_path / "handles.db")
    register_file_capabilities(registry, (workspace,), handles, DummyBackend())

    service = ActionCapabilityService(registry, PolicyEngine(), InMemoryAuditLedger(), SyntheticApprovalBroker())
    service.handles = handles
    service.executions = SQLiteExecutionStore(tmp_path / "executions.db")

    assistant_actions = AssistantActions(service, None, None)

    # Step 1 turn
    turn_step1 = ConversationTurn(
        producer=ACTOR,
        session_id="s1",
        role="user",
        text="Szukaj",
        metadata={"step_id": "1"},
    )
    intent1 = ActionIntent(
        source_turn_id=turn_step1.record_id,
        outcome="file.search",
        language="pl",
        query="file1",
    )
    cand1 = await assistant_actions.dispatch_intent(turn_step1, intent1, CancellationToken())
    assert cand1 is not None
    assert "file1.txt" in cand1.text

    # Step 2 turn (same record_id, but step_id=2)
    turn_step2 = ConversationTurn(
        record_id=turn_step1.record_id,
        producer=ACTOR,
        session_id="s1",
        role="user",
        text="Szukaj ponownie",
        metadata={"step_id": "2"},
    )
    intent2 = ActionIntent(
        source_turn_id=turn_step2.record_id,
        outcome="file.search",
        language="pl",
        query="file1",
    )
    cand2 = await assistant_actions.dispatch_intent(turn_step2, intent2, CancellationToken())
    assert cand2 is not None
    assert "file1.txt" in cand2.text
    # Verify execution store recorded both distinct requests without collision
    assert cand1.metadata.get("action_result") is not None
    assert cand2.metadata.get("action_result") is not None


def _make_context_package(turn: ConversationTurn) -> AssistantContextPackage:
    producer = ProducerIdentity(producer_id="test", kind="test", version="1.0.0")
    manifest = AssistantContextManifest(
        producer=producer,
        session_id=turn.session_id,
        turn_id=turn.record_id,
        evidence_required=False,
        routing_decision="recent_conversation",
    )
    return AssistantContextPackage(
        producer=producer,
        session_id=turn.session_id,
        turn_id=turn.record_id,
        manifest=manifest,
        content={"recent_turns": [], "tool_observations": []},
    )


def _make_budget() -> InferenceBudget:
    now = _utc_now()
    return InferenceBudget(
        record_id=_new_id(),
        timestamp=now,
        producer=ProducerIdentity(producer_id="test", kind="test", version="1.0.0"),
        max_input_tokens=2048,
        max_output_tokens=512,
        max_agent_turns=5,
        max_tool_calls=5,
        max_images=0,
        max_audio_seconds=0,
        max_latency_seconds=60,
        max_provider_cost=0,
        max_ram_bytes=2 * 1024 * 1024 * 1024,
        max_vram_bytes=0,
        cancellation_deadline=now + timedelta(seconds=60),
    )


class ScriptedAgentBackend(AssistantModelBackend):
    def __init__(self, steps: list[str]) -> None:
        self.steps = steps
        self.step_idx = 0
        self.requests: list[InferenceRequest] = []
        self._state = LifecycleState.RUNNING
        self.model_name = "scripted-agent"

    @property
    def state(self) -> LifecycleState:
        return self._state

    @property
    def is_remote(self) -> bool:
        return False

    async def start(self) -> Result[LifecycleState, ActionFailure]:
        return Success(LifecycleState.RUNNING)

    async def stop(self) -> Result[LifecycleState, ActionFailure]:
        return Success(LifecycleState.STOPPED)

    async def generate(
        self, request: InferenceRequest, cancellation: CancellationToken
    ) -> Result[AssistantCandidate, ActionFailure]:
        del cancellation
        self.requests.append(request)
        if self.step_idx < len(self.steps):
            reply = self.steps[self.step_idx]
            self.step_idx += 1
        else:
            reply = "No more scripted steps."
        return Success(
            AssistantCandidate(
                text=reply,
                tokens_in=100,
                tokens_out=20,
            )
        )

    def stream(self, request: InferenceRequest, cancellation: CancellationToken) -> NoReturn:
        del request, cancellation
        raise NotImplementedError


class TrackingRecognizer:
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def recognize(
        self, turn: ConversationTurn, cancellation: CancellationToken
    ) -> Result[ActionIntent, str]:
        del cancellation
        self.calls.append(turn.text)
        return Success(
            ActionIntent(
                source_turn_id=turn.record_id,
                outcome="clarify",
                language="pl",
                clarification="W którym katalogu szukać?",
            )
        )


async def test_bounded_step_controller_bypasses_intent_recognizer_for_file_search(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "1103858973.pdf").write_text("pdf content")
    (workspace / "notes.txt").write_text("text content")

    class DummyBackend:
        async def open(self, document: Document, cancellation: CancellationToken) -> Result[dict[str, Any], str]:
            return Success({"status": "OK"})

    registry = CapabilityRegistry()
    handles = SQLiteHandleStore(tmp_path / "handles.db")
    register_file_capabilities(registry, (workspace,), handles, DummyBackend())

    service = ActionCapabilityService(registry, PolicyEngine(), InMemoryAuditLedger(), SyntheticApprovalBroker())
    service.handles = handles
    service.executions = SQLiteExecutionStore(tmp_path / "executions.db")

    recognizer = TrackingRecognizer()
    actions = AssistantActions(capabilities=service, recognizer=recognizer, store=None)
    backend = ScriptedAgentBackend(
        steps=[
            '<tool_call>call: file.search{"query": "*.pdf"}</tool_call>',
            "Znaleziono dokument: 1103858973.pdf.",
        ]
    )
    controller = BoundedStepController(
        capabilities=service,
        backend=backend,
        actions=actions,
        bypass_intent_recognizer=True,
    )

    turn = ConversationTurn(
        producer=ACTOR,
        session_id="s-search",
        role="user",
        text="Przeszukaj katalogi pod kątem plików pdf.",
        data_class=DataClass.LOCAL,
    )
    res = await controller.process_turn(
        turn, _make_context_package(turn), _make_budget(), CancellationToken()
    )
    assert isinstance(res, Success)
    cand = res.unwrap()
    assert "1103858973.pdf" in cand.text
    # Ensure recognizer was bypassed completely, preventing premature clarify prompts
    assert len(recognizer.calls) == 0
    assert cand.metadata.get("tool_calls_executed") == 1


async def test_bounded_step_controller_preserves_choice_frame_when_bypassing_recognizer(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    pdf_file = workspace / "1103858973.pdf"
    pdf_file.write_text("pdf content")

    class DummyBackend:
        async def open(self, document: Document, cancellation: CancellationToken) -> Result[dict[str, Any], str]:
            return Success({"status": "OK", "path": str(document.path)})

    registry = CapabilityRegistry()
    handles = SQLiteHandleStore(tmp_path / "handles.db")
    doc_backend = DummyBackend()
    register_file_capabilities(registry, (workspace,), handles, doc_backend)

    service = ActionCapabilityService(registry, PolicyEngine(), InMemoryAuditLedger(), SyntheticApprovalBroker())
    service.handles = handles
    service.executions = SQLiteExecutionStore(tmp_path / "executions.db")

    now = _utc_now()
    doc_inspected = inspect_document(pdf_file, (workspace,), allow_directory=True)
    assert doc_inspected is not None
    res_handle = ResourceHandle(
        producer=ACTOR,
        timestamp=now,
        actor_id=ACTOR.producer_id,
        task_id="task-choice-1",
        kind="file",
        target=str(pdf_file),
        fingerprint=doc_inspected.fingerprint,
        operations=("document.open",),
        expires_at=now + timedelta(minutes=10),
        data_class=DataClass.LOCAL,
    )
    issued = handles.issue(res_handle)
    assert isinstance(issued, Success)
    handle_token = issued.unwrap()

    store = SQLiteMemoryGraphStore(tmp_path / "memory.db")
    frame = ActiveChoiceFrame(
        record_id="frame-choice-1",
        timestamp=now,
        producer=ProducerIdentity(producer_id="assistant", kind="agent", version="1.0.0"),
        session_id="session-choice",
        profile_scope="default",
        task_id="task-choice-1",
        source_turn_id="turn-assistant-0",
        intended_capability="document.open",
        entries=(
            ChoiceEntry(index=1, handle=handle_token, label="1103858973.pdf", metadata={"path": str(pdf_file)}),
        ),
        state=ChoiceState.ACTIVE,
        expires_at=now + timedelta(minutes=10),
        data_class=DataClass.LOCAL,
    )
    prev_turn = ConversationTurn(
        record_id="turn-assistant-0",
        producer=ProducerIdentity(producer_id="assistant", kind="agent", version="1.0.0"),
        session_id="session-choice",
        role="assistant",
        text="Wybierz dokument:\n1. 1103858973.pdf",
        status="COMPLETED",
        metadata={"active_choice_frame": frame.model_dump(mode="json")},
    )
    await store.accept_turn(prev_turn)

    recognizer = TrackingRecognizer()
    actions = AssistantActions(capabilities=service, recognizer=recognizer, store=store, profile="default")
    backend = ScriptedAgentBackend(steps=[])
    controller = BoundedStepController(
        capabilities=service,
        backend=backend,
        actions=actions,
        bypass_intent_recognizer=True,
    )

    turn = ConversationTurn(
        producer=ACTOR,
        session_id="session-choice",
        role="user",
        text="1",
        data_class=DataClass.LOCAL,
    )
    res = await controller.process_turn(
        turn, _make_context_package(turn), _make_budget(), CancellationToken()
    )
    assert isinstance(res, Success)
    cand = res.unwrap()
    # Choice was executed directly without invoking intent recognizer or model loop
    assert len(recognizer.calls) == 0
    assert len(backend.requests) == 0
    assert cand.text.startswith("Otworzyłem dokument")
    assert cand.metadata.get("consumed_choice_frame_id") == "frame-choice-1"
    assert cand.metadata.get("action_result") is not None

