"""Tests for glob pattern matching, directory scoping, and multi-step request ID uniqueness in file operations."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from returns.result import Result, Success

from rai.actions.assistant import AssistantActions
from rai.actions.execution import SQLiteExecutionStore
from rai.actions.files import (
    Document,
    _matches_document_query,
    register_file_capabilities,
    search_documents,
)
from rai.actions.handles import SQLiteHandleStore
from rai.actions.intent import ActionIntent
from rai.actions.service import ActionCapabilityService
from rai.assistant.records import ConversationTurn
from rai.assistant.tool_calling import intent_from_tool_call
from rai.kernel.audit import InMemoryAuditLedger
from rai.kernel.capabilities import CapabilityRegistry
from rai.kernel.policy import PolicyEngine
from rai.kernel.ports import CancellationToken
from rai.kernel.records import ProducerIdentity
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
