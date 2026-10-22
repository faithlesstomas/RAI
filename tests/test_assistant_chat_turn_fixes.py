"""Tests for tool calling, memory extraction/recall, HTTP browser fallback, and turn rollback."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
import hashlib
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from returns.result import Failure, Result, Success

from rai.actions.browser import BrowserCapability, CdpBrowserBackend
from rai.actions.handles import ResourceHandle, SQLiteHandleStore
from rai.actions.routing import match_choice_handle_by_text
from rai.assistant.extraction import SchemaConstrainedMemoryExtractor
from rai.inference.protocols import GenerationStats, InferenceResult
from rai.assistant.memory import (
    _EXPLICIT_FACT_PATTERNS,
    _modality,
    extract_memory_proposals,
)
from rai.assistant.ports import MemoryQuery
from rai.assistant.records import (
    AssistantContextManifest,
    AssistantResponse,
    ConversationTurn,
    MemoryOperation,
    MemoryOperationKind,
    MemoryRecord,
)
from rai.assistant.store import SQLiteMemoryGraphStore
from rai.assistant.tool_calling import (
    extract_model_tool_call,
    intent_from_tool_call,
    normalize_tool_capability,
    strip_tool_call_tags,
)
from rai.commands.approval import TerminalApprovalBroker, choose
from rai.kernel.ports import CancellationToken
from rai.kernel.records import (
    CapabilityRequest,
    DataClass,
    PolicyDecision,
    PolicyOutcome,
    ProducerIdentity,
    RiskClass,
    _new_id,
    _utc_now,
)


def test_tool_calling_without_call_prefix() -> None:
    text = '<tool_call>browser.search{"query": "ciemna materia"}</tool_call>'
    call = extract_model_tool_call(text)
    assert call is not None
    preamble, cap_name, args = call
    assert preamble == ""
    assert cap_name == "browser.search"
    assert args == {"query": "ciemna materia"}


def test_tool_alias_normalization() -> None:
    assert normalize_tool_capability("search.web") == "browser.search"
    assert normalize_tool_capability("read_page") == "browser.read_page"
    assert normalize_tool_capability("launch_application") == "application.launch"
    assert normalize_tool_capability("list_files") == "file.list"

    turn = ConversationTurn(
        producer=ProducerIdentity(producer_id="u", kind="user", version="1.0.0"),
        session_id="s",
        role="user",
        text="szukaj",
    )
    intent = intent_from_tool_call(turn, "search.web", {"query": "test"})
    assert intent is not None
    assert intent.outcome == "browser.search"
    assert intent.query == "test"


def test_strip_tool_call_tags() -> None:
    raw = "Oto wyniki:\n<tool_call>browser.search{\"query\": \"x\"}</tool_call>\nKoniec."
    stripped = strip_tool_call_tags(raw)
    assert "<tool_call>" not in stripped
    assert "</tool_call>" not in stripped
    assert "Oto wyniki:" in stripped
    assert "Koniec." in stripped


def test_routing_matches_choice_handle_by_url_or_title() -> None:
    frame_dict = {
        "entries": [
            {
                "index": 1,
                "handle": "h1",
                "label": "Nowe badania ciemnej materii",
                "metadata": {"url": "https://nauka.pl/ciemna-materia"},
            },
            {
                "index": 2,
                "handle": "h2",
                "label": "Kosmologia i kwanty",
                "metadata": {"url": "https://kosmos.org/badania"},
            },
        ]
    }
    # Match by partial URL
    handle = match_choice_handle_by_text(frame_dict, "https://nauka.pl/ciemna-materia")
    assert handle == "h1"

    # Match by partial label/title
    handle2 = match_choice_handle_by_text(frame_dict, "Przeczytaj artykuł o Kosmologia i kwanty")
    assert handle2 == "h2"


def test_explicit_memory_polite_regex_and_modality() -> None:
    text = "Czy możesz zapamiętać aby za każdym razem podsumowywać wyniki wyszukiwania w punktach?"
    assert any(p.search(text) for p in _EXPLICIT_FACT_PATTERNS)
    modality, negated, _ = _modality(text, 0, len(text))
    assert modality == "direct"  # "możesz" must NOT trigger "hedged" modality!
    assert not negated

    turn = ConversationTurn(
        producer=ProducerIdentity(producer_id="u", kind="user", version="1.0.0"),
        session_id="s",
        role="user",
        text=text,
    )
    proposals = extract_memory_proposals(turn.text, turn.record_id, turn.producer)
    assert len(proposals) == 1
    assert proposals[0].modality == "direct"
    assert "podsumowywać" in proposals[0].content["fact"]


@pytest.mark.asyncio
async def test_schema_constrained_memory_extractor_empty_candidates() -> None:
    class MockEngine:
        async def generate(
            self, *args: object, **kwargs: object
        ) -> Result[InferenceResult, Any]:
            return Success(
                InferenceResult(
                    text='{"candidates": [], "confidence": 0.0}',
                    stats=GenerationStats(input_tokens=10, output_tokens=5, total_time_sec=0.1, tokens_per_sec=50.0),
                )
            )

    extractor = SchemaConstrainedMemoryExtractor(
        engine=MockEngine(),
        model_name="mock",
        producer=ProducerIdentity(producer_id="p", kind="p", version="1"),
    )
    turn = ConversationTurn(
        producer=ProducerIdentity(producer_id="u", kind="user", version="1.0.0"),
        session_id="s",
        role="user",
        text="Jakieś zwykłe pytanie bez faktów.",
    )
    result = await extractor.extract(turn, CancellationToken())
    assert isinstance(result, Success)
    assert result.unwrap() == ()


@pytest.mark.asyncio
async def test_memory_recall_morphological_stem_fallback(tmp_path: Path) -> None:
    store = SQLiteMemoryGraphStore(tmp_path / "memory.sqlite3")
    await store.start()
    try:
        now = _utc_now()
        producer = ProducerIdentity(producer_id="test", kind="test", version="1.0.0")
        turn = ConversationTurn(
            record_id="t-1",
            producer=producer,
            session_id="s-1",
            role="user",
            text="Zapamiętaj to",
        )
        mem = MemoryRecord(
            record_id="mem-1",
            timestamp=now,
            producer=producer,
            kind="preference",
            topic="podsumowywanie_wyników",
            content={"fact": "Za każdym razem podsumowuj wyniki wyszukiwania w punktach"},
            source_turn_id="t-1",
            data_class=DataClass.PUBLIC,
            profile_scope="default",
            domain_scope="conversation",
            valid_from=now,
            recorded_at=now,
        )
        manifest = AssistantContextManifest(
            record_id="m-1",
            timestamp=now,
            producer=producer,
            session_id="s-1",
            turn_id="t-1",
        )
        resp = AssistantResponse(
            record_id="r-1",
            timestamp=now,
            producer=producer,
            session_id="s-1",
            turn_id="t-1",
            user_turn_id="t-1",
            request_id="req-1",
            manifest_id="m-1",
            text="Zapamiętałem.",
        )
        await store.accept_turn(turn)
        await store.commit_terminal(
            response=resp,
            manifest=manifest,
            assistant_turn=None,
            memories=(mem,),
            relations=(),
        )

        # Query using a inflected Polish word "podsumowywać" vs stored "podsumowuj"
        query = MemoryQuery(
            raw_text="Czy pamiętasz nasze preferencje odnośnie podsumowywania?",
            keywords=("podsumowywania", "preferencje"),
            profile_scope="default",
            domain_scopes=("conversation", "global"),
        )
        results = await store.retrieve_relevant_memories(
            profile_scope="default",
            query=query,
            data_classes=(DataClass.PUBLIC, DataClass.LOCAL),
        )
        assert isinstance(results, Success)
        retrieved = results.unwrap()
        assert len(retrieved) >= 1
        assert retrieved[0][0].record_id == "mem-1"
    finally:
        await store.stop()


@pytest.mark.asyncio
async def test_store_rollback_turn(tmp_path: Path) -> None:
    store = SQLiteMemoryGraphStore(tmp_path / "memory.sqlite3")
    await store.start()
    try:
        now = _utc_now()
        producer = ProducerIdentity(producer_id="test", kind="test", version="1.0.0")
        turn = ConversationTurn(
            record_id="t-abort",
            producer=producer,
            session_id="s-abort",
            role="user",
            text="To zapytanie zostanie przerwane",
        )
        await store.accept_turn(turn)

        # Check turn exists
        get_res = await store.get_turn("t-abort")
        assert isinstance(get_res, Success)
        assert get_res.unwrap() is not None

        # Rollback turn
        rb_res = await store.rollback_turn("t-abort")
        assert isinstance(rb_res, Success)
        assert rb_res.unwrap() == 1

        # Check turn is deleted
        get_after = await store.get_turn("t-abort")
        assert isinstance(get_after, Success)
        assert get_after.unwrap() is None
    finally:
        await store.stop()


@pytest.mark.asyncio
async def test_terminal_approval_broker_cancelled_on_keyboard_interrupt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    broker = TerminalApprovalBroker(profile=None)
    decision = PolicyDecision(
        producer=ProducerIdentity(producer_id="p", kind="p", version="1"),
        request_id="req-1",
        outcome=PolicyOutcome.ASK,
        risk_class=RiskClass.MODERATE,
        policy_version="1.0.0",
        reason_codes=("MODERATE_RISK_APPROVAL",),
        actor=ProducerIdentity(producer_id="p", kind="p", version="1"),
        data_class=DataClass.PUBLIC,
        target_resource="capability://browser.search",
        requested_side_effects=("network",),
        isolation="host-api",
        verification_plan=("observed",),
    )
    request = CapabilityRequest(
        producer=ProducerIdentity(producer_id="p", kind="p", version="1"),
        actor=ProducerIdentity(producer_id="p", kind="p", version="1"),
        capability="browser.search",
        arguments={"task_id": "t1", "query": "test"},
        data_class=DataClass.PUBLIC,
        target_resource="capability://browser.search",
        requested_side_effects=("network",),
        isolation="host-api",
        verification_plan=("observed",),
    )

    async def mock_choose(prompt: str, token: CancellationToken, timeout: float | None = None) -> str:
        token.cancel()
        return ""

    monkeypatch.setattr("rai.commands.approval.choose", mock_choose)
    token = CancellationToken()
    res = await broker.request_action(decision, request, token)
    assert isinstance(res, Failure)
    assert res.failure().code == "CANCELLED"


@pytest.mark.asyncio
async def test_browser_read_page_http_fallback(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    handles = SQLiteHandleStore(tmp_path / "handles.db")
    now = _utc_now()
    url = "https://example.com/ciemna-materia"
    handle = ResourceHandle(
        producer=ProducerIdentity(producer_id="p", kind="p", version="1"),
        timestamp=now,
        actor_id="p",
        task_id="t1",
        kind="url",
        target=url,
        fingerprint=hashlib.sha256(url.encode()).hexdigest(),
        operations=("browser.read_page",),
        expires_at=now + timedelta(hours=1),
        data_class=DataClass.PUBLIC,
    )
    issued_res = handles.issue(handle)
    assert isinstance(issued_res, Success)
    handle_id = issued_res.unwrap()

    backend = CdpBrowserBackend()  # No CDP endpoint configured
    cap = BrowserCapability("browser.read_page", handles, backend)

    async def mock_http_fetch(
        target_url: str, token: CancellationToken
    ) -> Result[dict[str, Any], str]:
        return Success({
            "target_id": "http-fallback",
            "observed_url": target_url,
            "title": "Strona o Ciemnej Materii",
            "text": "Tekst artykułu o ciemnej materii w kosmosie.",
            "untrusted_content": True,
        })

    monkeypatch.setattr("rai.actions.browser._fetch_http_page_snapshot", mock_http_fetch)

    token = CancellationToken()
    req = CapabilityRequest(
        producer=ProducerIdentity(producer_id="p", kind="p", version="1"),
        actor=ProducerIdentity(producer_id="p", kind="p", version="1"),
        capability="browser.read_page",
        arguments={"task_id": "t1", "handle": handle_id},
        data_class=DataClass.PUBLIC,
        target_resource=url,
        requested_side_effects=("network",),
        isolation="host-api",
        verification_plan=("observed",),
    )
    res = await cap.invoke(req, token)
    assert isinstance(res, Success)
    out = res.unwrap().output
    assert out["title"] == "Strona o Ciemnej Materii"
    assert "Tekst artykułu" in out["text"]
