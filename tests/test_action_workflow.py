# ruff: noqa: PLR2004, PLR0915
"""Unit tests for bounded research workflow and task step controller."""

from pathlib import Path
from typing import Any

from returns.result import Failure, Result, Success

from rai.actions.browser import register_browser_capabilities
from rai.actions.execution import SQLiteExecutionStore
from rai.actions.handles import SQLiteHandleStore
from rai.actions.service import ActionCapabilityService
from rai.actions.workflow import (
    ResearchWorkflowLimits,
    ResearchWorkflowState,
    TaskStepController,
)
from rai.kernel.audit import InMemoryAuditLedger
from rai.kernel.capabilities import CapabilityRegistry
from rai.kernel.policy import PolicyEngine
from rai.kernel.ports import CancellationToken
from rai.kernel.records import DataClass, ProducerIdentity
from rai.kernel.synthetic import SyntheticApprovalBroker, SyntheticUsageLedger


class SimpleMockBrowserBackend:
    async def search(self, query: str, token: CancellationToken) -> Result[tuple[dict[str, Any], ...], str]:
        return Success((
            {"name": "Result 1", "url": "https://example.com/1", "snippet": "Snippet 1"},
            {"name": "Result 2", "url": "https://example.com/2", "snippet": "Snippet 2"},
        ))

    async def open(self, url: str, token: CancellationToken) -> Result[dict[str, Any], str]:
        return Success({"target_id": "tab-1", "observed_url": url, "frame_id": "frame-1"})

    async def read(self, url: str, token: CancellationToken) -> Result[dict[str, Any], str]:
        return Success({"target_id": "tab-1", "observed_url": url, "title": "Page Title", "text": "Page body text"})


def test_workflow_initial_state_and_defaults() -> None:
    controller = TaskStepController()
    wf = controller.create_workflow(task_id="task-1", session_id="sess-1", query="Quantum computing")
    assert wf.state == ResearchWorkflowState.PLANNING
    assert wf.searches_executed == 0
    assert wf.page_reads_executed == 0
    assert len(wf.steps) == 0
    assert wf.limits.max_searches == 1
    assert wf.limits.max_page_reads == 2


def test_valid_full_workflow_lifecycle() -> None:
    controller = TaskStepController()
    wf = controller.create_workflow(task_id="task-1", session_id="sess-1", query="Quantum computing")

    # 1. PLANNING -> SEARCHING
    res1 = controller.transition_to_search(wf, approval_id="appr-search-1")
    assert isinstance(res1, Success)
    wf = res1.unwrap()
    assert wf.state == ResearchWorkflowState.SEARCHING
    assert wf.search_approval_id == "appr-search-1"

    # 2. SEARCHING -> SELECTING_SOURCES
    results = ({"name": "R1", "url": "https://example.com/1"}, {"name": "R2", "url": "https://example.com/2"})
    res2 = controller.record_search_completed(wf, results, tokens_in=50, tokens_out=100, latency_ms=250.0)
    assert isinstance(res2, Success)
    wf = res2.unwrap()
    assert wf.state == ResearchWorkflowState.SELECTING_SOURCES
    assert wf.searches_executed == 1
    assert len(wf.found_results) == 2

    # 3. SELECTING_SOURCES -> READING_PAGE (1st page)
    res3 = controller.transition_to_reading_page(wf, target_url="https://example.com/1", approval_id="appr-read-1")
    assert isinstance(res3, Success)
    wf = res3.unwrap()
    assert wf.state == ResearchWorkflowState.READING_PAGE

    # 4. READING_PAGE -> SELECTING_SOURCES (need 2nd page)
    res4 = controller.record_page_read_completed(
        wf, page_content={"title": "Page 1", "text": "Content 1"}, more_sources_needed=True,
    )
    assert isinstance(res4, Success)
    wf = res4.unwrap()
    assert wf.state == ResearchWorkflowState.SELECTING_SOURCES
    assert wf.page_reads_executed == 1

    # 5. SELECTING_SOURCES -> READING_PAGE (2nd page)
    res5 = controller.transition_to_reading_page(wf, target_url="https://example.com/2", approval_id="appr-read-2")
    assert isinstance(res5, Success)
    wf = res5.unwrap()
    assert wf.state == ResearchWorkflowState.READING_PAGE

    # 6. READING_PAGE -> SYNTHESIZING (limit 2 pages reached)
    res6 = controller.record_page_read_completed(
        wf, page_content={"title": "Page 2", "text": "Content 2"}, more_sources_needed=True,
    )
    assert isinstance(res6, Success)
    wf = res6.unwrap()
    assert wf.state == ResearchWorkflowState.SYNTHESIZING
    assert wf.page_reads_executed == 2

    # 7. SYNTHESIZING -> COMPLETE
    res7 = controller.transition_to_complete(wf, final_synthesis="Synthesized knowledge about quantum computing.")
    assert isinstance(res7, Success)
    wf = res7.unwrap()
    assert wf.state == ResearchWorkflowState.COMPLETE
    assert wf.final_synthesis is not None
    assert len(wf.steps) == 7


def test_invalid_state_transitions() -> None:
    controller = TaskStepController()
    wf = controller.create_workflow(task_id="task-1", session_id="sess-1", query="Quantum computing")

    # Cannot jump from PLANNING to COMPLETE
    bad_comp = controller.transition_to_complete(wf, "premature answer")
    assert isinstance(bad_comp, Failure)
    assert "INVALID_TRANSITION" in bad_comp.failure()

    # Cannot jump from PLANNING to READING_PAGE
    bad_read = controller.transition_to_reading_page(wf, "https://example.com", "appr-1")
    assert isinstance(bad_read, Failure)
    assert "INVALID_TRANSITION" in bad_read.failure()


def test_hard_limits_enforced() -> None:
    controller = TaskStepController()
    custom_limits = ResearchWorkflowLimits(max_searches=1, max_page_reads=1, max_total_steps=4)
    wf = controller.create_workflow(task_id="task-1", session_id="sess-1", query="test", limits=custom_limits)

    wf = controller.transition_to_search(wf, "appr-s1").unwrap()
    wf = controller.record_search_completed(wf, ({"url": "https://example.com"},)).unwrap()

    # Attempting second search when max_searches=1 must fail
    wf_planning_mock = wf.model_copy(update={"state": ResearchWorkflowState.PLANNING})
    second_search = controller.transition_to_search(wf_planning_mock, "appr-s2")
    assert isinstance(second_search, Failure)
    assert second_search.failure() == "SEARCH_LIMIT_EXCEEDED"

    # Reading 1st page
    wf = controller.transition_to_reading_page(wf, "https://example.com", "appr-r1").unwrap()
    wf = controller.record_page_read_completed(wf, {"title": "P1"}, more_sources_needed=True).unwrap()
    # Since max_page_reads=1, it must automatically transition to SYNTHESIZING
    assert wf.state == ResearchWorkflowState.SYNTHESIZING


def test_multistage_approval_isolation() -> None:
    controller = TaskStepController()
    wf = controller.create_workflow(task_id="task-1", session_id="sess-1", query="test")
    wf = controller.transition_to_search(wf, approval_id="appr-search-shared").unwrap()
    wf = controller.record_search_completed(wf, ()).unwrap()

    # Cannot reuse the search approval token for reading an external web page
    reused = controller.transition_to_reading_page(wf, "https://example.com/1", approval_id="appr-search-shared")
    assert isinstance(reused, Failure)
    assert "APPROVAL_REUSE_FORBIDDEN" in reused.failure()

    # Reading with valid distinct approval works
    wf = controller.transition_to_reading_page(wf, "https://example.com/1", approval_id="appr-read-1").unwrap()
    wf = controller.record_page_read_completed(wf, {"title": "P1"}, more_sources_needed=True).unwrap()

    # Reusing the first page read approval token for a second page read fails
    dup_token = controller.transition_to_reading_page(wf, "https://example.com/2", approval_id="appr-read-1")
    assert isinstance(dup_token, Failure)
    assert "APPROVAL_ALREADY_USED" in dup_token.failure()


async def test_unified_usage_ledger_recording() -> None:
    ledger = SyntheticUsageLedger()
    controller = TaskStepController(ledger=ledger)
    wf = controller.create_workflow(task_id="task-usage-1", session_id="sess-1", query="test")

    # Record intent recognition usage
    u1 = await controller.record_usage(
        wf, processor="intent_recognizer", backend="lemonade", model="Gemma-4-E2B",
        tokens_in=120, tokens_out=15, tool_calls=0, latency_ms=45.0,
    )
    assert isinstance(u1, Success)

    # Record tool execution usage
    u2 = await controller.record_usage(
        wf, processor="browser_search", backend="browser", model="ddgs",
        tokens_in=0, tokens_out=0, tool_calls=1, latency_ms=320.0,
    )
    assert isinstance(u2, Success)

    # Record final synthesis generation usage
    u3 = await controller.record_usage(
        wf, processor="assistant_synthesizer", backend="lemonade", model="Gemma-4-E2B",
        tokens_in=450, tokens_out=180, tool_calls=0, latency_ms=850.0,
    )
    assert isinstance(u3, Success)

    # Verify ledger has all records for this task
    records = (await ledger.list_for_task("task-usage-1")).unwrap()
    assert len(records) == 3
    total_tokens_in = sum(r.input_tokens for r in records)
    total_tokens_out = sum(r.output_tokens for r in records)
    total_tools = sum(r.tool_calls for r in records)
    assert total_tokens_in == 570
    assert total_tokens_out == 195
    assert total_tools == 1


async def test_execute_bounded_search_integration(tmp_path: Path) -> None:
    registry = CapabilityRegistry()
    handles = SQLiteHandleStore(tmp_path / "handles.db")
    browser = SimpleMockBrowserBackend()
    register_browser_capabilities(registry, handles, browser)

    audit = InMemoryAuditLedger()
    capabilities = ActionCapabilityService(registry, PolicyEngine(), audit, SyntheticApprovalBroker())
    capabilities.handles = handles
    capabilities.executions = SQLiteExecutionStore(tmp_path / "executions.db")

    ledger = SyntheticUsageLedger()
    controller = TaskStepController(ledger=ledger)
    wf = controller.create_workflow(task_id="task-exec-1", session_id="sess-1", query="arg search")

    res = await controller.execute_bounded_search(
        wf, capabilities, query="arg search", cancellation=CancellationToken(),
        tokens_in=30, tokens_out=10, latency_ms=100.0,
    )
    assert isinstance(res, Success)
    updated_wf, action_res = res.unwrap()
    assert updated_wf.state == ResearchWorkflowState.SELECTING_SOURCES
    assert updated_wf.searches_executed == 1
    assert len(updated_wf.found_results) == 2

    # Check ledger recorded search
    task_usage = (await ledger.list_for_task("task-exec-1")).unwrap()
    assert len(task_usage) == 1
    assert task_usage[0].tool_calls == 1
