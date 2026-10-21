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
    AgentWorkflowState,
    BoundedStepController,
    BoundedWorkflow,
    BoundedWorkflowLimits,
    BoundedWorkflowStep,
    ResearchWorkflow,
    ResearchWorkflowLimits,
    ResearchWorkflowState,
    TaskStepController,
)
from rai.assistant.service import AssistantService
from rai.assistant.store import SQLiteMemoryGraphStore
from rai.kernel.audit import InMemoryAuditLedger
from rai.kernel.capabilities import CapabilityDescriptor, CapabilityRegistry, RegisteredCapability
from rai.kernel.policy import PolicyEngine
from rai.kernel.ports import CancellationToken
from rai.kernel.records import ActionFailure, DataClass, ProducerIdentity, RiskClass
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


async def test_execute_bounded_read_integration(tmp_path: Path) -> None:
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
    wf = controller.create_workflow(task_id="task-read-1", session_id="sess-1", query="arg search")

    search_res = await controller.execute_bounded_search(
        wf, capabilities, query="arg search", cancellation=CancellationToken(),
    )
    assert isinstance(search_res, Success)
    wf, _ = search_res.unwrap()

    first_item = wf.found_results[0]
    read_res = await controller.execute_bounded_read(
        wf,
        capabilities,
        handle=first_item["handle"],
        target_url=first_item["url"],
        approval_id="appr-read-1",
        cancellation=CancellationToken(),
        more_sources_needed=False,
    )
    assert isinstance(read_res, Success)
    wf, action_res = read_res.unwrap()
    assert wf.state == ResearchWorkflowState.SYNTHESIZING
    assert wf.page_reads_executed == 1
    assert len(wf.read_pages) == 1
    assert wf.read_pages[0]["title"] == "Page Title"

    task_usage = (await ledger.list_for_task("task-read-1")).unwrap()
    assert len(task_usage) == 2
    assert task_usage[0].processor == "browser_search"
    assert task_usage[1].processor == "browser_read_page"


async def test_run_bounded_research_e2e(tmp_path: Path) -> None:
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

    async def mock_synthesizer(workflow: ResearchWorkflow, token: CancellationToken) -> Result[str, ActionFailure]:
        return Success(f"Synthesized from {len(workflow.read_pages)} sources for '{workflow.query}'")

    res = await controller.run_bounded_research(
        task_id="task-e2e-1",
        session_id="sess-e2e",
        query="Quantum mechanics",
        capabilities=capabilities,
        synthesizer=mock_synthesizer,
        cancellation=CancellationToken(),
    )
    assert isinstance(res, Success)
    wf = res.unwrap()
    assert wf.state == ResearchWorkflowState.COMPLETE
    assert wf.searches_executed == 1
    assert wf.page_reads_executed == 2
    assert wf.final_synthesis == "Synthesized from 2 sources for 'Quantum mechanics'"

    # 1 search + 2 reads + 1 synthesis = 4 usage records
    task_usage = (await ledger.list_for_task("task-e2e-1")).unwrap()
    assert len(task_usage) == 4
    processors = [r.processor for r in task_usage]
    assert processors == ["browser_search", "browser_read_page", "browser_read_page", "assistant_synthesizer"]


async def test_assistant_service_research_workflow_e2e(tmp_path: Path) -> None:
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
    store = SQLiteMemoryGraphStore(tmp_path / "memory.db")
    service = AssistantService(store=store, step_controller=controller)

    await service.start()
    try:
        res = await service.run_research_workflow(
            query="AI safety",
            session_id="sess-research",
            capabilities=capabilities,
        )
        assert isinstance(res, Success)
        wf = res.unwrap()
        assert wf.state == ResearchWorkflowState.COMPLETE
        assert wf.searches_executed == 1
        assert wf.page_reads_executed == 2
        assert wf.final_synthesis is not None
        assert "Odpowiedź na podstawie odczytanej strony" in wf.final_synthesis
    finally:
        await service.stop()


def test_generic_workflow_initial_state_and_defaults() -> None:
    controller = BoundedStepController()
    wf = controller.create_generic_workflow(task_id="task-gen-1", session_id="sess-1", goal="Run analysis")
    assert wf.state == AgentWorkflowState.PLANNING
    assert wf.tool_calls_executed == 0
    assert len(wf.steps) == 0
    assert wf.limits.max_steps == 8
    assert wf.limits.max_tool_calls == 5


def test_generic_workflow_valid_transitions() -> None:
    controller = BoundedStepController()
    wf = controller.create_generic_workflow(task_id="task-gen-1", session_id="sess-1", goal="Run analysis")

    # PLANNING -> EVALUATING
    res1 = controller.transition_generic(wf, AgentWorkflowState.EVALUATING, action_taken="evaluate:system.echo")
    assert isinstance(res1, Success)
    wf = res1.unwrap()
    assert wf.state == AgentWorkflowState.EVALUATING

    # EVALUATING -> EXECUTING
    res2 = controller.transition_generic(wf, AgentWorkflowState.EXECUTING, action_taken="execute:system.echo")
    assert isinstance(res2, Success)
    wf = res2.unwrap()
    assert wf.state == AgentWorkflowState.EXECUTING

    # EXECUTING -> OBSERVING
    res3 = controller.transition_generic(wf, AgentWorkflowState.OBSERVING, action_taken="observed:system.echo")
    assert isinstance(res3, Success)
    wf = res3.unwrap()
    assert wf.state == AgentWorkflowState.OBSERVING

    # OBSERVING -> SYNTHESIZING
    res4 = controller.transition_generic(wf, AgentWorkflowState.SYNTHESIZING, action_taken="begin_synthesis")
    assert isinstance(res4, Success)
    wf = res4.unwrap()
    assert wf.state == AgentWorkflowState.SYNTHESIZING

    # SYNTHESIZING -> COMPLETE
    res5 = controller.transition_generic(wf, AgentWorkflowState.COMPLETE, action_taken="synthesis_completed")
    assert isinstance(res5, Success)
    wf = res5.unwrap()
    assert wf.state == AgentWorkflowState.COMPLETE
    assert len(wf.steps) == 5


def test_generic_workflow_invalid_transitions() -> None:
    controller = BoundedStepController()
    wf = controller.create_generic_workflow(task_id="task-gen-1", session_id="sess-1", goal="Run analysis")

    # Cannot jump PLANNING -> OBSERVING
    bad1 = controller.transition_generic(wf, AgentWorkflowState.OBSERVING, action_taken="illegal")
    assert isinstance(bad1, Failure)
    assert "INVALID_TRANSITION" in bad1.failure()

    # Once COMPLETE, cannot transition back to PLANNING
    complete_wf = wf.model_copy(update={"state": AgentWorkflowState.COMPLETE})
    bad2 = controller.transition_generic(complete_wf, AgentWorkflowState.PLANNING, action_taken="illegal")
    assert isinstance(bad2, Failure)
    assert "INVALID_TRANSITION" in bad2.failure()


def test_generic_workflow_step_limits_exceeded() -> None:
    controller = BoundedStepController()
    limits = BoundedWorkflowLimits(max_steps=2)
    wf = controller.create_generic_workflow(task_id="task-gen-1", session_id="sess-1", goal="Run analysis", limits=limits)

    res1 = controller.transition_generic(wf, AgentWorkflowState.EVALUATING, action_taken="step1")
    assert isinstance(res1, Success)
    wf = res1.unwrap()

    res2 = controller.transition_generic(wf, AgentWorkflowState.EXECUTING, action_taken="step2")
    assert isinstance(res2, Success)
    wf = res2.unwrap()

    # Step 3 exceeds limit of 2
    res3 = controller.transition_generic(wf, AgentWorkflowState.OBSERVING, action_taken="step3")
    assert isinstance(res3, Failure)
    assert "STEP_LIMIT_EXCEEDED" in res3.failure()


async def test_generic_workflow_execute_generic_step_and_approvals(tmp_path: Path) -> None:
    registry = CapabilityRegistry()
    registry.register(
        RegisteredCapability(
            CapabilityDescriptor(
                name="system.echo",
                description="Echo input text",
                input_schema={"type": "object", "properties": {"message": {"type": "string"}}},
                risk_class=RiskClass.LOW,
                side_effects=(),
                isolation="host-api",
                verification_plan=("observed",),
            ),
            handler=lambda args: {"echoed": args.get("message", "")},
        )
    )
    handles = SQLiteHandleStore(tmp_path / "handles.db")
    audit = InMemoryAuditLedger()
    capabilities = ActionCapabilityService(registry, PolicyEngine(), audit, SyntheticApprovalBroker())
    capabilities.handles = handles
    capabilities.executions = SQLiteExecutionStore(tmp_path / "executions.db")

    ledger = SyntheticUsageLedger()
    controller = BoundedStepController(ledger=ledger)
    wf = controller.create_generic_workflow(task_id="task-step-1", session_id="sess-1", goal="Echo hello")

    # Step execution with approval
    res = await controller.execute_generic_step(
        workflow=wf,
        capabilities=capabilities,
        capability_name="system.echo",
        arguments={"message": "hello RAI"},
        cancellation=CancellationToken(),
        approval_id="appr-echo-1",
    )
    assert isinstance(res, Success)
    updated_wf, action_res = res.unwrap()
    assert updated_wf.state == AgentWorkflowState.OBSERVING
    assert updated_wf.tool_calls_executed == 1
    assert "appr-echo-1" in updated_wf.approvals_used
    assert action_res.output == {"echoed": "hello RAI"}
    assert len(updated_wf.observations) == 1

    # Deliberate next step: OBSERVING -> PLANNING
    plan_tx = controller.transition_generic(updated_wf, AgentWorkflowState.PLANNING, action_taken="deliberate")
    assert isinstance(plan_tx, Success)
    updated_wf = plan_tx.unwrap()

    # Re-using the same approval token must fail
    reuse_res = await controller.execute_generic_step(
        workflow=updated_wf,
        capabilities=capabilities,
        capability_name="system.echo",
        arguments={"message": "hello again"},
        cancellation=CancellationToken(),
        approval_id="appr-echo-1",
    )
    assert isinstance(reuse_res, Failure)
    assert reuse_res.failure().code == "APPROVAL_ALREADY_USED"


async def test_generic_workflow_agentic_loop_e2e(tmp_path: Path) -> None:
    registry = CapabilityRegistry()
    registry.register(
        RegisteredCapability(
            CapabilityDescriptor(
                name="calc.add",
                description="Add numbers",
                input_schema={"type": "object", "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}}},
                risk_class=RiskClass.LOW,
                side_effects=(),
                isolation="host-api",
                verification_plan=("observed",),
            ),
            handler=lambda args: {"sum": args.get("a", 0) + args.get("b", 0)},
        )
    )
    handles = SQLiteHandleStore(tmp_path / "handles.db")
    audit = InMemoryAuditLedger()
    capabilities = ActionCapabilityService(registry, PolicyEngine(), audit, SyntheticApprovalBroker())
    capabilities.handles = handles
    capabilities.executions = SQLiteExecutionStore(tmp_path / "executions.db")

    ledger = SyntheticUsageLedger()
    controller = BoundedStepController(ledger=ledger)

    turn = 0

    async def mock_decider(wf: BoundedWorkflow, ct: CancellationToken) -> Result[dict[str, Any], ActionFailure]:
        nonlocal turn
        turn += 1
        if turn == 1:
            return Success({
                "action_type": "tool",
                "capability": "calc.add",
                "arguments": {"a": 5, "b": 7},
            })
        return Success({
            "action_type": "finish",
            "text": f"The calculated result is {wf.observations[0]['sum']}",
        })

    loop_res = await controller.run_agentic_loop(
        task_id="task-loop-1",
        session_id="sess-loop",
        goal="Calculate 5 + 7",
        capabilities=capabilities,
        step_decider=mock_decider,
        cancellation=CancellationToken(),
    )
    assert isinstance(loop_res, Success)
    wf = loop_res.unwrap()
    assert wf.state == AgentWorkflowState.COMPLETE
    assert wf.tool_calls_executed == 1
    assert wf.final_synthesis == "The calculated result is 12"
    assert len(wf.steps) > 0


async def test_assistant_service_bounded_task_e2e(tmp_path: Path) -> None:
    registry = CapabilityRegistry()
    registry.register(
        RegisteredCapability(
            CapabilityDescriptor(
                name="system.info",
                description="Get system info",
                input_schema={"type": "object", "properties": {}},
                risk_class=RiskClass.LOW,
                side_effects=(),
                isolation="host-api",
                verification_plan=("observed",),
            ),
            handler=lambda req: {"os": "Linux", "hostname": "rai-node"},
        )
    )
    handles = SQLiteHandleStore(tmp_path / "handles.db")
    audit = InMemoryAuditLedger()
    capabilities = ActionCapabilityService(registry, PolicyEngine(), audit, SyntheticApprovalBroker())
    capabilities.handles = handles
    capabilities.executions = SQLiteExecutionStore(tmp_path / "executions.db")

    ledger = SyntheticUsageLedger()
    controller = BoundedStepController(ledger=ledger)
    store = SQLiteMemoryGraphStore(tmp_path / "memory.db")
    service = AssistantService(store=store, step_controller=controller)

    await service.start()
    try:
        res = await service.run_bounded_task(
            goal="Tell me system information",
            session_id="sess-task-e2e",
            capabilities=capabilities,
        )
        assert isinstance(res, Success)
        wf = res.unwrap()
        assert wf.state == AgentWorkflowState.COMPLETE
        assert wf.final_synthesis is not None
    finally:
        await service.stop()


def test_large_application_list_compacting_and_backend_representation() -> None:
    from rai.actions.workflow import _compact_observation_content
    from rai.assistant.backends.local import LocalAssistantBackend

    apps = [
        {
            "name": f"App {i}",
            "desktop_id": f"app_{i}.desktop",
            "categories": ["AudioVideo", "Player", "Video"] if i % 2 == 0 else ["Utility"],
            "comment": f"Description for app {i}",
            "generic_name": "Media Player" if i % 2 == 0 else "Tool",
            "handle": f"handle_{i}",
            "expires_at": "2026-10-21T12:00:00Z",
        }
        for i in range(125)
    ]
    raw_output = {"total_count": len(apps), "applications": apps}
    compacted = _compact_observation_content(raw_output)

    # All 125 applications must remain in the compacted output (not dropped)
    assert compacted["total_count"] == 125
    assert len(compacted["applications"]) == 125
    assert len(compacted["application_names"]) == 125
    # Technical fields removed, domain fields preserved
    first = compacted["applications"][0]
    assert "handle" not in first
    assert "expires_at" not in first
    assert first["name"] == "App 0"
    assert first["categories"] == ["AudioVideo", "Player", "Video"]

    # Local backend representation should keep all apps without truncation
    backend = LocalAssistantBackend()
    repr_str = backend._bound_content_repr(compacted, max_len=backend.max_observation_characters)
    assert "truncated" not in repr_str
    assert "App 124" in repr_str

