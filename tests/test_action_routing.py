"""Routing resolves finite model choices to runtime-issued handles before any invocation."""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from returns.result import Failure, Success

from rai.actions.capabilities import failure, result
from rai.actions.intent import ActionIntent
from rai.actions.routing import route_action
from rai.assistant.records import ConversationTurn
from rai.kernel.capabilities import CapabilityDescriptor
from rai.kernel.ports import CancellationToken
from rai.kernel.records import ProducerIdentity, RiskClass

ACTOR = ProducerIdentity(producer_id="user", kind="user", version="1.0.0")
NAMES = (
    "browser.search", "browser.open_result", "browser.read_page",
    "system.volume.get", "system.volume.set", "process.inspect",
)


def descriptor(name: str) -> CapabilityDescriptor:
    return CapabilityDescriptor(
        name=name, description="Routing fixture", input_schema={"type": "object"},
        risk_class=RiskClass.LOW, side_effects=(), isolation="host-api", verification_plan=("observed",),
    )


class Service:
    """Capability service double that records requests and returns scripted outputs."""

    def __init__(self, outputs: dict[str, Any], missing: tuple[str, ...] = (), failing: tuple[str, ...] = ()) -> None:
        self.registry = SimpleNamespace(descriptor=lambda name: None if name in missing else descriptor(name))
        self.outputs = outputs
        self.failing = failing
        self.requests: list[Any] = []

    async def invoke(self, request: Any, token: CancellationToken) -> tuple[Any, Any]:
        self.requests.append(request)
        decision = SimpleNamespace(record_id=f"decision:{request.capability}")
        if request.capability in self.failing:
            return decision, Failure(failure(request, "BACKEND_FAILED"))
        return decision, Success(result(request, self.outputs[request.capability], {"observed": True}))


class Store:
    def __init__(self, metadata: dict[str, Any] | None = None, scope: str = "default") -> None:
        reply = SimpleNamespace(record_id="previous", role="assistant", data_class=turn().data_class,
                                metadata={"profile_scope": scope, "action": metadata or {}})
        self.chain = Success([reply])

    async def get_recent_reply_chain(self, **_: Any) -> Any:
        return self.chain


def turn() -> ConversationTurn:
    return ConversationTurn(producer=ACTOR, session_id="routing", role="user", text="request")


def intent(outcome: str, language: str = "en", **fields: Any) -> ActionIntent:
    return ActionIntent(source_turn_id=turn().record_id, outcome=outcome, language=language, **fields)


async def route(service: Service, store: Any, value: ActionIntent) -> Any:
    return await route_action(service, store, "default", turn(), value, CancellationToken())  # type: ignore[arg-type]


async def test_search_lists_numbered_results_and_keeps_authority_in_metadata() -> None:
    entries = [{"name": "Python", "handle": "h1"}, {"name": "Docs", "handle": "h2"}]
    service = Service({"browser.search": {"results": entries}})
    candidate = await route(service, None, intent("browser.search", query="python"))
    assert candidate.text == "1. Python\n2. Docs"
    assert candidate.metadata["action_choices"]["kind"] == "url"
    assert candidate.metadata["remote_tokens"] == 0
    assert candidate.metadata["policy_decision_id"] == "decision:browser.search"
    assert "action_proposal" not in candidate.metadata
    assert service.requests[0].arguments["query"] == "python"


async def test_empty_results_are_reported_in_user_language() -> None:
    service = Service({"browser.search": {"results": []}})
    assert (await route(service, None, intent("browser.search", query="x"))).text == "No results."
    assert (await route(service, None, intent("browser.search", "pl", query="x"))).text == "Brak wyników."


async def test_process_inspection_shows_pids_and_process_choices() -> None:
    service = Service({"process.inspect": {"processes": [{"name": "bash", "pid": 7}]}})
    candidate = await route(service, None, intent("process.inspect", query="bash"))
    assert candidate.text == "1. bash (PID 7)"
    assert candidate.metadata["action_choices"]["kind"] == "process"


async def test_selected_result_uses_previous_handle_not_model_text() -> None:
    previous = {"action_choices": {"kind": "url", "task_id": "task-1", "entries": [{"name": "A", "handle": "ha"},
                                                                                  {"name": "B", "handle": "hb"}]}}
    service = Service({"browser.read_page": {"title": "Title", "text": "Body"},
                       "browser.open_result": {}})
    read = await route(service, Store(previous), intent("browser.read_page", selection=2))
    assert read.text == "Title\n\nBody"
    assert read.metadata["untrusted_content"] is True
    assert read.metadata["action_proposal"]["resource_handle"] == "hb"
    assert service.requests[0].arguments == {"task_id": "task-1", "handle": "hb"}
    opened = await route(service, Store(previous), intent("browser.open_result", "pl", selection=1))
    assert opened.text == "Wykonanie potwierdzone."
    assert (await route(service, Store(previous), intent("browser.open_result", selection=1))).text == "Execution verified."


async def test_missing_stale_or_wrong_kind_selection_asks_for_search_first() -> None:
    service = Service({})
    wrong_kind = {"action_choices": {"kind": "file", "task_id": "t", "entries": [{"name": "A", "handle": "h"}]}}
    for store in (None, Store(), Store(wrong_kind), Store({"action_choices": {"kind": "url", "task_id": "t", "entries": []}})):
        candidate = await route(service, store, intent("browser.read_page", selection=1))
        assert candidate.text.startswith("Which result?")
    polish = await route(service, None, intent("browser.read_page", "pl", selection=1))
    assert polish.text.startswith("Który wynik")
    assert not service.requests


async def test_other_profile_context_cannot_supply_authority() -> None:
    previous = {"action_choices": {"kind": "url", "task_id": "t", "entries": [{"name": "A", "handle": "h"}]}}
    candidate = await route(Service({}), Store(previous, scope="other"), intent("browser.open_result", selection=1))
    assert candidate.text.startswith("Which result?")


async def test_volume_set_reads_device_handle_before_changing_it() -> None:
    service = Service({"system.volume.get": {"handle": "device-1", "volumes": [30]}, "system.volume.set": {}})
    candidate = await route(service, None, intent("system.volume.set", percent=40))
    assert [request.capability for request in service.requests] == ["system.volume.get", "system.volume.set"]
    assert service.requests[1].arguments["handle"] == "device-1"
    assert service.requests[1].arguments["percent"] == 40
    assert candidate.metadata["action_proposal"]["resource_handle"] == "device-1"
    assert candidate.text == "Execution verified."


async def test_volume_read_reports_levels() -> None:
    service = Service({"system.volume.get": {"volumes": [30, 55]}})
    assert (await route(service, None, intent("system.volume.get"))).text == "Volume: 30%, 55%"
    assert (await route(service, None, intent("system.volume.get", "pl"))).text == "Głośność: 30%, 55%"


async def test_volume_set_stops_when_current_device_cannot_be_read() -> None:
    for kwargs in ({"failing": ("system.volume.get",)}, {"missing": ("system.volume.get",)}):
        service = Service({}, **kwargs)
        candidate = await route(service, None, intent("system.volume.set", percent=10))
        assert candidate.text == "Could not read the audio device."
        assert all(request.capability == "system.volume.get" for request in service.requests)
    polish = await route(Service({}, missing=("system.volume.get",)), None, intent("system.volume.set", "pl", percent=10))
    assert polish.text == "Nie udało się odczytać urządzenia audio."


async def test_unavailable_capability_is_reported_without_invocation() -> None:
    service = Service({}, missing=("browser.search",))
    assert (await route(service, None, intent("browser.search", query="x"))).text == "Capability unavailable."
    assert (await route(service, None, intent("browser.search", "pl", query="x"))).text == "Funkcja jest niedostępna."
    assert not service.requests


async def test_failed_execution_is_never_presented_as_success() -> None:
    service = Service({}, failing=("browser.search",))
    candidate = await route(service, None, intent("browser.search", query="x"))
    assert candidate.text == "Execution not verified: BACKEND_FAILED."
    assert candidate.metadata["action_result"]["code"] == "BACKEND_FAILED"
    assert "action_choices" not in candidate.metadata
    polish = await route(service, None, intent("browser.search", "pl", query="x"))
    assert polish.text == "Nie potwierdzam wykonania: BACKEND_FAILED."
