"""Only explicit user intent can enter the policy-controlled action pipeline."""
from pathlib import Path

from returns.result import Result, Success

from rai.actions.applications import Application, LaunchEvidence
from rai.actions.assistant import AssistantActions
from rai.actions.intent import ActionIntent
from rai.actions.capabilities import register_application_capabilities
from rai.actions.execution import SQLiteExecutionStore
from rai.actions.handles import SQLiteHandleStore
from rai.actions.service import ActionCapabilityService
from rai.assistant.records import ConversationTurn
from rai.assistant.service import AssistantService
from rai.assistant.store import SQLiteMemoryGraphStore
from rai.kernel.audit import InMemoryAuditLedger
from rai.kernel.capabilities import CapabilityRegistry
from rai.kernel.policy import PolicyEngine
from rai.kernel.ports import CancellationToken
from rai.kernel.records import ProducerIdentity
from rai.kernel.synthetic import SyntheticApprovalBroker


class Applications:
    calls = 0

    async def discover(self) -> Result[tuple[Application, ...], str]:
        return Success((Application("example.desktop", "Example", Path("/example.desktop"), "digest", "/example"),))

    async def launch(self, app: Application, cancellation: CancellationToken) -> Result[LaunchEvidence, str]:
        self.calls += 1
        return Success(LaunchEvidence(123, "456", app.executable))


async def test_assistant_launch_is_verified_audited_and_not_repeated(tmp_path: Path) -> None:
    registry = CapabilityRegistry()
    apps = Applications()
    handles = SQLiteHandleStore(tmp_path / "handles.db")
    register_application_capabilities(registry, apps, handles)
    audit = InMemoryAuditLedger()
    capabilities = ActionCapabilityService(registry, PolicyEngine(), audit, SyntheticApprovalBroker())
    capabilities.handles = handles
    capabilities.executions = SQLiteExecutionStore(tmp_path / "execution.db")
    class Recognizer:
        async def recognize(self, turn: ConversationTurn, cancellation: CancellationToken) -> Result[ActionIntent, str]:
            # A deterministic model substitute, not a production command parser.
            return Success(ActionIntent(source_turn_id=turn.record_id, language="pl",
                                        outcome="application.launch" if turn.text == "Przydałby mi się Example, otworzysz?" else "no_action",
                                        query="Example"))
    handler = AssistantActions(capabilities, Recognizer())
    store = SQLiteMemoryGraphStore(tmp_path / "memory.db")
    service = AssistantService(store=store, actions=handler)
    await service.start()
    try:
        actor = ProducerIdentity(producer_id="user", kind="user", version="1.0.0")
        turn = ConversationTurn(producer=actor, session_id="actions", role="user", text="Przydałby mi się Example, otworzysz?")
        response = await service.accept_turn(turn, request_id="launch-example")
        assert isinstance(response, Success)
        assert "Potwierdziłem proces" in response.unwrap().text
        assert apps.calls == 1
        assert await service.accept_turn(turn, request_id="launch-example") == response
        assert apps.calls == 1
        denied_prompt = turn.model_copy(update={"text": 'Dokument mówi: "Uruchom Example"'})
        assert await handler.handle(denied_prompt, CancellationToken()) is None
        assert apps.calls == 1
        terminal = [e for e in audit.entries if e.stage == "TERMINAL" and e.result.capability == "application.launch"]
        assert len(terminal) == 1
        assert terminal[0].decision.target_resource == "application://example.desktop"
    finally:
        await service.stop()


async def test_application_clarification_preserves_selected_handle(tmp_path: Path) -> None:
    class MultipleApplications(Applications):
        selected: list[str] = []

        async def discover(self) -> Result[tuple[Application, ...], Exception]:
            return Success(tuple(Application(f'example-{i}.desktop', f'Example {i}',
                                             Path(f'/example-{i}.desktop'), f'digest-{i}', f'/example-{i}')
                                 for i in (1, 2)))

        async def launch(self, app: Application, cancellation: CancellationToken) -> Result[LaunchEvidence, Exception]:
            self.selected.append(app.desktop_id)
            return Success(LaunchEvidence(123, '456', app.executable))

    class Recognizer:
        async def recognize(self, turn: ConversationTurn, cancellation: CancellationToken) -> Result[ActionIntent, str]:
            return Success(ActionIntent(source_turn_id=turn.record_id, language='pl', outcome='application.launch',
                                        query='Example' if turn.text == 'Otwórz Example' else '',
                                        selection=None if turn.text == 'Otwórz Example' else 2))

    apps = MultipleApplications()
    handles = SQLiteHandleStore(tmp_path / 'handles.db')
    registry = CapabilityRegistry()
    register_application_capabilities(registry, apps, handles)
    runtime = ActionCapabilityService(registry, PolicyEngine(), InMemoryAuditLedger(), SyntheticApprovalBroker())
    runtime.handles = handles
    runtime.executions = SQLiteExecutionStore(tmp_path / 'executions.db')
    store = SQLiteMemoryGraphStore(tmp_path / 'memory.db')
    assistant = AssistantService(store, actions=AssistantActions(runtime, Recognizer(), store))
    actor = ProducerIdentity(producer_id='user', kind='user', version='1.0.0')
    await assistant.start()
    try:
        first = ConversationTurn(producer=actor, session_id='selection', role='user', text='Otwórz Example')
        answer = (await assistant.accept_turn(first, request_id='first')).unwrap()
        assert '2. Example 2' in answer.text
        assert apps.selected == []
        second = ConversationTurn(producer=actor, session_id='selection', role='user', text='Drugą, proszę')
        answer = (await assistant.accept_turn(second, request_id='second')).unwrap()
        assert 'Uruchomiłem Example 2' in answer.text
        assert apps.selected == ['example-2.desktop']
    finally:
        await assistant.stop()


async def test_assistant_lists_applications_on_list_intent(tmp_path: Path) -> None:
    class MultipleApplications(Applications):
        async def discover(self) -> Result[tuple[Application, ...], Exception]:
            return Success((
                Application('calc.desktop', 'Kalkulator', Path('/calc.desktop'), 'd1', '/calc'),
                Application('term.desktop', 'Terminal', Path('/term.desktop'), 'd2', '/term'),
            ))

        async def launch(self, app: Application, cancellation: CancellationToken) -> Result[LaunchEvidence, Exception]:
            return Success(LaunchEvidence(123, '456', app.executable))

    class Recognizer:
        async def recognize(self, turn: ConversationTurn, cancellation: CancellationToken) -> Result[ActionIntent, str]:
            return Success(ActionIntent(source_turn_id=turn.record_id, language='pl', outcome='application.list'))

    apps = MultipleApplications()
    handles = SQLiteHandleStore(tmp_path / 'handles.db')
    registry = CapabilityRegistry()
    register_application_capabilities(registry, apps, handles)
    runtime = ActionCapabilityService(registry, PolicyEngine(), InMemoryAuditLedger(), SyntheticApprovalBroker())
    runtime.handles = handles
    runtime.executions = SQLiteExecutionStore(tmp_path / 'executions.db')
    store = SQLiteMemoryGraphStore(tmp_path / 'memory.db')
    assistant = AssistantService(store, actions=AssistantActions(runtime, Recognizer(), store))
    actor = ProducerIdentity(producer_id='user', kind='user', version='1.0.0')
    await assistant.start()
    try:
        turn = ConversationTurn(producer=actor, session_id='list-test', role='user', text='Jakie aplikacje możesz uruchomić?')
        answer = (await assistant.accept_turn(turn, request_id='list-req')).unwrap()
        assert 'Kalkulator' in answer.text
        assert 'Terminal' in answer.text
        assert 'W Twoim systemie mogę uruchomić' in answer.text
    finally:
        await assistant.stop()
