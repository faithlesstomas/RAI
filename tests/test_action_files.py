"""File search and selection preserve allowed-root identity across user turns."""
from pathlib import Path

from returns.result import Failure, Result, Success

from rai.actions.assistant import AssistantActions
from rai.actions.capabilities import register_application_capabilities
from rai.actions.execution import SQLiteExecutionStore
from rai.actions.files import Document, inspect_document, register_file_capabilities, search_documents
from rai.actions.handles import SQLiteHandleStore
from rai.actions.intent import ActionIntent
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
from rai.kernel.transport import normalize_request


def test_search_excludes_symlinks_executables_and_unapproved_roots(tmp_path: Path) -> None:
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    (allowed / "notes.txt").write_text("hello")
    outside = tmp_path / "outside.txt"
    outside.write_text("private")
    (allowed / "shortcut.txt").symlink_to(outside)
    executable = allowed / "executable.txt"
    executable.write_text("script")
    executable.chmod(0o700)
    assert [item.path.name for item in search_documents((allowed,), ".txt")] == ["notes.txt"]
    assert inspect_document(outside, (allowed,)) is None


async def test_followup_opens_selected_result_and_rejects_replacement(tmp_path: Path) -> None:
    documents = tmp_path / "documents"
    documents.mkdir()
    for name in ("note-a.txt", "note-b.txt"):
        (documents / name).write_text(name)

    class Backend:
        opened = []

        async def open(self, document: Document, cancellation: CancellationToken) -> Result[dict, str]:
            self.opened.append(document.path.name)
            return Success({"open_file_descriptor": True, "document_fingerprint": document.fingerprint})

    class Recognizer:
        async def recognize(self, turn: ConversationTurn, cancellation: CancellationToken) -> Result[ActionIntent, str]:
            return Success(ActionIntent(source_turn_id=turn.record_id, language="pl",
                                        outcome="file.search" if turn.text == "Poszukaj moich notatek" else "document.open",
                                        query="note", selection=2))

    handles = SQLiteHandleStore(tmp_path / "handles.db")
    backend = Backend()
    registry = CapabilityRegistry()
    register_file_capabilities(registry, (documents,), handles, backend)
    capabilities = ActionCapabilityService(registry, PolicyEngine(), InMemoryAuditLedger(), SyntheticApprovalBroker())
    capabilities.handles = handles
    capabilities.executions = SQLiteExecutionStore(tmp_path / "execution.db")
    store = SQLiteMemoryGraphStore(tmp_path / "memory.db")
    assistant = AssistantService(store, actions=AssistantActions(capabilities, Recognizer(), store))
    actor = ProducerIdentity(producer_id="user", kind="user", version="1.0.0")
    await assistant.start()
    try:
        first = ConversationTurn(producer=actor, session_id="files", role="user", text="Poszukaj moich notatek")
        response = (await assistant.accept_turn(first, request_id="search")).unwrap()
        assert "2. note-b.txt" in response.text
        followup = ConversationTurn(producer=actor, session_id="files", role="user", text="Otwórz drugą, proszę")
        response = (await assistant.accept_turn(followup, request_id="open")).unwrap()
        assert "potwierdzone" in response.text
        assert backend.opened == ["note-b.txt"]
        await assistant.accept_turn(followup, request_id="open")
        assert backend.opened == ["note-b.txt"]
        listing = normalize_request(registry.descriptor("file.search"), {"task_id": "swap", "query": "note-b"}, actor=actor)
        listed = (await capabilities.invoke(listing))[1].unwrap()
        handle = listed.output["documents"][0]["handle"]
        (documents / "note-b.txt").unlink()
        (documents / "note-b.txt").symlink_to(documents / "note-a.txt")
        opening = normalize_request(registry.descriptor("document.open"), {"task_id": "swap", "handle": handle}, actor=actor)
        outcome = (await capabilities.invoke(opening))[1]
        assert isinstance(outcome, Failure)
        assert outcome.failure().code == "STALE_RESOURCE"
        assert backend.opened == ["note-b.txt"]
    finally:
        await assistant.stop()


def test_absolute_document_query_stays_within_allowed_roots(tmp_path):
    allowed = tmp_path / 'allowed'
    allowed.mkdir()
    document = allowed / 'note.md'
    document.write_text('test')
    outside = tmp_path / 'outside.md'
    outside.write_text('outside')
    assert search_documents((allowed,), str(document))[0].path == document
    assert search_documents((allowed,), str(allowed / '..' / 'outside.md')) == ()
