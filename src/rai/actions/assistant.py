"""Explicit user action intents; retrieved/model text has no execution authority."""
from __future__ import annotations

from returns.result import Failure

from rai.assistant.records import AssistantCandidate, ConversationTurn
from rai.assistant.ports import MemoryGraphStore
from .dialogue import previous_action
from rai.kernel.ports import CancellationToken
from rai.kernel.service import CapabilityService
from rai.kernel.transport import normalize_request

from .records import ActionProposal
from .intent import IntentRecognizer, ActionIntent

class AssistantActions:
    """Route explicit current-turn requests through the same capability service."""

    def __init__(self, capabilities: CapabilityService, recognizer: IntentRecognizer,
                 store: MemoryGraphStore | None = None, profile: str = "default") -> None:
        self.capabilities = capabilities
        self.recognizer = recognizer
        self.store, self.profile = store, profile

    async def handle(  # noqa: PLR0911, PLR0912
        self, turn: ConversationTurn, cancellation: CancellationToken,
    ) -> AssistantCandidate | None:
        if turn.role != "user":
            return None
        recognized = await self.recognizer.recognize(turn, cancellation)
        if isinstance(recognized, Failure):
            return AssistantCandidate(text="Nie udało się bezpiecznie rozpoznać intencji. Nie wykonałem działania.",
                                      metadata={"intent_failure": recognized.failure()})
        intent = recognized.unwrap()
        if intent.outcome == "no_action":
            return None
        if intent.outcome == "clarify":
            return AssistantCandidate(
                text=intent.question + "\n" + "\n".join(f"{i}. {option}" for i, option in enumerate(intent.options, 1)),
                metadata={"action_clarification": intent.model_dump(mode="json")},
            )
        def message(polish: str, english: str) -> str:
            return polish if intent.language == "pl" else english

        if intent.outcome in {"file.search", "document.open"}:
            return await self._document_intent(turn, intent, cancellation)
        if intent.outcome.startswith(("browser.", "system.volume.", "process.")):
            from .routing import route_action  # noqa: PLC0415
            return await route_action(self.capabilities, self.store, self.profile, turn, intent, cancellation)
        query = intent.query.strip()
        task_id = f"action:{turn.record_id}"
        if intent.selection is not None:
            previous = await previous_action(self.store, turn, self.profile)
            choices = previous.get("action_choices", {})
            entries = choices.get("entries", ())
            if choices.get("kind") != "application" or intent.selection > len(entries):
                return AssistantCandidate(text=message("Wybierz aplikację z aktualnej listy.",
                                                       "Select an application from the current list."))
            matches = [entries[intent.selection - 1]]
            task_id = choices["task_id"]
        else:
            descriptor = self.capabilities.registry.descriptor("application.list")
            if descriptor is None:
                return AssistantCandidate(text=message("Uruchamianie aplikacji jest niedostępne.", "Application launching is unavailable."))
            listing = normalize_request(
                descriptor, {"task_id": task_id, "query": query}, actor=turn.producer,
                request_id=f"{turn.record_id}:applications", data_class=turn.data_class,
            )
            _, listed = await self.capabilities.invoke(listing, cancellation)
            if isinstance(listed, Failure):
                return AssistantCandidate(text=message(f"Nie mogę sprawdzić aplikacji: {listed.failure().code}.", f"Application discovery failed: {listed.failure().code}."))
            applications = listed.unwrap().output["applications"]
            exact = [app for app in applications if query.casefold() in
                     {app["name"].casefold(), app["desktop_id"].casefold()}]
            matches = exact or applications
        if len(matches) != 1:
            names = "\n".join(f"{i}. {app['name']} ({app['desktop_id']})" for i, app in enumerate(matches, 1))
            return AssistantCandidate(
                text=(message(f"Którą aplikację chcesz uruchomić? Dostępne opcje: {names}.",
                              f"Which application would you like to launch? Available options: {names}.")
                      if matches else message("Nie znalazłem takiej zainstalowanej aplikacji.",
                                              "I could not find that installed application.")),
                metadata={"action_choices": {"kind": "application", "task_id": task_id, "entries": matches}},
            )
        application = matches[0]
        proposal = ActionProposal(
            producer=turn.producer, source_turn_id=turn.record_id, task_id=task_id,
            capability="application.launch", resource_handle=application["handle"],
            data_class=turn.data_class,
        )
        descriptor = self.capabilities.registry.descriptor(proposal.capability)
        if descriptor is None:
            return AssistantCandidate(text=message("Uruchamianie aplikacji jest niedostępne.", "Application launching is unavailable."))
        request = normalize_request(
            descriptor, {"task_id": proposal.task_id, "handle": proposal.resource_handle},
            actor=turn.producer, request_id=f"{turn.record_id}:launch", data_class=turn.data_class,
        )
        decision, executed = await self.capabilities.invoke(request, cancellation)
        terminal = executed.failure() if isinstance(executed, Failure) else executed.unwrap()
        metadata = {
            "action_proposal": proposal.model_dump(mode="json"),
            "action_result": terminal.model_dump(mode="json"),
            "policy_decision_id": decision.record_id if decision else None,
            "remote_tokens": 0,
        }
        if isinstance(executed, Failure):
            return AssistantCandidate(
                text=message(f"Nie potwierdzam uruchomienia aplikacji {application['name']}: {terminal.code}.",
                             f"I could not verify that {application['name']} launched: {terminal.code}."),
                metadata=metadata,
            )
        return AssistantCandidate(
            text=message(f"Uruchomiłem {application['name']}. Potwierdziłem proces aplikacji w systemie.",
                         f"Launched {application['name']}. I verified the application process."),
            metadata=metadata,
        )

    async def _document_intent(self, turn: ConversationTurn, intent: ActionIntent,
                               cancellation: CancellationToken) -> AssistantCandidate:
        task_id = f"action:{turn.record_id}"
        descriptor = self.capabilities.registry.descriptor(intent.outcome)
        if descriptor is None:
            return AssistantCandidate(text=("Obsługa dokumentów jest niedostępna."
                                            if intent.language == "pl" else "Document actions are unavailable."))
        if intent.outcome == "file.search":
            descriptor = self.capabilities.registry.descriptor("file.search")
            request = normalize_request(descriptor, {"query": intent.query, "task_id": task_id},
                                        actor=turn.producer, request_id=f"{turn.record_id}:file-search", data_class=turn.data_class)
            _, fetched = await self.capabilities.invoke(request, cancellation)
            if isinstance(fetched, Failure):
                return AssistantCandidate(text=(f"Wyszukiwanie dokumentów nie powiodło się: {fetched.failure().code}."
                                                if intent.language == "pl" else
                                                f"Document search failed: {fetched.failure().code}."))
            entries = fetched.unwrap().output["documents"]
            text = "\n".join(f"{i}. {item['name']}" for i, item in enumerate(entries, 1))
            return AssistantCandidate(text=text or ("Nie znaleziono dokumentów." if intent.language == "pl" else "No documents found."),
                                      metadata={"action_choices": {"kind": "file", "task_id": task_id, "entries": entries}})
        previous = await previous_action(self.store, turn, self.profile)
        choices = previous.get("action_choices", {})
        entries = choices.get("entries", ())
        if choices.get("kind") != "file" or intent.selection is None or intent.selection > len(entries):
            return AssistantCandidate(text=("Który dokument mam otworzyć? Najpierw wyszukaj dokumenty i wybierz wynik."
                                            if intent.language == "pl" else "Which document should I open? Search for documents and select a result first."))
        selected = entries[intent.selection - 1]
        proposal = ActionProposal(producer=turn.producer, source_turn_id=turn.record_id,
                                  task_id=choices["task_id"], capability="document.open",
                                  resource_handle=selected["handle"], data_class=turn.data_class)
        request = normalize_request(descriptor, {"task_id": choices["task_id"], "handle": selected["handle"]},
                                    actor=turn.producer, request_id=f"{turn.record_id}:document-open", data_class=turn.data_class)
        decision, executed = await self.capabilities.invoke(request, cancellation)
        terminal = executed.failure() if isinstance(executed, Failure) else executed.unwrap()
        text = (("Otwarcie dokumentu potwierdzone." if intent.language == "pl" else "Document opening verified.")
                if not isinstance(executed, Failure) else
                (f"Nie potwierdzam otwarcia: {terminal.code}." if intent.language == "pl" else f"Opening not verified: {terminal.code}."))
        return AssistantCandidate(text=text, metadata={"action_result": terminal.model_dump(mode="json"),
                                                       "action_proposal": proposal.model_dump(mode="json"),
                                                       "remote_tokens": 0,
                                                       "policy_decision_id": decision.record_id if decision else None})
