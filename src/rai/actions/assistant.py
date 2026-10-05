"""Explicit user action intents; retrieved/model text has no execution authority."""
from __future__ import annotations

from collections.abc import Callable
from returns.result import Failure

from rai.assistant.records import AssistantCandidate, ConversationTurn
from rai.assistant.ports import MemoryGraphStore
from .dialogue import previous_action
from rai.kernel.ports import CancellationToken
from rai.kernel.service import CapabilityService
from rai.kernel.transport import normalize_request

from .records import ActionProposal
from .intent import IntentRecognizer, ActionIntent

MAX_DISPLAYED_APPLICATIONS = 15

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

        if intent.outcome in {"file.access", "file.list", "file.search", "document.open"}:
            return await self._document_intent(turn, intent, cancellation)
        if intent.outcome == "application.list":
            return await self._list_applications_intent(turn, intent, cancellation, message)
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

    async def _document_intent(self, turn: ConversationTurn, intent: ActionIntent,  # noqa: PLR0911, PLR0912, PLR0915
                               cancellation: CancellationToken) -> AssistantCandidate:
        task_id = f"action:{turn.record_id}"
        descriptor = self.capabilities.registry.descriptor(intent.outcome)
        if descriptor is None:
            return AssistantCandidate(text=("Obsługa dokumentów jest niedostępna."
                                            if intent.language == "pl" else "Document actions are unavailable."))
        if intent.outcome == "file.access":
            request = normalize_request(descriptor, {"task_id": task_id}, actor=turn.producer,
                                        request_id=f"{turn.record_id}:file-access", data_class=turn.data_class)
            _, accessed = await self.capabilities.invoke(request, cancellation)
            if isinstance(accessed, Failure):
                return AssistantCandidate(text=("Nie mogę odczytać konfiguracji dostępu." if intent.language == "pl"
                                                else "Could not read access configuration."))
            roots = accessed.unwrap().output["allowed_roots"]
            if intent.language == "pl":
                text = ("Dozwolone katalogi wyszukiwania i otwierania dokumentów:\n" + "\n".join(f"- {root}" for root in roots)
                        if roots else "Nie skonfigurowano dozwolonych katalogów dokumentów. Ustaw actions.allowed_file_roots w konfiguracji RAI.")
                text += "\nEdycja plików nie jest obecnie obsługiwana. To zakres akcji dokumentów, nie pełne uprawnienia procesu w systemie."
            else:
                text = ("Allowed document search/open directories:\n" + "\n".join(f"- {root}" for root in roots)
                        if roots else "No document directories are configured. Set actions.allowed_file_roots in the RAI configuration.")
                text += "\nFile editing is not currently supported. This describes document actions, not all OS process permissions."
            return AssistantCandidate(text=text, metadata={"action_result": accessed.unwrap().model_dump(mode="json"), "remote_tokens": 0})
        if intent.outcome == "file.list":
            request = normalize_request(descriptor, {"query": intent.query, "task_id": task_id},
                                        actor=turn.producer, request_id=f"{turn.record_id}:file-list", data_class=turn.data_class)
            _, listed = await self.capabilities.invoke(request, cancellation)
            if isinstance(listed, Failure):
                return AssistantCandidate(text=(f"Nie mogę wylistować katalogu: {listed.failure().code}."
                    if intent.language == "pl" else f"Cannot list directory: {listed.failure().code}."))
            output = listed.unwrap().output
            text = "\n".join(f"- {entry['name']} ({entry['kind']})" for entry in output["entries"])
            if output["truncated"]:
                text += "\n" + ("Lista skrócona." if intent.language == "pl" else "List truncated.")
            return AssistantCandidate(text=text or ("Katalog jest pusty." if intent.language == "pl" else "Directory is empty."),
                metadata={"action_result": listed.unwrap().model_dump(mode="json"), "remote_tokens": 0})
        if intent.outcome == "file.search":
            descriptor = self.capabilities.registry.descriptor("file.search")
            request = normalize_request(descriptor, {"query": intent.query, "task_id": task_id},
                                        actor=turn.producer, request_id=f"{turn.record_id}:file-search", data_class=turn.data_class)
            _, fetched = await self.capabilities.invoke(request, cancellation)
            if isinstance(fetched, Failure):
                if fetched.failure().code == "FILE_ROOTS_NOT_CONFIGURED":
                    return AssistantCandidate(text=(
                        "Nie mogę wyszukać ani otworzyć pliku: nie skonfigurowano dozwolonych katalogów. "
                        "Ustaw actions.allowed_file_roots w konfiguracji RAI, a następnie uruchom rozmowę ponownie."
                        if intent.language == "pl" else
                        "File search/open is unavailable: no directories are configured. "
                        "Set actions.allowed_file_roots in the RAI configuration, then restart the conversation."))
                return AssistantCandidate(text=(f"Wyszukiwanie dokumentów nie powiodło się: {fetched.failure().code}."
                                                if intent.language == "pl" else
                                                f"Document search failed: {fetched.failure().code}."))
            entries = fetched.unwrap().output["documents"]
            text = "\n".join(f"{i}. {item['name']}" for i, item in enumerate(entries, 1))
            return AssistantCandidate(text=text or ("Nie znaleziono dokumentów." if intent.language == "pl" else "No documents found."),
                                      metadata={"action_choices": {"kind": "file", "task_id": task_id, "entries": entries}})
        if intent.query and intent.selection is None:
            searched = await self._document_intent(turn, intent.model_copy(update={"outcome": "file.search"}), cancellation)
            choices = searched.metadata.get("action_choices", {})
            entries = choices.get("entries", ())
            if len(entries) != 1:
                return searched
            selection = 1
        else:
            previous = await previous_action(self.store, turn, self.profile)
            choices = previous.get("action_choices", {})
            entries = choices.get("entries", ())
            selection = intent.selection
        if choices.get("kind") != "file" or selection is None or selection > len(entries):
            return AssistantCandidate(text=("Który dokument mam otworzyć? Najpierw wyszukaj dokumenty i wybierz wynik."
                                            if intent.language == "pl" else "Which document should I open? Search for documents and select a result first."))
        selected = entries[selection - 1]
        proposal = ActionProposal(producer=turn.producer, source_turn_id=turn.record_id,
                                  task_id=choices["task_id"], capability="document.open",
                                  resource_handle=selected["handle"], data_class=turn.data_class)
        arguments = {"task_id": choices["task_id"], "handle": selected["handle"]}
        if intent.application:
            app_descriptor = self.capabilities.registry.descriptor("application.list")
            if app_descriptor is None:
                return AssistantCandidate(text="Application selection unavailable.")
            listing = normalize_request(app_descriptor, {"task_id": choices["task_id"], "query": intent.application},
                actor=turn.producer, request_id=f"{turn.record_id}:document-applications", data_class=turn.data_class)
            _, apps = await self.capabilities.invoke(listing, cancellation)
            if isinstance(apps, Failure):
                return AssistantCandidate(text="Nie mogę sprawdzić aplikacji." if intent.language == "pl" else "Cannot resolve application.")
            matches = apps.unwrap().output["applications"]
            exact = [app for app in matches if intent.application.casefold() in {app["desktop_id"].casefold(), app["name"].casefold()}]
            matches = exact or matches
            if len(matches) != 1:
                names = ", ".join(app["desktop_id"] for app in matches)
                return AssistantCandidate(text=("Doprecyzuj aplikację i ścieżkę: " if intent.language == "pl"
                    else "Specify the application and path: ") + (names or intent.application))
            arguments["application_handle"] = matches[0]["handle"]
        request = normalize_request(descriptor, arguments,
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

    async def _list_applications_intent(
        self, turn: ConversationTurn, intent: ActionIntent,
        cancellation: CancellationToken, message: Callable[[str, str], str],
    ) -> AssistantCandidate:
        descriptor = self.capabilities.registry.descriptor("application.list")
        if descriptor is None:
            return AssistantCandidate(
                text=message("Lista aplikacji jest niedostępna.", "Application listing is unavailable.")
            )
        task_id = f"action:{turn.record_id}"
        listing = normalize_request(
            descriptor, {"task_id": task_id, "query": intent.query.strip()}, actor=turn.producer,
            request_id=f"{turn.record_id}:applications", data_class=turn.data_class,
        )
        _, listed = await self.capabilities.invoke(listing, cancellation)
        if isinstance(listed, Failure):
            return AssistantCandidate(
                text=message(f"Nie mogę sprawdzić aplikacji: {listed.failure().code}.",
                             f"Application discovery failed: {listed.failure().code}.")
            )
        applications = listed.unwrap().output["applications"]
        names = [str(app["name"]) for app in applications]
        if not names:
            return AssistantCandidate(
                text=message("Nie znalazłem zainstalowanych aplikacji w systemie.",
                             "I could not find installed applications in the system.")
            )
        sample = names[:MAX_DISPLAYED_APPLICATIONS]
        more = (
            f" (oraz {len(names) - MAX_DISPLAYED_APPLICATIONS} innych)"
            if len(names) > MAX_DISPLAYED_APPLICATIONS
            else ""
        )
        if intent.language == "pl":
            reply = (
                "W Twoim systemie mogę uruchomić zainstalowane aplikacje, na przykład:\n"
                + "\n".join(f"- {n}" for n in sample)
                + f"{more}\n\nAby uruchomić wybraną aplikację, napisz po prostu np. 'Uruchom {sample[0]}'."
            )
        else:
            reply = (
                "I can launch installed applications on your system, for example:\n"
                + "\n".join(f"- {n}" for n in sample)
                + f"{more}\n\nTo launch an application, simply say e.g. 'Launch {sample[0]}'."
            )
        metadata = {
            "action_choices": {"kind": "application", "task_id": task_id, "entries": applications},
            "action_result": {"status": "SUCCEEDED", "output": {"applications": applications}},
        }
        return AssistantCandidate(
            text=reply,
            metadata=metadata,
        )
