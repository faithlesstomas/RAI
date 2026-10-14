from datetime import timedelta
from returns.result import Failure, Result

from rai.assistant.ports import MemoryGraphStore
from rai.assistant.records import AssistantCandidate, ConversationTurn
from rai.kernel.ports import CancellationToken
from rai.kernel.service import CapabilityService
from rai.kernel.records import ActionFailure, ActionResult, PolicyDecision, _new_id, _utc_now
from rai.kernel.transport import normalize_request
from .dialogue import previous_action
from .intent import ActionIntent
from .records import ActionProposal, ActiveChoiceFrame, ChoiceEntry, ChoiceState


async def route_action(  # noqa: PLR0913, PLR0912, PLR0915
    service: CapabilityService, store: MemoryGraphStore | None, profile: str,
    turn: ConversationTurn, intent: ActionIntent, token: CancellationToken,
) -> AssistantCandidate:
    """Preserve data classification and source identity across each action step."""
    name = intent.outcome
    task_id = f"action:{turn.record_id}"
    arguments = {"task_id": task_id}
    polish = intent.language == "pl"
    choices = {}
    proposal = None

    async def invoke(capability: str, params: dict) -> tuple[PolicyDecision | None, Result[ActionResult, ActionFailure]] | None:
        descriptor = service.registry.descriptor(capability)
        if descriptor is None:
            return None
        request = normalize_request(descriptor, params, actor=turn.producer,
                                    request_id=f"{turn.record_id}:{capability}", data_class=turn.data_class)
        return await service.invoke(request, token)

    if name in {"browser.open_result", "browser.read_page"}:
        previous = await previous_action(store, turn, profile)
        frame_dict = previous.get("active_choice_frame")
        if isinstance(frame_dict, dict) and frame_dict.get("state") in {"CANCELLED", ChoiceState.CANCELLED.value}:
            return AssistantCandidate(
                text="Poprzedni wybór został anulowany." if polish else "The previous choice was cancelled.",
                metadata={"action_failure": "FRAME_CANCELLED"},
            )
        choices = previous.get("action_choices", {})
        entries = choices.get("entries", ())
        if name == "browser.read_page" and intent.selection is None and previous.get("selected_resource_handle"):
            arguments = {"task_id": choices.get("task_id", task_id), "handle": previous["selected_resource_handle"]}
        elif choices.get("kind") != "url" or intent.selection is None or intent.selection > len(entries):
            return AssistantCandidate(text=("Który wynik wybrać? Wyszukaj strony i podaj numer wyniku."
                                            if polish else "Which result? Search the web and select a result number."))
        else:
            arguments = {"task_id": choices["task_id"], "handle": entries[intent.selection - 1]["handle"]}
    elif name == "system.volume.set":
        initial = await invoke("system.volume.get", arguments)
        if initial is None or isinstance(initial[1], Failure):
            return AssistantCandidate(text=("Nie udało się odczytać urządzenia audio."
                                            if polish else "Could not read the audio device."))
        arguments.update(handle=initial[1].unwrap().output["handle"], percent=intent.percent)
    elif name in {"browser.search", "process.inspect"}:
        arguments["query"] = intent.query
    elif name == "activity.query":
        if intent.lookback_minutes is not None:
            arguments["lookback_minutes"] = intent.lookback_minutes
        if intent.query:
            arguments["query"] = intent.query
    if "handle" in arguments:
        proposal = ActionProposal(producer=turn.producer, source_turn_id=turn.record_id,
                                  task_id=arguments["task_id"], capability=name,
                                  resource_handle=arguments["handle"], data_class=turn.data_class)
    invoked = await invoke(name, arguments)
    if invoked is None:
        return AssistantCandidate(text="Funkcja jest niedostępna." if polish else "Capability unavailable.")
    decision, executed = invoked
    terminal = executed.failure() if isinstance(executed, Failure) else executed.unwrap()
    metadata = {"action_result": terminal.model_dump(mode="json"), "remote_tokens": 0,
                "policy_decision_id": decision.record_id if decision else None}
    if proposal:
        metadata["action_proposal"] = proposal.model_dump(mode="json")
    if isinstance(executed, Failure):
        text = f"Nie potwierdzam wykonania: {terminal.code}." if polish else f"Execution not verified: {terminal.code}."
    else:
        output = executed.unwrap().output
        if name in {"browser.search", "process.inspect"}:
            entries = output["results" if name == "browser.search" else "processes"]
            choices = {"kind": "url" if name == "browser.search" else "process", "task_id": task_id, "entries": entries}
            if name == "browser.search":
                lines = []
                for i, entry in enumerate(entries, 1):
                    snippet = f" - {entry['snippet']}" if entry.get("snippet") else ""
                    lines.append(f"{i}. {entry['name']}{snippet}")
                text = "\n".join(lines)
                metadata["requires_synthesis"] = bool(entries)
                metadata["untrusted_content"] = True
            else:
                text = "\n".join(f"{i}. {entry['name']}" + (f" (PID {entry['pid']})" if 'pid' in entry else '')
                                 for i, entry in enumerate(entries, 1))
            text = text or ("Brak wyników." if polish else "No results.")
        elif name == "browser.read_page":
            text = output["title"] + "\n\n" + output["text"]
            metadata["untrusted_content"] = True
            metadata["requires_synthesis"] = True
            metadata["selected_resource_handle"] = arguments.get("handle")
        elif name == "browser.open_result":
            text = "Wykonanie potwierdzone." if polish else "Execution verified."
            metadata["selected_resource_handle"] = arguments.get("handle")
            if choices:
                metadata["action_choices"] = choices
        elif name == "system.volume.get":
            text = ("Głośność: " if polish else "Volume: ") + ", ".join(f"{v}%" for v in output["volumes"])
        elif name == "activity.query":
            episodes = output.get("episodes", ())
            if not episodes:
                reason = output.get("empty_reason")
                if reason == "classification_filtered":
                    text = ("Historia istnieje, ale jej klasyfikacja prywatności przekracza zakres tej rozmowy. "
                            "Dla danych PRIVATE uruchom czat z --data-class PRIVATE." if polish else
                            "History exists, but its privacy classification exceeds this conversation scope. "
                            "For PRIVATE data, start chat with --data-class PRIVATE.")
                elif reason == "no_match":
                    text = "Brak aktywności pasującej do filtra." if polish else "No activity matches the filter."
                else:
                    text = ("Brak zapisanych epizodów w wybranym okresie. Sprawdź /status oraz kolektory serwera."
                            if polish else "No stored episodes in the selected period. Check /status and the daemon collectors.")
            else:
                lines = []
                for ep in episodes:
                    apps = ", ".join(ep.get("applications", ())) or ("system" if polish else "system")
                    projects = ", ".join(ep.get("projects", ()))
                    started = str(ep.get("started_at", ""))
                    ended = str(ep.get("ended_at", ""))
                    time_range = (
                        f"{started[:16]} - {ended[11:16]}"
                        if len(started) >= 16 and len(ended) >= 16  # noqa: PLR2004
                        else (started or ended or "")
                    )
                    summary = f"[{time_range}] {apps}" if time_range else f"[{apps}]"
                    if projects:
                        summary += f" ({projects})"
                    if ep.get("activity_types"):
                        summary += f": {', '.join(ep['activity_types'])}"
                    lines.append(f"- {summary}")
                prefix = "Ostatnia aktywność w systemie:\n" if polish else "Recent system activity:\n"
                text = prefix + "\n".join(lines)
                metadata["requires_synthesis"] = True
                metadata["untrusted_content"] = False
        else:
            text = "Wykonanie potwierdzone." if polish else "Execution verified."
        if name in {"browser.search", "process.inspect"} and choices:
            metadata["action_choices"] = choices
            if name in {"browser.search", "process.inspect"} and entries:
                choice_entries = tuple(
                    ChoiceEntry(
                        index=i,
                        handle=entry["handle"],
                        label=entry["name"],
                        metadata={
                            **({"url": entry["url"]} if "url" in entry else {}),
                            **({"snippet": entry["snippet"]} if "snippet" in entry else {}),
                        },
                    )
                    for i, entry in enumerate(entries, 1)
                    if "handle" in entry
                )
                if choice_entries:
                    now = _utc_now()
                    frame = ActiveChoiceFrame(
                        record_id=_new_id(),
                        timestamp=now,
                        producer=turn.producer,
                        session_id=turn.session_id,
                        profile_scope=profile,
                        task_id=task_id,
                        source_turn_id=turn.record_id,
                        intended_capability="browser.open_result" if name == "browser.search" else "process.inspect",
                        entries=choice_entries,
                        state=ChoiceState.ACTIVE,
                        expires_at=now + timedelta(minutes=15),
                        data_class=turn.data_class,
                    )
                    metadata["active_choice_frame"] = frame.model_dump(mode="json")
    return AssistantCandidate(text=text, metadata=metadata)
