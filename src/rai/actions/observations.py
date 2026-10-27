"""Pure-Python observation processing capabilities (safe grep/awk over stored artifacts)."""

from __future__ import annotations

import asyncio
import json
import re
from typing import Any, Iterator, Sequence

from returns.result import Failure, Result, Success

from rai.assistant.observation_budget import (
    persist_observation_artifact,
    render_observation_payload,
)
from rai.assistant.ports import ObservationStore
from rai.assistant.records import ObservationArtifact
from rai.kernel.capabilities import (
    CapabilityDescriptor,
    CapabilityRegistry,
    RegisteredCapability,
)
from rai.kernel.ports import CancellationToken
from rai.kernel.records import (
    ActionFailure,
    ActionResult,
    CapabilityRequest,
    ProducerIdentity,
    RiskClass,
    _new_id,
    _utc_now,
)

PRODUCER = ProducerIdentity(
    producer_id="rai.actions.observations", kind="capability", version="1.0.0"
)
MAX_REGEX_PATTERN_LEN = 200
REGEX_TIMEOUT_SECONDS = 2.0
MAX_SCAN_LIMIT = 10000
MAX_VISITED_LIMIT = 10000
YIELD_INTERVAL = 100
DEFAULT_PAGE_LIMIT = 100
MAX_PAGE_LIMIT = 1000
MAX_GET_CHAR_LIMIT = 64000
MAX_ECHOED_ARG_CHAR_LIMIT = 1000
FALLBACK_FIELD_CHAR_LIMIT = 200


def _fail(request: CapabilityRequest, code: str, message: str) -> ActionFailure:
    return ActionFailure(
        record_id=f"failure:{request.record_id}:{code}",
        producer=PRODUCER,
        correlation_id=request.correlation_id,
        request_id=request.record_id,
        capability=request.capability,
        code=code,
        message=message,
    )


def _ok(
    request: CapabilityRequest, output: dict[str, Any], verification: dict[str, Any]
) -> ActionResult:
    return ActionResult(
        record_id=f"result:{request.record_id}",
        producer=PRODUCER,
        correlation_id=request.correlation_id,
        request_id=request.record_id,
        capability=request.capability,
        output=output,
        verification=verification,
    )


def _extract_primary_list(content: dict[str, Any]) -> tuple[str | None, Sequence[Any]]:
    for key in ("applications", "results", "documents", "entries", "processes", "items", "episodes"):
        if key in content and isinstance(content[key], (list, tuple)):
            return key, content[key]
    for k, v in content.items():
        if isinstance(v, (list, tuple)):
            return str(k), v
    return None, ()


LINE_BREAK_PATTERN = re.compile(r"\r\n|[\n\r\x0b\x0c\x1c\x1d\x1e\x85\u2028\u2029]")


def _iter_text_lines(text: str) -> Iterator[str]:
    """Iterate over lines lazily with semantics matching str.splitlines()."""
    if not text:
        return
    start = 0
    for match in LINE_BREAK_PATTERN.finditer(text):
        yield text[start : match.start()]
        start = match.end()
    if start < len(text):
        yield text[start:]


def _bound_string(
    s: str, max_json_chars: int = MAX_GET_CHAR_LIMIT
) -> tuple[str, bool, str | None]:
    """Ensure a string's JSON-serialized representation strictly does not exceed max_json_chars."""
    if len(json.dumps(s)) <= max_json_chars:
        return s, True, None

    low = 0
    high = len(s)
    best_p = 0
    while low <= high:
        mid = (low + high) // 2
        omitted = len(s) - mid
        candidate = s[:mid] + f"... [truncated {omitted} chars]"
        if len(json.dumps(candidate)) <= max_json_chars:
            best_p = mid
            low = mid + 1
        else:
            high = mid - 1

    omitted = len(s) - best_p
    truncated = s[:best_p] + f"... [truncated {omitted} chars]"
    reason = (
        f"Value exceeded maximum character limit ({max_json_chars} of {len(s)} chars returned). "
        "Use observation.slice or observation.search."
    )
    return truncated, False, reason


def _bound_output_collection(
    items: Sequence[Any],
    max_items: int = MAX_PAGE_LIMIT,
    max_chars: int = MAX_GET_CHAR_LIMIT,
) -> tuple[list[Any], bool, str | None]:
    """Bound an output list/collection by both item count and serialized JSON size."""
    bounded: list[Any] = []
    hit_item_limit = False
    hit_char_limit = False

    cur_len = 2  # Brackets "[]"
    for count, item in enumerate(items):
        if count >= max_items:
            hit_item_limit = True
            break
        try:
            item_str = json.dumps(item)
        except (TypeError, ValueError):
            item_str = json.dumps(str(item))

        additional_len = len(item_str) + (2 if count > 0 else 0)
        if cur_len + additional_len > max_chars:
            if not bounded:
                budget = max(50, max_chars - 2)
                if isinstance(item, str):
                    truncated_str, _, _ = _bound_string(item, budget)
                    bounded.append(truncated_str)
                elif isinstance(item, dict):
                    preview = render_observation_payload(item, budget_chars=budget - 20)
                    cand_dict = {"_preview": preview}
                    if len(json.dumps(cand_dict)) > budget:
                        bounded_prev, _, _ = _bound_string(preview, budget - 30)
                        cand_dict = {"_preview": bounded_prev}
                    bounded.append(cand_dict)
                else:
                    truncated_str, _, _ = _bound_string(str(item), budget)
                    bounded.append(truncated_str)
            hit_char_limit = True
            break
        bounded.append(item)
        cur_len += additional_len

    complete = not hit_item_limit and not hit_char_limit and len(bounded) == len(items)
    reason: str | None = None
    if hit_char_limit:
        reason = (
            f"Output payload exceeded maximum size limit ({max_chars} chars, "
            f"{len(bounded)} of {len(items)} items returned)"
        )
    elif hit_item_limit:
        reason = (
            f"List value exceeded maximum item limit ({max_items} of {len(items)} items returned)"
        )

    return bounded, complete, reason


def _bound_echoed_value(val: Any, max_json_chars: int = 1000) -> tuple[Any, bool]:  # noqa: ANN401
    """Ensure caller-provided filter value echoed in output does not exceed size limit."""
    if val is None or isinstance(val, (int, float, bool)):
        return val, False
    try:
        val_json = json.dumps(val)
    except (TypeError, ValueError):
        val_json = json.dumps(str(val))
        val = str(val)

    if len(val_json) <= max_json_chars:
        return val, False

    if isinstance(val, str):
        truncated_str, _, _ = _bound_string(val, max_json_chars)
        return truncated_str, True

    if isinstance(val, dict):
        preview = render_observation_payload(val, budget_chars=max(50, max_json_chars - 30))
        cand: dict[str, Any] = {"_preview": preview}
        if len(json.dumps(cand)) > max_json_chars:
            bounded_prev, _, _ = _bound_string(preview, max(50, max_json_chars - 30))
            cand = {"_preview": bounded_prev}
        return cand, True

    if isinstance(val, (list, tuple)):
        preview = render_observation_payload(list(val), budget_chars=max(50, max_json_chars - 30))
        cand_list: dict[str, Any] = {"_preview": preview}
        if len(json.dumps(cand_list)) > max_json_chars:
            bounded_prev, _, _ = _bound_string(preview, max(50, max_json_chars - 30))
            cand_list = {"_preview": bounded_prev}
        return cand_list, True

    truncated_str, _, _ = _bound_string(str(val), max_json_chars)
    return truncated_str, True


def _trim_output_value(output: dict[str, Any], max_chars: int) -> None:
    val = output.get("value")
    if isinstance(val, str):
        overhead = len(json.dumps(output)) - len(json.dumps(val))
        budget = max(50, max_chars - overhead - 20)
        output["value"], _, _ = _bound_string(val, budget)
    elif isinstance(val, (list, tuple)):
        val_list = list(val)
        while val_list and len(json.dumps(output)) > max_chars:
            val_list.pop()
            output["value"] = val_list
    elif isinstance(val, dict):
        preview = render_observation_payload(val, budget_chars=max(50, max_chars - 600))
        cand: dict[str, Any] = {"_preview": preview}
        if len(json.dumps(cand)) > max(50, max_chars - 600):
            preview, _, _ = _bound_string(preview, max(50, max_chars - 630))
            cand = {"_preview": preview}
        output["value"] = cand


def _enforce_final_payload_limit(
    output: dict[str, Any],
    list_key: str | None = None,
    max_chars: int = MAX_GET_CHAR_LIMIT,
) -> dict[str, Any]:
    """Strictly guarantee that len(json.dumps(output)) <= max_chars."""
    if len(json.dumps(output)) <= max_chars:
        return output

    output["complete"] = False
    output["reason"] = f"Output payload exceeded maximum size limit ({max_chars} chars)"

    if list_key and isinstance(output.get(list_key), list):
        items = output[list_key]
        while items and len(json.dumps(output)) > max_chars:
            items.pop()
        if not items and len(json.dumps(output)) > max_chars:
            for k, v in list(output.items()):
                if isinstance(v, str) and len(json.dumps(v)) > FALLBACK_FIELD_CHAR_LIMIT:
                    output[k], _, _ = _bound_string(v, FALLBACK_FIELD_CHAR_LIMIT)
    elif "counts" in output and isinstance(output["counts"], dict):
        counts = output["counts"]
        while counts and len(json.dumps(output)) > max_chars:
            counts.popitem()
    elif "value" in output:
        _trim_output_value(output, max_chars)

    if len(json.dumps(output)) > max_chars:
        output_str = json.dumps(output)
        preview, _, _ = _bound_string(output_str, max(50, max_chars - 50))
        return {
            "ref": output.get("ref", ""),
            "complete": False,
            "reason": f"Output payload exceeded maximum size limit ({max_chars} chars)",
            "_preview": preview,
        }

    return output


async def _get_artifact(
    store: ObservationStore | None,
    ref: str,
    session_id: str | None = None,
) -> ObservationArtifact | None:
    if store is None or not session_id:
        return None
    res = await store.get(session_id, ref)
    if isinstance(res, Success):
        return res.unwrap()
    return None


async def _search_text(
    text: str,
    compiled: re.Pattern[str],
    cancellation: CancellationToken,
) -> Result[tuple[list[dict[str, Any]], bool], None]:
    matched: list[dict[str, Any]] = []
    hit_scan_limit = False
    lines_iter = _iter_text_lines(text)
    curr_line = next(lines_iter, None)
    if curr_line is None:
        return Success((matched, hit_scan_limit))

    prev_line: str | None = None
    idx = 1
    while curr_line is not None:
        if idx % YIELD_INTERVAL == 0:
            await asyncio.sleep(0)
            if cancellation.cancelled:
                return Failure(None)
        if idx > MAX_VISITED_LIMIT:
            hit_scan_limit = True
            break

        next_line = next(lines_iter, None)
        if compiled.search(curr_line):
            ctx = [line for line in (prev_line, curr_line, next_line) if line is not None]
            context_snippet = "\n".join(ctx)
            matched.append({
                "line_number": idx,
                "line": curr_line.strip(),
                "context": context_snippet,
            })
            if len(matched) >= MAX_SCAN_LIMIT:
                hit_scan_limit = True
                break

        prev_line = curr_line
        curr_line = next_line
        idx += 1

    return Success((matched, hit_scan_limit))


async def _search_records(
    records: Sequence[Any],
    compiled: re.Pattern[str],
    fields: list[str] | None,
    cancellation: CancellationToken,
) -> Result[tuple[list[Any], bool], None]:
    matched: list[Any] = []
    hit_scan_limit = False
    for count, item in enumerate(records, 1):
        if count % YIELD_INTERVAL == 0:
            await asyncio.sleep(0)
            if cancellation.cancelled:
                return Failure(None)
        if count > MAX_VISITED_LIMIT:
            hit_scan_limit = True
            break
        if isinstance(item, dict):
            search_keys = fields if fields else list(item.keys())
            item_matched = False
            for k in search_keys:
                val = item.get(k)
                if val is not None and compiled.search(str(val)):
                    item_matched = True
                    break
            if item_matched:
                matched.append(item)
        elif compiled.search(str(item)):
            matched.append(item)
        if len(matched) >= MAX_SCAN_LIMIT:
            hit_scan_limit = True
            break
    return Success((matched, hit_scan_limit))


class ObservationSearch:
    """Grep-like search over record collections or text lines in stored artifacts."""

    name = "observation.search"

    def __init__(self, store: ObservationStore) -> None:
        self.store = store

    async def invoke(  # noqa: PLR0911, PLR0912
        self, request: CapabilityRequest, cancellation: CancellationToken
    ) -> Result[ActionResult, ActionFailure]:
        if cancellation.cancelled:
            return Failure(_fail(request, "CANCELLED", "Search cancelled"))

        session_id = str(request.arguments.get("session_id", "")).strip()
        ref = str(request.arguments.get("ref", "")).strip()
        pattern = str(request.arguments.get("pattern", "")).strip()
        fields = request.arguments.get("fields")
        case_sensitive = bool(request.arguments.get("case", False))
        offset = max(0, int(request.arguments.get("offset", 0)))
        raw_limit = request.arguments.get("limit")
        limit = min(
            MAX_PAGE_LIMIT,
            max(1, int(raw_limit if raw_limit is not None else DEFAULT_PAGE_LIMIT)),
        )

        if not session_id:
            return Failure(_fail(request, "UNAUTHORIZED", "Missing session_id"))
        if not ref or not pattern:
            return Failure(_fail(request, "INVALID_ARGUMENT", "Missing ref or pattern"))
        if len(pattern) > MAX_REGEX_PATTERN_LEN:
            return Failure(_fail(request, "PATTERN_TOO_LONG", "Pattern exceeds limit"))

        artifact = await _get_artifact(self.store, ref, session_id=session_id)
        if artifact is None:
            return Failure(_fail(request, "ARTIFACT_NOT_FOUND", f"Artifact {ref} not found"))

        flags = 0 if case_sensitive else re.IGNORECASE
        compiled = re.compile(re.escape(pattern), flags)

        content = artifact.content
        if "text" in content and isinstance(content["text"], str):
            search_res = await _search_text(content["text"], compiled, cancellation)
        else:
            _, records = _extract_primary_list(content)
            search_res = await _search_records(records, compiled, fields, cancellation)

        if isinstance(search_res, Failure):
            return Failure(_fail(request, "CANCELLED", "Search cancelled"))
        matched, hit_scan_limit = search_res.unwrap()

        total_matches = len(matched)
        paginated = matched[offset : offset + limit]
        bounded_matches, payload_complete, payload_reason = _bound_output_collection(
            paginated, max_items=limit, max_chars=MAX_GET_CHAR_LIMIT
        )
        complete = (
            not hit_scan_limit
            and offset == 0
            and total_matches <= len(bounded_matches)
            and payload_complete
        )
        if hit_scan_limit:
            reason = f"Hit maximum search scan limit ({MAX_VISITED_LIMIT} items inspected or {MAX_SCAN_LIMIT} matches)"
        elif not payload_complete and payload_reason:
            reason = payload_reason
        elif not complete:
            reason = f"Results paginated ({len(bounded_matches)} of {total_matches} matches returned)"
        else:
            reason = None

        output: dict[str, Any] = {
            "ref": ref,
            "pattern": pattern,
            "total_matches": total_matches,
            "offset": offset,
            "limit": limit,
            "complete": complete,
            "reason": reason,
            "matches": bounded_matches,
        }
        output = _enforce_final_payload_limit(
            output, list_key="matches", max_chars=MAX_GET_CHAR_LIMIT
        )

        derived_id = await persist_observation_artifact(
            store=self.store,
            session_id=artifact.session_id,
            turn_id=artifact.turn_id,
            capability=self.name,
            request_id=request.record_id,
            data_class=artifact.data_class,
            output=output,
            derived_from=(ref,),
        )
        if derived_id:
            output["artifact_id"] = derived_id

        return Success(_ok(request, output, {"source_artifact_id": ref}))


def _evaluate_filter_condition(item_val: Any, op: str, val: Any) -> bool:  # noqa: ANN401, PLR0911, PLR0912
    if item_val is None:
        return False
    if op == "equals":
        return str(item_val).casefold() == str(val).casefold()
    if op == "contains":
        if isinstance(item_val, (list, tuple)):
            return any(str(val).casefold() in str(x).casefold() for x in item_val)
        return str(val).casefold() in str(item_val).casefold()
    if op == "startswith":
        return str(item_val).casefold().startswith(str(val).casefold())
    if op == "endswith":
        return str(item_val).casefold().endswith(str(val).casefold())
    if op == "in":
        if isinstance(val, (list, tuple)):
            return any(str(item_val).casefold() == str(v).casefold() for v in val)
        return str(item_val).casefold() in str(val).casefold()
    try:
        num_item = float(item_val)
        num_val = float(val)
        if op == "gt":
            return num_item > num_val
        if op == "gte":
            return num_item >= num_val
        if op == "lt":
            return num_item < num_val
        if op == "lte":
            return num_item <= num_val
    except (ValueError, TypeError):
        return False
    return False


async def _filter_records(
    records: Sequence[Any],
    field: str,
    op: str,
    val: Any,  # noqa: ANN401
    cancellation: CancellationToken,
) -> Result[tuple[list[Any], bool], None]:
    filtered: list[Any] = []
    hit_scan_limit = False
    for count, rec in enumerate(records, 1):
        if count % YIELD_INTERVAL == 0:
            await asyncio.sleep(0)
            if cancellation.cancelled:
                return Failure(None)
        if count > MAX_VISITED_LIMIT:
            hit_scan_limit = True
            break
        if isinstance(rec, dict) and _evaluate_filter_condition(rec.get(field), op, val):
            filtered.append(rec)
            if len(filtered) >= MAX_SCAN_LIMIT:
                hit_scan_limit = True
                break
    return Success((filtered, hit_scan_limit))


class ObservationFilter:
    """Filter records by structured field comparisons (contains, equals, numeric)."""

    name = "observation.filter"

    def __init__(self, store: ObservationStore) -> None:
        self.store = store

    async def invoke(
        self, request: CapabilityRequest, cancellation: CancellationToken
    ) -> Result[ActionResult, ActionFailure]:
        if cancellation.cancelled:
            return Failure(_fail(request, "CANCELLED", "Filter cancelled"))

        session_id = str(request.arguments.get("session_id", "")).strip()
        ref = str(request.arguments.get("ref", "")).strip()
        field = str(request.arguments.get("field", "")).strip()
        op = str(request.arguments.get("op", "equals")).strip().lower()
        val = request.arguments.get("value")
        offset = max(0, int(request.arguments.get("offset", 0)))
        raw_limit = request.arguments.get("limit")
        limit = min(
            MAX_PAGE_LIMIT,
            max(1, int(raw_limit if raw_limit is not None else DEFAULT_PAGE_LIMIT)),
        )

        if not session_id:
            return Failure(_fail(request, "UNAUTHORIZED", "Missing session_id"))
        if not ref or not field:
            return Failure(_fail(request, "INVALID_ARGUMENT", "Missing ref or field"))

        artifact = await _get_artifact(self.store, ref, session_id=session_id)
        if artifact is None:
            return Failure(_fail(request, "ARTIFACT_NOT_FOUND", f"Artifact {ref} not found"))

        _, records = _extract_primary_list(artifact.content)
        filter_res = await _filter_records(records, field, op, val, cancellation)
        if isinstance(filter_res, Failure):
            return Failure(_fail(request, "CANCELLED", "Filter cancelled"))
        filtered, hit_scan_limit = filter_res.unwrap()

        bounded_val, val_truncated = _bound_echoed_value(
            val, max_json_chars=MAX_ECHOED_ARG_CHAR_LIMIT
        )

        total_matches = len(filtered)
        paginated = filtered[offset : offset + limit]
        bounded_results, payload_complete, payload_reason = _bound_output_collection(
            paginated, max_items=limit, max_chars=MAX_GET_CHAR_LIMIT
        )
        complete = (
            not hit_scan_limit
            and offset == 0
            and total_matches <= len(bounded_results)
            and payload_complete
            and not val_truncated
        )
        if hit_scan_limit:
            reason = f"Hit maximum filter scan limit ({MAX_VISITED_LIMIT} items inspected or {MAX_SCAN_LIMIT} matches)"
        elif not payload_complete and payload_reason:
            reason = payload_reason
        elif val_truncated:
            reason = "Echoed filter value exceeded maximum argument limit (truncated)"
        elif not complete:
            reason = f"Results paginated ({len(bounded_results)} of {total_matches} matches returned)"
        else:
            reason = None

        output: dict[str, Any] = {
            "ref": ref,
            "field": field,
            "op": op,
            "value": bounded_val,
            "total_matches": total_matches,
            "offset": offset,
            "limit": limit,
            "complete": complete,
            "reason": reason,
            "results": bounded_results,
        }
        output = _enforce_final_payload_limit(
            output, list_key="results", max_chars=MAX_GET_CHAR_LIMIT
        )

        derived_id = await persist_observation_artifact(
            store=self.store,
            session_id=artifact.session_id,
            turn_id=artifact.turn_id,
            capability=self.name,
            request_id=request.record_id,
            data_class=artifact.data_class,
            output=output,
            derived_from=(ref,),
        )
        if derived_id:
            output["artifact_id"] = derived_id

        return Success(_ok(request, output, {"source_artifact_id": ref}))


async def _slice_text(
    text: str, offset: int, limit: int, cancellation: CancellationToken
) -> tuple[list[str] | None, int, bool]:
    sliced_raw: list[str] = []
    inspected = 0
    hit_scan_limit = False
    target_end = offset + limit

    for line in _iter_text_lines(text):
        inspected += 1
        if inspected % YIELD_INTERVAL == 0:
            await asyncio.sleep(0)
            if cancellation.cancelled:
                return None, 0, False
        if inspected > MAX_VISITED_LIMIT:
            hit_scan_limit = True
            break
        idx = inspected - 1
        if offset <= idx < target_end:
            sliced_raw.append(line)

    total_count = inspected if not hit_scan_limit else MAX_VISITED_LIMIT
    return sliced_raw, total_count, hit_scan_limit


def _slice_records(
    all_items: Sequence[Any], offset: int, limit: int
) -> tuple[list[Any], int, bool]:
    total_count = len(all_items)
    hit_scan_limit = False
    if offset > MAX_VISITED_LIMIT:
        hit_scan_limit = True
        sliced_raw: list[Any] = []
    else:
        end_idx = min(offset + limit, MAX_VISITED_LIMIT)
        sliced_raw = list(all_items[offset:end_idx])
        if total_count > MAX_VISITED_LIMIT and offset + limit > MAX_VISITED_LIMIT:
            hit_scan_limit = True
    return sliced_raw, total_count, hit_scan_limit


class ObservationSlice:
    """Return a pagination window of records or lines from an artifact."""

    name = "observation.slice"

    def __init__(self, store: ObservationStore) -> None:
        self.store = store

    async def invoke(
        self, request: CapabilityRequest, cancellation: CancellationToken
    ) -> Result[ActionResult, ActionFailure]:
        if cancellation.cancelled:
            return Failure(_fail(request, "CANCELLED", "Slice cancelled"))

        session_id = str(request.arguments.get("session_id", "")).strip()
        ref = str(request.arguments.get("ref", "")).strip()
        offset = max(0, int(request.arguments.get("offset", 0)))
        limit = min(1000, max(1, int(request.arguments.get("limit", 20))))

        if not session_id:
            return Failure(_fail(request, "UNAUTHORIZED", "Missing session_id"))
        if not ref:
            return Failure(_fail(request, "INVALID_ARGUMENT", "Missing ref"))

        artifact = await _get_artifact(self.store, ref, session_id=session_id)
        if artifact is None:
            return Failure(_fail(request, "ARTIFACT_NOT_FOUND", f"Artifact {ref} not found"))

        content = artifact.content
        if "text" in content and isinstance(content["text"], str):
            sliced_raw, total_count, hit_scan_limit = await _slice_text(
                content["text"], offset, limit, cancellation
            )
            if sliced_raw is None:
                return Failure(_fail(request, "CANCELLED", "Slice cancelled"))
        else:
            _, all_items = _extract_primary_list(content)
            sliced_raw, total_count, hit_scan_limit = _slice_records(
                all_items, offset, limit
            )

        bounded_items, payload_complete, payload_reason = _bound_output_collection(
            sliced_raw, max_items=limit, max_chars=MAX_GET_CHAR_LIMIT
        )
        complete = (
            not hit_scan_limit
            and offset == 0
            and total_count <= len(bounded_items)
            and payload_complete
        )
        if hit_scan_limit:
            reason = f"Hit maximum slice scan limit ({MAX_VISITED_LIMIT} items inspected)"
        elif not payload_complete and payload_reason:
            reason = payload_reason
        elif not complete:
            reason = f"Results paginated ({len(bounded_items)} of {total_count} items returned)"
        else:
            reason = None

        output: dict[str, Any] = {
            "ref": ref,
            "offset": offset,
            "limit": limit,
            "total_count": total_count,
            "complete": complete,
            "reason": reason,
            "items": bounded_items,
        }
        output = _enforce_final_payload_limit(
            output, list_key="items", max_chars=MAX_GET_CHAR_LIMIT
        )

        derived_id = await persist_observation_artifact(
            store=self.store,
            session_id=artifact.session_id,
            turn_id=artifact.turn_id,
            capability=self.name,
            request_id=request.record_id,
            data_class=artifact.data_class,
            output=output,
            derived_from=(ref,),
        )
        if derived_id:
            output["artifact_id"] = derived_id

        return Success(_ok(request, output, {"source_artifact_id": ref}))


async def _count_grouped_records(
    records: Sequence[Any], field: str, cancellation: CancellationToken
) -> tuple[dict[str, int], int, bool] | None:
    counts: dict[str, int] = {}
    visited = 0
    hit_scan_limit = False

    for rec in records:
        visited += 1
        if visited % YIELD_INTERVAL == 0:
            await asyncio.sleep(0)
            if cancellation.cancelled:
                return None
        if visited > MAX_VISITED_LIMIT:
            hit_scan_limit = True
            break
        if isinstance(rec, dict):
            v = rec.get(field)
            if isinstance(v, (list, tuple)):
                for sub_v in v:
                    visited += 1
                    if visited % YIELD_INTERVAL == 0:
                        await asyncio.sleep(0)
                        if cancellation.cancelled:
                            return None
                    if visited > MAX_VISITED_LIMIT:
                        hit_scan_limit = True
                        break
                    s = str(sub_v)
                    counts[s] = counts.get(s, 0) + 1
                if hit_scan_limit:
                    break
            else:
                s = str(v) if v is not None else "null"
                counts[s] = counts.get(s, 0) + 1

    total_scanned = min(visited, MAX_VISITED_LIMIT) if hit_scan_limit else visited
    return counts, total_scanned, hit_scan_limit


def _bound_counts_dictionary(
    counts: dict[str, int],
    max_groups: int = MAX_PAGE_LIMIT,
    max_chars: int = MAX_GET_CHAR_LIMIT,
) -> tuple[dict[str, int], bool, bool]:
    sorted_counts = dict(sorted(counts.items(), key=lambda x: x[1], reverse=True))
    bounded_counts: dict[str, int] = {}
    hit_group_limit = False
    hit_payload_limit = False

    cur_len = 2  # "{}"
    for g_idx, (k, cnt) in enumerate(sorted_counts.items()):
        if g_idx >= max_groups:
            hit_group_limit = True
            break
        k_json = json.dumps(k)
        add_len = len(k_json) + 2 + len(str(cnt)) + (2 if g_idx > 0 else 0)
        if cur_len + add_len > max_chars:
            if not bounded_counts:
                truncated_k, _, _ = _bound_string(k, max(50, max_chars - 30))
                bounded_counts[truncated_k] = cnt
            hit_payload_limit = True
            break
        bounded_counts[k] = cnt
        cur_len += add_len

    return bounded_counts, hit_group_limit, hit_payload_limit


class ObservationCount:
    """Count records or calculate group-by frequencies in an artifact."""

    name = "observation.count"

    def __init__(self, store: ObservationStore) -> None:
        self.store = store

    async def invoke(
        self, request: CapabilityRequest, cancellation: CancellationToken
    ) -> Result[ActionResult, ActionFailure]:
        if cancellation.cancelled:
            return Failure(_fail(request, "CANCELLED", "Count cancelled"))

        session_id = str(request.arguments.get("session_id", "")).strip()
        ref = str(request.arguments.get("ref", "")).strip()
        field = request.arguments.get("field")

        if not session_id:
            return Failure(_fail(request, "UNAUTHORIZED", "Missing session_id"))
        if not ref:
            return Failure(_fail(request, "INVALID_ARGUMENT", "Missing ref"))

        artifact = await _get_artifact(self.store, ref, session_id=session_id)
        if artifact is None:
            return Failure(_fail(request, "ARTIFACT_NOT_FOUND", f"Artifact {ref} not found"))

        _, records = _extract_primary_list(artifact.content)

        output: dict[str, Any]
        if not field:
            total_records = len(records)
            hit_scan_limit = total_records > MAX_VISITED_LIMIT
            reported_count = min(total_records, MAX_VISITED_LIMIT) if hit_scan_limit else total_records
            complete = not hit_scan_limit
            reason = (
                f"Hit maximum count scan limit ({MAX_VISITED_LIMIT} records inspected)"
                if hit_scan_limit
                else None
            )
            output = {
                "ref": ref,
                "total_count": reported_count,
                "complete": complete,
                "reason": reason,
            }
        else:
            count_res = await _count_grouped_records(records, str(field), cancellation)
            if count_res is None:
                return Failure(_fail(request, "CANCELLED", "Count cancelled"))
            raw_counts, total_scanned, hit_scan_limit = count_res
            bounded_counts, hit_group_limit, hit_payload_limit = _bound_counts_dictionary(raw_counts)

            complete = (
                not hit_scan_limit
                and not hit_group_limit
                and not hit_payload_limit
                and len(bounded_counts) == len(raw_counts)
            )
            if hit_scan_limit:
                reason = f"Hit maximum count scan limit ({MAX_VISITED_LIMIT} records inspected)"
            elif hit_payload_limit:
                reason = (
                    f"Count results exceeded maximum payload limit "
                    f"({MAX_GET_CHAR_LIMIT} chars, {len(bounded_counts)} of {len(raw_counts)} groups returned)"
                )
            elif hit_group_limit:
                reason = (
                    f"Count results exceeded maximum group limit "
                    f"({MAX_PAGE_LIMIT} of {len(raw_counts)} groups returned)"
                )
            else:
                reason = None

            output = {
                "ref": ref,
                "field": str(field),
                "total_records": total_scanned,
                "complete": complete,
                "reason": reason,
                "counts": bounded_counts,
            }

        output = _enforce_final_payload_limit(
            output, list_key=None, max_chars=MAX_GET_CHAR_LIMIT
        )

        derived_id = await persist_observation_artifact(
            store=self.store,
            session_id=artifact.session_id,
            turn_id=artifact.turn_id,
            capability=self.name,
            request_id=request.record_id,
            data_class=artifact.data_class,
            output=output,
            derived_from=(ref,),
        )
        if derived_id:
            output["artifact_id"] = derived_id

        return Success(_ok(request, output, {"source_artifact_id": ref}))


def _parse_json_pointer(path: str) -> Result[list[str], str]:
    if path == "":
        return Success([])
    if not path.startswith("/"):
        return Failure("JSON pointer must be empty or start with '/'")
    segments: list[str] = []
    for raw_seg in path[1:].split("/"):
        if re.search(r"~(?![01])", raw_seg) or raw_seg.endswith("~"):
            return Failure(f"Invalid escape sequence in segment '{raw_seg}'")
        segments.append(raw_seg.replace("~1", "/").replace("~0", "~"))
    return Success(segments)


def _traverse_json_pointer(
    curr: Any,  # noqa: ANN401
    segments: list[str],
) -> Result[Any, tuple[str, str]]:
    for seg in segments:
        if isinstance(curr, dict):
            if seg not in curr:
                return Failure(("PATH_NOT_FOUND", f"Key '{seg}' not found in object"))
            curr = curr[seg]
        elif isinstance(curr, (list, tuple)):
            if not seg.isdigit() or (len(seg) > 1 and seg.startswith("0")):
                return Failure(("INVALID_POINTER", f"Invalid array index '{seg}'"))
            idx = int(seg)
            if idx >= len(curr):
                return Failure(("PATH_NOT_FOUND", f"Index {idx} out of range"))
            curr = curr[idx]
        else:
            return Failure(
                ("INVALID_POINTER", f"Cannot index into {type(curr).__name__} with '{seg}'")
            )
    return Success(curr)


def _bound_get_value(
    value: Any, budget_chars: int = MAX_GET_CHAR_LIMIT - 500  # noqa: ANN401
) -> tuple[Any, bool, str | None]:  # noqa: ANN401, PLR0911
    """Ensure observation.get output is safely bounded and report truncation."""
    if isinstance(value, str):
        if len(json.dumps(value)) > budget_chars:
            return _bound_string(value, budget_chars)
        return value, True, None

    if isinstance(value, (list, tuple)):
        bounded_items, complete, reason = _bound_output_collection(
            value, max_items=MAX_PAGE_LIMIT, max_chars=budget_chars
        )
        if not complete and reason:
            reason = f"{reason}. Use observation.slice to paginate."
        return bounded_items, complete, reason

    if isinstance(value, dict):
        serialized = json.dumps(value)
        if len(serialized) > budget_chars:
            reduced_text = render_observation_payload(value, budget_chars=max(50, budget_chars - 30))
            cand_dict = {"_preview": reduced_text}
            if len(json.dumps(cand_dict)) > budget_chars:
                bounded_prev, _, _ = _bound_string(reduced_text, max(50, budget_chars - 30))
                cand_dict = {"_preview": bounded_prev}
            return (
                cand_dict,
                False,
                f"Object payload exceeded maximum size limit ({len(serialized)} bytes > {MAX_GET_CHAR_LIMIT}). Use a specific JSON pointer path, observation.slice, or observation.filter.",
            )
        return value, True, None

    return value, True, None


class ObservationGet:
    """Extract a single field or nested path using JSON pointer notation."""

    name = "observation.get"

    def __init__(self, store: ObservationStore) -> None:
        self.store = store

    async def invoke(  # noqa: PLR0911
        self, request: CapabilityRequest, cancellation: CancellationToken
    ) -> Result[ActionResult, ActionFailure]:
        if cancellation.cancelled:
            return Failure(_fail(request, "CANCELLED", "Get cancelled"))

        session_id = str(request.arguments.get("session_id", "")).strip()
        ref = str(request.arguments.get("ref", "")).strip()
        if "path" not in request.arguments:
            return Failure(_fail(request, "INVALID_ARGUMENT", "Missing path"))
        path = str(request.arguments.get("path", ""))

        if not session_id:
            return Failure(_fail(request, "UNAUTHORIZED", "Missing session_id"))
        if not ref:
            return Failure(_fail(request, "INVALID_ARGUMENT", "Missing ref"))

        parse_res = _parse_json_pointer(path)
        if isinstance(parse_res, Failure):
            return Failure(_fail(request, "INVALID_POINTER", parse_res.failure()))
        segments = parse_res.unwrap()

        artifact = await _get_artifact(self.store, ref, session_id=session_id)
        if artifact is None:
            return Failure(_fail(request, "ARTIFACT_NOT_FOUND", f"Artifact {ref} not found"))

        trav_res = _traverse_json_pointer(artifact.content, segments)
        if isinstance(trav_res, Failure):
            err_code, err_msg = trav_res.failure()
            return Failure(_fail(request, err_code, err_msg))

        bounded_val, complete, reason = _bound_get_value(trav_res.unwrap())
        output: dict[str, Any] = {
            "ref": ref,
            "path": path,
            "value": bounded_val,
            "complete": complete,
            "reason": reason,
        }
        output = _enforce_final_payload_limit(
            output, list_key=None, max_chars=MAX_GET_CHAR_LIMIT
        )
        return Success(_ok(request, output, {"source_artifact_id": ref}))


def register_observation_capabilities(
    registry: CapabilityRegistry, store: ObservationStore
) -> None:
    """Register all observation processing capabilities into the capability registry."""
    descriptors_and_impls = (
        (
            ObservationSearch(store),
            {
                "session_id": {"type": "string", "description": "Session identifier for session-scoped artifacts"},
                "ref": {"type": "string", "description": "Artifact reference handle (e.g. obs_1)"},
                "pattern": {"type": "string", "description": "Substring search pattern"},
                "fields": {"type": "array", "items": {"type": "string"}, "description": "Optional fields to restrict search to"},
                "case": {"type": "boolean", "description": "Case-sensitive search"},
                "offset": {"type": "integer", "minimum": 0, "default": 0},
                "limit": {"type": "integer", "minimum": 1, "maximum": 1000, "default": 100},
            },
            ["ref", "pattern"],
            "Search within a stored observation artifact without loading all data into prompt.",
        ),
        (
            ObservationFilter(store),
            {
                "session_id": {"type": "string", "description": "Session identifier for session-scoped artifacts"},
                "ref": {"type": "string", "description": "Artifact reference handle (e.g. obs_1)"},
                "field": {"type": "string", "description": "Field name to filter on"},
                "op": {"type": "string", "enum": ["contains", "equals", "startswith", "endswith", "in", "gt", "gte", "lt", "lte"]},
                "value": {"description": "Target value to compare against"},
                "offset": {"type": "integer", "minimum": 0, "default": 0},
                "limit": {"type": "integer", "minimum": 1, "maximum": 1000, "default": 100},
            },
            ["ref", "field", "op", "value"],
            "Filter records inside a stored observation artifact by structured conditions.",
        ),
        (
            ObservationSlice(store),
            {
                "session_id": {"type": "string", "description": "Session identifier for session-scoped artifacts"},
                "ref": {"type": "string", "description": "Artifact reference handle (e.g. obs_1)"},
                "offset": {"type": "integer", "minimum": 0, "default": 0},
                "limit": {"type": "integer", "minimum": 1, "maximum": 1000, "default": 20},
            },
            ["ref"],
            "Retrieve a pagination slice of records or text lines from a stored artifact.",
        ),
        (
            ObservationCount(store),
            {
                "session_id": {"type": "string", "description": "Session identifier for session-scoped artifacts"},
                "ref": {"type": "string", "description": "Artifact reference handle (e.g. obs_1)"},
                "field": {"type": "string", "description": "Optional field for group-by count"},
            },
            ["ref"],
            "Count records or group-by frequency distribution in a stored artifact.",
        ),
        (
            ObservationGet(store),
            {
                "session_id": {"type": "string", "description": "Session identifier for session-scoped artifacts"},
                "ref": {"type": "string", "description": "Artifact reference handle (e.g. obs_1)"},
                "path": {"type": "string", "description": "JSON pointer path (e.g. /applications/0/name)"},
            },
            ["ref", "path"],
            "Extract one field or record from a stored artifact using JSON pointer notation.",
        ),
    )

    for impl, props, required, desc in descriptors_and_impls:
        descriptor = CapabilityDescriptor(
            name=impl.name,
            description=desc,
            input_schema={
                "type": "object",
                "properties": props,
                "required": required,
                "additionalProperties": False,
            },
            risk_class=RiskClass.LOW,
            side_effects=(),
            isolation="host-api",
            verification_plan=("artifact-reference-resolved",),
        )
        registry.register(RegisteredCapability(descriptor, implementation=impl))
