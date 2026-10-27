"""Pure-Python observation processing capabilities (safe grep/awk over stored artifacts)."""

from __future__ import annotations

import asyncio
import itertools
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


def _iter_text_lines(text: str) -> Iterator[str]:
    start = 0
    text_len = len(text)
    while start < text_len:
        end = text.find("\n", start)
        if end == -1:
            yield text[start:]
            break
        line = text[start:end]
        if line.endswith("\r"):
            line = line[:-1]
        yield line
        start = end + 1


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

    async def invoke(  # noqa: PLR0911
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
        complete = not hit_scan_limit and offset == 0 and total_matches <= limit
        if hit_scan_limit:
            reason = f"Hit maximum search scan limit ({MAX_VISITED_LIMIT} items inspected or {MAX_SCAN_LIMIT} matches)"
        elif not complete:
            reason = f"Results paginated ({len(paginated)} of {total_matches} matches returned)"
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
            "matches": paginated,
        }

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

        total_matches = len(filtered)
        paginated = filtered[offset : offset + limit]
        complete = not hit_scan_limit and offset == 0 and total_matches <= limit
        if hit_scan_limit:
            reason = f"Hit maximum filter scan limit ({MAX_VISITED_LIMIT} items inspected or {MAX_SCAN_LIMIT} matches)"
        elif not complete:
            reason = f"Results paginated ({len(paginated)} of {total_matches} matches returned)"
        else:
            reason = None

        output: dict[str, Any] = {
            "ref": ref,
            "field": field,
            "op": op,
            "value": val,
            "total_matches": total_matches,
            "offset": offset,
            "limit": limit,
            "complete": complete,
            "reason": reason,
            "results": paginated,
        }

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
            text = content["text"]
            total_count = text.count("\n") + (1 if text and not text.endswith("\n") else 0)
            sliced = list(itertools.islice(_iter_text_lines(text), offset, offset + limit))
        else:
            _, all_items = _extract_primary_list(content)
            total_count = len(all_items)
            sliced = list(all_items[offset : offset + limit])

        output: dict[str, Any] = {
            "ref": ref,
            "offset": offset,
            "limit": limit,
            "total_count": total_count,
            "items": sliced,
        }

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
            output = {
                "ref": ref,
                "total_count": len(records),
            }
        else:
            counts: dict[str, int] = {}
            for rec in records:
                if isinstance(rec, dict):
                    v = rec.get(str(field))
                    if isinstance(v, (list, tuple)):
                        for sub_v in v:
                            s = str(sub_v)
                            counts[s] = counts.get(s, 0) + 1
                    else:
                        s = str(v) if v is not None else "null"
                        counts[s] = counts.get(s, 0) + 1
            output = {
                "ref": ref,
                "field": str(field),
                "total_records": len(records),
                "counts": dict(sorted(counts.items(), key=lambda x: x[1], reverse=True)),
            }

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


def _bound_get_value(value: Any) -> tuple[Any, bool, str | None]:  # noqa: ANN401, PLR0911
    """Ensure observation.get output is safely bounded and report truncation."""
    if isinstance(value, str):
        if len(value) > MAX_GET_CHAR_LIMIT:
            truncated = value[:MAX_GET_CHAR_LIMIT] + f"... [truncated {len(value) - MAX_GET_CHAR_LIMIT} chars]"
            return (
                truncated,
                False,
                f"Value exceeded maximum character limit ({MAX_GET_CHAR_LIMIT} of {len(value)} chars returned). Use observation.slice or observation.search.",
            )
        return value, True, None

    if isinstance(value, (list, tuple)):
        if len(value) > MAX_PAGE_LIMIT:
            bounded = list(value[:MAX_PAGE_LIMIT])
            return (
                bounded,
                False,
                f"List value exceeded maximum item limit ({MAX_PAGE_LIMIT} of {len(value)} items returned). Use observation.slice to paginate.",
            )
        serialized = json.dumps(value)
        if len(serialized) > MAX_GET_CHAR_LIMIT:
            bounded_items: list[Any] = []
            cur_len = 2
            for item in value:
                item_str = json.dumps(item)
                if cur_len + len(item_str) + 2 > MAX_GET_CHAR_LIMIT:
                    break
                bounded_items.append(item)
                cur_len += len(item_str) + 2
            return (
                bounded_items,
                False,
                f"Serialized list exceeded payload limit ({len(bounded_items)} of {len(value)} items returned). Use observation.slice to paginate.",
            )
        return list(value) if isinstance(value, tuple) else value, True, None

    if isinstance(value, dict):
        serialized = json.dumps(value)
        if len(serialized) > MAX_GET_CHAR_LIMIT:
            reduced_text = render_observation_payload(value, budget_chars=MAX_GET_CHAR_LIMIT)
            return (
                {"_preview": reduced_text},
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
        path = str(request.arguments.get("path", "")).strip()

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
