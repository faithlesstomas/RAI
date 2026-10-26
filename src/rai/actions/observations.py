"""Pure-Python observation processing capabilities (safe grep/awk over stored artifacts)."""

from __future__ import annotations

import asyncio
import json
import re
from typing import Any

from returns.result import Failure, Result, Success

from rai.assistant.observation_budget import persist_observation_artifact
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
DEFAULT_PAGE_LIMIT = 100
MAX_PAGE_LIMIT = 1000


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


def _extract_primary_list(content: dict[str, Any]) -> tuple[str | None, list[Any]]:
    for key in ("applications", "results", "documents", "entries", "processes", "items", "episodes"):
        if key in content and isinstance(content[key], (list, tuple)):
            return key, list(content[key])
    for k, v in content.items():
        if isinstance(v, (list, tuple)):
            return str(k), list(v)
    return None, []


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


def _search_text_lines(
    lines: list[str],
    compiled: re.Pattern[str],
    cancellation: CancellationToken,
) -> Result[tuple[list[dict[str, Any]], bool], None]:
    matched: list[dict[str, Any]] = []
    for idx, line in enumerate(lines, 1):
        if idx % 100 == 0 and cancellation.cancelled:
            return Failure(None)
        if compiled.search(line):
            start_ctx = max(0, idx - 2)
            end_ctx = min(len(lines), idx + 1)
            context_snippet = "\n".join(lines[start_ctx:end_ctx])
            matched.append({
                "line_number": idx,
                "line": line.strip(),
                "context": context_snippet,
            })
            if len(matched) >= MAX_SCAN_LIMIT:
                return Success((matched, True))
    return Success((matched, False))


def _search_records(
    records: list[Any],
    compiled: re.Pattern[str],
    fields: list[str] | None,
    cancellation: CancellationToken,
) -> Result[tuple[list[Any], bool], None]:
    matched: list[Any] = []
    for count, item in enumerate(records):
        if count % 100 == 0 and cancellation.cancelled:
            return Failure(None)
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
            return Success((matched, True))
    return Success((matched, False))


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
            lines = content["text"].splitlines()
            search_res = _search_text_lines(lines, compiled, cancellation)
        else:
            _, records = _extract_primary_list(content)
            search_res = _search_records(records, compiled, fields, cancellation)

        if isinstance(search_res, Failure):
            return Failure(_fail(request, "CANCELLED", "Search cancelled"))
        matched, hit_scan_limit = search_res.unwrap()

        total_matches = len(matched)
        paginated = matched[offset : offset + limit]
        complete = not hit_scan_limit and offset == 0 and total_matches <= limit
        if hit_scan_limit:
            reason = f"Hit maximum search scan limit ({MAX_SCAN_LIMIT})"
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
        filtered: list[Any] = []
        hit_scan_limit = False

        for count, rec in enumerate(records):
            if count % 100 == 0 and cancellation.cancelled:
                return Failure(_fail(request, "CANCELLED", "Filter cancelled"))
            if isinstance(rec, dict) and _evaluate_filter_condition(rec.get(field), op, val):
                filtered.append(rec)
                if len(filtered) >= MAX_SCAN_LIMIT:
                    hit_scan_limit = True
                    break

        total_matches = len(filtered)
        paginated = filtered[offset : offset + limit]
        complete = not hit_scan_limit and offset == 0 and total_matches <= limit
        if hit_scan_limit:
            reason = f"Hit maximum filter scan limit ({MAX_SCAN_LIMIT})"
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
            all_items = content["text"].splitlines()
        else:
            _, all_items = _extract_primary_list(content)

        sliced = all_items[offset : offset + limit]
        output: dict[str, Any] = {
            "ref": ref,
            "offset": offset,
            "limit": limit,
            "total_count": len(all_items),
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
    if not path.startswith("/"):
        return Failure("JSON pointer must start with '/'")
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
        path = str(request.arguments.get("path", "")).strip()

        if not session_id:
            return Failure(_fail(request, "UNAUTHORIZED", "Missing session_id"))
        if not ref or not path:
            return Failure(_fail(request, "INVALID_ARGUMENT", "Missing ref or path"))

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

        output = {
            "ref": ref,
            "path": path,
            "value": trav_res.unwrap(),
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
            ["session_id", "ref", "pattern"],
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
            ["session_id", "ref", "field", "op", "value"],
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
            ["session_id", "ref"],
            "Retrieve a pagination slice of records or text lines from a stored artifact.",
        ),
        (
            ObservationCount(store),
            {
                "session_id": {"type": "string", "description": "Session identifier for session-scoped artifacts"},
                "ref": {"type": "string", "description": "Artifact reference handle (e.g. obs_1)"},
                "field": {"type": "string", "description": "Optional field for group-by count"},
            },
            ["session_id", "ref"],
            "Count records or group-by frequency distribution in a stored artifact.",
        ),
        (
            ObservationGet(store),
            {
                "session_id": {"type": "string", "description": "Session identifier for session-scoped artifacts"},
                "ref": {"type": "string", "description": "Artifact reference handle (e.g. obs_1)"},
                "path": {"type": "string", "description": "JSON pointer path (e.g. /applications/0/name)"},
            },
            ["session_id", "ref", "path"],
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
