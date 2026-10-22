"""Pure functional observation budgeter and structural context reducer."""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, ConfigDict, Field
from returns.result import Success

from rai.kernel.records import (
    DataClass,
    ProducerIdentity,
    ProvenanceReference,
    _new_id,
    _utc_now,
)

from .ports import ObservationStore
from .records import (
    ObservationArtifact,
    ObservationSchemaSummary,
    ToolObservation,
    summarize_observation_schema,
)

TECHNICAL_FIELDS = frozenset(
    {
        "handle",
        "fingerprint",
        "expires_at",
        "allowed_roots_checked",
        "current_user_and_start_time",
        "verification",
    }
)


class RenderedObservation(BaseModel):
    """Budgeted representation of an observation ready for model context injection."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    artifact_id: str | None = None
    capability: str = "tool"
    summary: str = ""
    content: dict[str, Any] = Field(default_factory=dict)
    rendered_text: str = ""
    shown_count: int | None = None
    total_count: int | None = None
    is_reduced: bool = False
    preview_footer: str | None = None


def _clean_technical_fields(val: Any) -> Any:  # noqa: ANN401
    """Recursively strip technical capability fields like handles, fingerprints, and expiry."""
    if isinstance(val, dict):
        return {
            k: _clean_technical_fields(v)
            for k, v in val.items()
            if str(k) not in TECHNICAL_FIELDS
        }
    if isinstance(val, (list, tuple)):
        return [_clean_technical_fields(item) for item in val]
    return val


def _shorten_strings(val: Any, max_str_len: int = 250) -> tuple[Any, bool]:  # noqa: ANN401
    """Recursively shorten long string values with explicit in-string markers."""
    truncated = False
    if isinstance(val, str):
        if len(val) > max_str_len:
            marker = f"... [truncated string: {max_str_len} shown of {len(val)}]"
            return val[:max_str_len] + marker, True
        return val, False
    if isinstance(val, dict):
        res = {}
        for k, v in val.items():
            shortened, was_trunc = _shorten_strings(v, max_str_len)
            res[k] = shortened
            if was_trunc:
                truncated = True
        return res, truncated
    if isinstance(val, (list, tuple)):
        res_list = []
        for item in val:
            shortened, was_trunc = _shorten_strings(item, max_str_len)
            res_list.append(shortened)
            if was_trunc:
                truncated = True
        return res_list, truncated
    return val, False


def render_observation_payload(content: object, budget_chars: int = 64000) -> str:  # noqa: PLR0911, PLR0912
    """Serialize and bound content structurally to valid JSON without mid-token string cutting."""
    if isinstance(content, str):
        if len(content) <= budget_chars:
            return content
        return content[:budget_chars] + f"... [truncated {len(content) - budget_chars} chars]"

    cleaned = _clean_technical_fields(content)
    if isinstance(cleaned, dict):
        limits_by_key = {
            "results": 20,
            "capabilities": 50,
            "application_names": 1024,
            "applications": 1024,
            "documents": 100,
            "entries": 100,
            "processes": 100,
        }
        bounded_dict: dict[str, Any] = {}
        for k, v in cleaned.items():
            limit = limits_by_key.get(str(k), 3)
            if isinstance(v, (list, tuple)) and len(v) > limit:
                bounded_dict[k] = list(v[:limit]) + [
                    f"... (+{len(v) - limit} more items truncated)"
                ]
            else:
                bounded_dict[k] = v
        try:
            dumped = json.dumps(bounded_dict, ensure_ascii=False)
        except Exception:  # noqa: BLE001
            dumped = str(bounded_dict)

        if len(dumped) <= budget_chars:
            return dumped

        rendered = render_observation(
            {"content": cleaned, "capability": "tool", "summary": ""},
            budget_chars=budget_chars,
        )
        return rendered.rendered_text

    if isinstance(cleaned, (list, tuple)):
        total_len = len(cleaned)
        k = total_len
        while k > 0:
            candidate = {
                "items": list(cleaned[:k]),
                "_meta": {"shown": k, "total": total_len, "truncated": True},
            }
            dumped = json.dumps(candidate, ensure_ascii=False)
            if len(dumped) <= budget_chars:
                return dumped
            k = max(0, int(k * 0.7) - 1)
        fallback = {"_meta": {"shown": 0, "total": total_len, "truncated": True}}
        return json.dumps(fallback, ensure_ascii=False)

    try:
        dumped = json.dumps(cleaned, ensure_ascii=False)
    except Exception:  # noqa: BLE001
        dumped = str(cleaned)
    if len(dumped) <= budget_chars:
        return dumped
    return dumped[:budget_chars]


def render_observation(  # noqa: PLR0912, PLR0915
    artifact: ObservationArtifact | ToolObservation | dict[str, Any],
    budget_chars: int = 4000,
) -> RenderedObservation:
    """Render an observation according to context budget using structural reduction."""
    artifact_id: str | None = None
    capability: str = "tool"
    summary: str = ""
    raw_content: Any = {}

    if isinstance(artifact, ObservationArtifact):
        artifact_id = artifact.artifact_id
        capability = artifact.capability
        raw_content = artifact.content
        summary = f"Observation {artifact_id} ({capability})"
    elif isinstance(artifact, ToolObservation):
        artifact_id = artifact.artifact_id
        capability = artifact.capability
        summary = artifact.summary
        raw_content = artifact.content
    elif isinstance(artifact, dict):
        if "content" in artifact and any(
            k in artifact for k in ("capability", "summary", "artifact_id", "source_id")
        ):
            artifact_id = artifact.get("artifact_id")
            capability = str(artifact.get("capability", "tool"))
            summary = str(artifact.get("summary", ""))
            raw_content = artifact.get("content", {})
        else:
            artifact_id = artifact.get("artifact_id")
            capability = str(artifact.get("capability", "tool"))
            summary = str(artifact.get("summary", ""))
            raw_content = artifact
    else:
        raw_content = artifact

    content_dict = raw_content if isinstance(raw_content, dict) else {"value": raw_content}
    cleaned = _clean_technical_fields(content_dict)
    full_json = json.dumps(cleaned, ensure_ascii=False)

    schema: ObservationSchemaSummary = (
        artifact.schema_summary
        if isinstance(artifact, ObservationArtifact)
        else summarize_observation_schema(cleaned)
    )
    total_records = schema.total_count

    # If within budget, return full observation
    if len(full_json) <= budget_chars:
        return RenderedObservation(
            artifact_id=artifact_id,
            capability=capability,
            summary=summary,
            content=cleaned,
            rendered_text=full_json,
            shown_count=total_records,
            total_count=total_records,
            is_reduced=False,
            preview_footer=None,
        )

    # Need reduction
    is_reduced = True
    active_content = cleaned

    # Find the primary list field to slice, if any
    list_field_name: str | None = None
    max_list_len = 0
    for k, v in active_content.items():
        if isinstance(v, (list, tuple)) and len(v) > max_list_len:
            max_list_len = len(v)
            list_field_name = k

    # Step 1: Try shortening long strings if present
    shortened_content, had_trunc_strings = _shorten_strings(active_content, max_str_len=300)
    shortened_json = json.dumps(shortened_content, ensure_ascii=False)
    if len(shortened_json) <= budget_chars:
        shown = total_records
        footer = _build_footer(artifact_id, capability, schema, shown, total_records)
        return RenderedObservation(
            artifact_id=artifact_id,
            capability=capability,
            summary=summary,
            content=shortened_content,
            rendered_text=shortened_json,
            shown_count=shown,
            total_count=total_records,
            is_reduced=True,
            preview_footer=footer,
        )

    # Step 2: Binary search / linear scan on list items
    if list_field_name is not None and max_list_len > 0:
        raw_list = active_content[list_field_name]
        low = 0
        high = max_list_len
        best_k = 0
        best_candidate: dict[str, Any] = {}

        # Scan downwards from high or binary search
        while low <= high:
            mid = (low + high) // 2
            cand_items = list(raw_list[:mid])
            shortened_items, _ = _shorten_strings(cand_items, max_str_len=200)
            item_fields = schema.item_fields.get(list_field_name, ())
            meta_entry: dict[str, Any] = {
                "shown": mid,
                "total": max_list_len,
                "truncated": True,
            }
            if item_fields:
                meta_entry["fields"] = list(item_fields)
            if artifact_id:
                meta_entry["artifact_id"] = artifact_id

            cand_dict = {
                **active_content,
                list_field_name: shortened_items,
                "_meta": meta_entry,
            }
            cand_json = json.dumps(cand_dict, ensure_ascii=False)
            if len(cand_json) <= budget_chars:
                best_k = mid
                best_candidate = cand_dict
                low = mid + 1
            else:
                high = mid - 1

        if best_k > 0:
            footer = _build_footer(artifact_id, capability, schema, best_k, max_list_len)
            return RenderedObservation(
                artifact_id=artifact_id,
                capability=capability,
                summary=summary,
                content=best_candidate,
                rendered_text=json.dumps(best_candidate, ensure_ascii=False),
                shown_count=best_k,
                total_count=max_list_len,
                is_reduced=True,
                preview_footer=footer,
            )

    # Step 3: Schema-only preview fallback
    item_fields = (
        schema.item_fields.get(list_field_name, ())
        if list_field_name
        else ()
    )
    schema_preview: dict[str, Any] = {
        "_schema_preview": {
            "top_level_keys": list(schema.top_level_keys),
            "total_count": total_records or max_list_len,
            "fields": list(item_fields),
            "truncated": True,
        }
    }
    if artifact_id:
        schema_preview["_schema_preview"]["artifact_id"] = artifact_id
    schema_json = json.dumps(schema_preview, ensure_ascii=False)
    footer = _build_footer(
        artifact_id, capability, schema, 0, total_records or max_list_len
    )
    return RenderedObservation(
        artifact_id=artifact_id,
        capability=capability,
        summary=summary,
        content=schema_preview,
        rendered_text=schema_json,
        shown_count=0,
        total_count=total_records or max_list_len,
        is_reduced=True,
        preview_footer=footer,
    )


def _build_footer(
    artifact_id: str | None,
    capability: str,
    schema: ObservationSchemaSummary,
    shown: int | None,
    total: int | None,
) -> str:
    """Generate model affordance footer informing the LLM how to query the full artifact."""
    handle = artifact_id or f"artifact({capability})"
    fields_list = []
    for item_f in schema.item_fields.values():
        fields_list.extend(item_f)
    if not fields_list and schema.top_level_keys:
        fields_list = [k for k in schema.top_level_keys if not k.startswith("_")]

    fields_str = f" (fields: {', '.join(fields_list[:5])})" if fields_list else ""
    total_str = f"{total}" if total is not None else "multiple"
    shown_str = f"{shown}" if shown is not None else "0"

    return (
        f"{handle}: {total_str} records{fields_str}; {shown_str} shown. "
        f"Use observation.search/filter on {handle} or Python to inspect the rest."
    )


async def persist_observation_artifact(  # noqa: PLR0913
    store: ObservationStore | None,
    session_id: str,
    turn_id: str,
    capability: str,
    request_id: str,
    data_class: DataClass | str,
    output: dict[str, Any],
    producer: ProducerIdentity | None = None,
    derived_from: tuple[str, ...] = (),
) -> str | None:
    """Persist a complete execution output into ObservationStore if available and permitted."""
    if store is None or not hasattr(store, "put"):
        return None
    dc = DataClass(data_class) if isinstance(data_class, str) else data_class
    if dc in (DataClass.SECRET, DataClass.BLOCKED):
        return None

    artifact_id: str
    if hasattr(store, "allocate_artifact_id"):
        alias_res = await store.allocate_artifact_id(session_id)
        if isinstance(alias_res, Success):
            artifact_id = alias_res.unwrap()
        else:
            artifact_id = f"obs_{_new_id()[:8]}"
    else:
        artifact_id = f"obs_{_new_id()[:8]}"

    json_str = json.dumps(output, ensure_ascii=False)
    prod = producer or ProducerIdentity(
        producer_id="capability-runner", kind="service", version="1.0.0"
    )
    artifact = ObservationArtifact(
        record_id=_new_id(),
        artifact_id=artifact_id,
        session_id=session_id,
        turn_id=turn_id,
        capability=capability,
        request_id=request_id,
        data_class=dc,
        content=output,
        schema_summary=summarize_observation_schema(output),
        size_chars=len(json_str),
        created_at=_utc_now(),
        provenance=(
            ProvenanceReference(
                source_id=request_id,
                source_type="action_result",
                source_version="1.0.0",
                relation="RECORDED_FROM",
                producer=prod,
            ),
        ),
        derived_from=derived_from,
        producer=prod,
    )
    put_res = await store.put(artifact)
    if isinstance(put_res, Success):
        return artifact_id
    return None
