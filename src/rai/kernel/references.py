"""Kernel-resolved observation references for capability tool pipelines."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

from returns.result import Failure, Result, Success

from rai.assistant.ports import ObservationStore
from rai.assistant.records import ObservationArtifact
from rai.kernel.records import (
    DataClass,
    ProducerIdentity,
    ProvenanceReference,
    max_data_class,
)

PRODUCER = ProducerIdentity(
    producer_id="rai.kernel.references",
    kind="resolver",
    version="1.0.0",
)


@dataclass(frozen=True)
class ArtifactReference:
    """A single observation reference parsed from capability arguments."""

    ref: str
    path: str = ""


@dataclass(frozen=True)
class PreflightReferenceScan:
    """Metadata gathered from scanning capability arguments for artifact references."""

    has_references: bool
    referenced_artifacts: dict[str, ObservationArtifact]
    effective_data_class: DataClass
    referenced_ids: tuple[str, ...]


def is_artifact_reference(val: object) -> bool:
    """Check if a value represents a kernel observation reference object."""
    if isinstance(val, (dict, Mapping)):
        return "$ref" in val and isinstance(val["$ref"], str)
    return False


def extract_references(data: object) -> list[ArtifactReference]:
    """Find all artifact references in an arbitrary nested data structure."""
    refs: list[ArtifactReference] = []

    def _walk(item: object) -> None:
        if is_artifact_reference(item):
            refs.append(
                ArtifactReference(
                    ref=str(item["$ref"]).strip(),
                    path=str(item.get("path", "")).strip(),
                )
            )
            return
        if isinstance(item, (dict, Mapping)):
            for v in item.values():
                _walk(v)
        elif isinstance(item, (list, tuple, Sequence)) and not isinstance(
            item, (str, bytes, bytearray)
        ):
            for v in item:
                _walk(v)

    _walk(data)
    return refs


def parse_json_pointer(path: str) -> Result[list[str], str]:
    """Parse an RFC 6901 JSON pointer into unescaped path segments."""
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


def traverse_json_pointer(
    curr: object,
    segments: list[str],
) -> Result[Any, tuple[str, str]]:
    """Traverse a JSON-compatible data structure along parsed segments."""
    for seg in segments:
        if isinstance(curr, (dict, Mapping)):
            if seg not in curr:
                return Failure(
                    ("PATH_NOT_FOUND", f"Key '{seg}' not found in object")
                )
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
                (
                    "INVALID_POINTER",
                    f"Cannot index into {type(curr).__name__} with '{seg}'",
                )
            )
    return Success(curr)


async def get_artifact_for_reference(  # noqa: PLR0911
    store: ObservationStore,
    session_id: str,
    ref: str,
    now: datetime | None = None,
) -> Result[ObservationArtifact, tuple[str, str]]:
    """Lookup an artifact by session alias (e.g. 'obs_1') or stable ID within session scope."""
    if not session_id:
        return Failure(
            ("UNAUTHORIZED", "Cannot resolve reference without session context")
        )

    current_time = now or datetime.now(timezone.utc)
    # Query with minimum time to retrieve record regardless of expiration, so we can distinguish expired vs not found
    get_res = await store.get(
        session_id, ref, now=datetime.min.replace(tzinfo=timezone.utc)
    )
    if isinstance(get_res, Failure):
        return Failure(("STORE_ERROR", str(get_res.failure().message)))

    artifact = get_res.unwrap()
    if artifact is None:
        return Failure(
            (
                "ARTIFACT_NOT_FOUND",
                f"Referenced observation '{ref}' not found in session",
            )
        )

    # Enforce session isolation
    if artifact.session_id != session_id:
        return Failure(
            (
                "SESSION_VIOLATION",
                f"Referenced observation '{ref}' does not belong to session",
            )
        )

    # Check expiration
    if artifact.expires_at is not None and artifact.expires_at < current_time:
        return Failure(
            ("REFERENCE_EXPIRED", f"Referenced observation '{ref}' has expired")
        )

    # Check forbidden data classes
    if artifact.data_class in (DataClass.SECRET, DataClass.BLOCKED):
        return Failure(
            (
                "DATA_CLASS_FORBIDDEN",
                f"Referenced observation '{ref}' contains forbidden data class: {artifact.data_class}",
            )
        )

    return Success(artifact)


async def preflight_scan_references(
    arguments: dict[str, Any],
    store: ObservationStore | None,
    session_id: str,
    base_data_class: DataClass,
    now: datetime | None = None,
) -> Result[PreflightReferenceScan, tuple[str, str]]:
    """Scan arguments for references, look up artifacts, and compute effective data class."""
    raw_refs = extract_references(arguments)
    if not raw_refs:
        return Success(
            PreflightReferenceScan(
                has_references=False,
                referenced_artifacts={},
                effective_data_class=base_data_class,
                referenced_ids=(),
            )
        )

    if store is None:
        return Failure(
            (
                "STORE_UNAVAILABLE",
                "ObservationStore is not available to resolve references",
            )
        )

    artifacts: dict[str, ObservationArtifact] = {}
    for ref_item in raw_refs:
        if ref_item.ref not in artifacts:
            res = await get_artifact_for_reference(
                store, session_id, ref_item.ref, now=now
            )
            if isinstance(res, Failure):
                return Failure(res.failure())
            artifacts[ref_item.ref] = res.unwrap()

    effective_dc = max_data_class(
        base_data_class, *(art.data_class for art in artifacts.values())
    )
    sorted_ids = tuple(
        sorted({art.artifact_id or art.record_id for art in artifacts.values()})
    )

    return Success(
        PreflightReferenceScan(
            has_references=True,
            referenced_artifacts=artifacts,
            effective_data_class=effective_dc,
            referenced_ids=sorted_ids,
        )
    )


def resolve_arguments(
    arguments: dict[str, Any],
    artifacts: dict[str, ObservationArtifact],
) -> Result[dict[str, Any], tuple[str, str]]:
    """Substitute all reference objects in arguments with their concrete values."""

    def _resolve_val(val: object) -> Result[Any, tuple[str, str]]:  # noqa: PLR0911
        if is_artifact_reference(val):
            ref_name = str(val["$ref"]).strip()
            artifact = artifacts.get(ref_name)
            if artifact is None:
                return Failure(
                    ("ARTIFACT_NOT_FOUND", f"Artifact '{ref_name}' missing from resolution map")
                )
            pointer_path = str(val.get("path", "")).strip()
            if not pointer_path:
                return Success(artifact.content)

            parse_res = parse_json_pointer(pointer_path)
            if isinstance(parse_res, Failure):
                return Failure(("INVALID_POINTER", parse_res.failure()))

            trav_res = traverse_json_pointer(artifact.content, parse_res.unwrap())
            if isinstance(trav_res, Failure):
                return Failure(trav_res.failure())
            return Success(trav_res.unwrap())

        if isinstance(val, dict):
            resolved_dict: dict[str, Any] = {}
            for k, v in val.items():
                sub_res = _resolve_val(v)
                if isinstance(sub_res, Failure):
                    return sub_res
                resolved_dict[k] = sub_res.unwrap()
            return Success(resolved_dict)

        if isinstance(val, (list, tuple)):
            resolved_list: list[Any] = []
            for item in val:
                sub_res = _resolve_val(item)
                if isinstance(sub_res, Failure):
                    return sub_res
                resolved_list.append(sub_res.unwrap())
            return Success(tuple(resolved_list) if isinstance(val, tuple) else resolved_list)

        return Success(val)

    top_res = _resolve_val(arguments)
    if isinstance(top_res, Failure):
        return Failure(top_res.failure())
    return Success(dict(top_res.unwrap()))


def make_provenance_references(
    artifacts: Sequence[ObservationArtifact],
) -> tuple[ProvenanceReference, ...]:
    """Construct provenance references linking an execution result to its source observation artifacts."""
    return tuple(
        ProvenanceReference(
            source_id=art.artifact_id or art.record_id,
            source_type="observation_artifact",
            source_version="1.0.0",
            relation="input_argument",
            producer=art.producer,
        )
        for art in artifacts
    )
