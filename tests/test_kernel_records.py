"""Conformance tests for the language-neutral Stage 1 records."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from pydantic import ValidationError

from rai.kernel.records import (
    ActionFailure,
    ActionResult,
    CapabilityRequest,
    Claim,
    ContextManifest,
    ContextManifestItem,
    ContextPackage,
    DataClass,
    DeviceDescriptor,
    Episode,
    InferenceBudget,
    KernelRecord,
    MediaReference,
    Observation,
    PolicyDecision,
    PolicyOutcome,
    ProducerIdentity,
    ProvenanceReference,
    RiskClass,
    Task,
    UsageRecord,
    parse_record,
)
from rai.kernel.schemas import kernel_json_schema

FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "kernel" / "v1"
NOW = datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
PRODUCER = ProducerIdentity(
    producer_id="synthetic-runtime", kind="test", version="1.0.0"
)
SOURCE = ProvenanceReference(
    source_id="obs-1",
    source_type="observation",
    source_version="1.0.0",
    relation="derived-from",
    producer=PRODUCER,
)


def _metadata() -> dict[str, object]:
    return {"timestamp": NOW, "producer": PRODUCER, "correlation_id": "corr-1"}


def _all_record_examples() -> tuple[KernelRecord, ...]:  # noqa: PLR0915
    metadata = _metadata()
    budget = InferenceBudget(
        **metadata,
        max_input_tokens=100,
        max_output_tokens=50,
        max_agent_turns=1,
        max_tool_calls=1,
        max_images=0,
        max_audio_seconds=0,
        max_latency_seconds=5,
        max_provider_cost=0,
        max_ram_bytes=1024,
        max_vram_bytes=0,
        cancellation_deadline=NOW + timedelta(seconds=5),
    )
    manifest = ContextManifest(
        **metadata,
        destination="synthetic-backend",
        items=(
            ContextManifestItem(
                source_id="obs-1", source_type="observation", data_class=DataClass.LOCAL
            ),
        ),
    )
    return (
        DeviceDescriptor(**metadata, display_name="test", platform="linux"),
        MediaReference(
            **metadata,
            uri="rai-media://fixture",
            media_type="text/plain",
            sha256="0" * 64,
            size_bytes=0,
            data_class=DataClass.LOCAL,
        ),
        Observation(
            **metadata, kind="person_present", payload={"present": True}, data_class=DataClass.LOCAL
        ),
        Episode(
            **metadata,
            started_at=NOW,
            ended_at=NOW,
            observation_ids=("obs-1",),
            provenance=(SOURCE,),
        ),
        Claim(
            **metadata,
            statement="A person is present",
            confidence=1,
            epistemic_status="observed",
            data_class=DataClass.LOCAL,
            provenance=(SOURCE,),
        ),
        Task(**metadata, objective="notify the user", provenance=(SOURCE,)),
        manifest,
        ContextPackage(
            **metadata,
            task_id="task-1",
            manifest=manifest,
            content={"observations": []},
            provenance=(SOURCE,),
        ),
        budget,
        CapabilityRequest(
            **metadata,
            actor=PRODUCER,
            capability="test.echo",
            arguments={"text": "hello"},
            data_class=DataClass.LOCAL,
            target_resource="memory://echo",
            requested_side_effects=(),
            isolation="in-process",
            budget=budget,
            verification_plan=("compare-output",),
        ),
        ActionResult(
            **metadata,
            request_id="request-1",
            capability="test.echo",
            output={"text": "hello"},
            verification={"matched": True},
            provenance=(SOURCE,),
        ),
        ActionFailure(
            **metadata,
            request_id="request-1",
            capability="test.echo",
            code="CANCELLED",
            message="cancelled",
        ),
        PolicyDecision(
            **metadata,
            request_id="request-1",
            outcome=PolicyOutcome.ALLOW,
            risk_class=RiskClass.LOW,
            policy_version="1.0.0",
            reason_codes=("LOW_RISK_LOCAL",),
            actor=PRODUCER,
            data_class=DataClass.LOCAL,
            target_resource="memory://echo",
            requested_side_effects=(),
            isolation="in-process",
            budget_id=budget.record_id,
            verification_plan=("compare-output",),
        ),
        UsageRecord(
            **metadata,
            task_id="task-1",
            processor="synthetic",
            backend="local",
            model="none",
            provider="local",
            input_tokens=0,
            output_tokens=0,
            tool_calls=1,
            images=0,
            audio_seconds=0,
            latency_seconds=0,
            provider_cost=0,
        ),
    )


def test_every_domain_record_round_trips_through_dispatch_parser() -> None:
    for record in _all_record_examples():
        decoded = parse_record(record.model_dump(mode="json"))
        assert decoded == record
        with pytest.raises(ValidationError):
            record.record_id = "mutated"  # type: ignore[misc]


def test_records_are_deeply_immutable() -> None:
    observation = Observation(
        timestamp=NOW,
        producer=PRODUCER,
        kind="nested",
        payload={"outer": {"items": ["one"]}},
        data_class=DataClass.LOCAL,
    )
    with pytest.raises(TypeError, match="immutable"):
        observation.payload["new"] = True
    with pytest.raises(TypeError, match="immutable"):
        observation.payload["outer"]["new"] = True
    with pytest.raises(AttributeError):
        observation.payload["outer"]["items"].append("two")


def test_positive_language_neutral_fixture() -> None:
    data = json.loads((FIXTURE_ROOT / "observation.valid.json").read_text(encoding="utf-8"))
    parsed = parse_record(data)
    assert isinstance(parsed, Observation)
    assert parsed.payload == {"present": True}


def test_negative_language_neutral_fixtures_are_rejected() -> None:
    cases = json.loads((FIXTURE_ROOT / "observation.invalid.json").read_text(encoding="utf-8"))
    for case in cases:
        with pytest.raises((ValidationError, ValueError), match=".+"):
            parse_record(case["record"])


def test_schema_contains_every_record_and_explicit_discriminator() -> None:
    schema = kernel_json_schema()
    assert schema["x-rai-schema-version"] == "1.0.0"
    assert schema["discriminator"]["propertyName"] == "record_type"
    for record in _all_record_examples():
        assert type(record).__name__ in schema["$defs"]


def test_published_schema_matches_runtime_contract() -> None:
    published = Path("schemas/rai.kernel.v1.schema.json")
    assert json.loads(published.read_text(encoding="utf-8")) == kernel_json_schema()
