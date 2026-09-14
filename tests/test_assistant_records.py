"""Conformance tests for the Rich Assistant domain records."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from rai.assistant.records import (
    ASSISTANT_RECORD_TYPES,
    MAX_PROPOSALS_PER_TURN,
    AssistantCandidate,
    AssistantContextManifest,
    AssistantContextPackage,
    AssistantResponse,
    ConversationTurn,
    InferenceRequest,
    MemoryProposal,
    MemoryRecord,
    parse_assistant_record,
)
from rai.assistant.schemas import assistant_json_schema
from rai.kernel.records import DataClass, ProducerIdentity

FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "assistant" / "v1"


def test_parse_valid_turn_fixture() -> None:
    data = json.loads((FIXTURE_ROOT / "turn.valid.json").read_text(encoding="utf-8"))
    record = parse_assistant_record(data)
    assert isinstance(record, ConversationTurn)
    assert record.role == "user"
    assert record.text.startswith("Zapamiętaj")
    assert record.status == "ACCEPTED"


def test_parse_valid_proposal_fixture() -> None:
    data = json.loads((FIXTURE_ROOT / "proposal.valid.json").read_text(encoding="utf-8"))
    record = parse_assistant_record(data)
    assert isinstance(record, MemoryProposal)
    assert record.topic == "code_examples"
    assert record.content["preference"] == "Guile"


def test_parse_valid_memory_fixture() -> None:
    data = json.loads((FIXTURE_ROOT / "memory.valid.json").read_text(encoding="utf-8"))
    record = parse_assistant_record(data)
    assert isinstance(record, MemoryRecord)
    assert record.topic == "code_examples"
    assert record.content["preference"] == "Guile"
    assert len(record.provenance) == 1
    assert record.provenance[0].relation == "DERIVED_FROM"


def test_parse_valid_manifest_fixture() -> None:
    data = json.loads((FIXTURE_ROOT / "manifest.valid.json").read_text(encoding="utf-8"))
    record = parse_assistant_record(data)
    assert isinstance(record, AssistantContextManifest)
    assert "mem-valid-001" in record.durable_memory_ids
    assert record.ranking_reasons["mem-valid-001"].startswith("topic exact match")


def test_parse_valid_package_fixture() -> None:
    data = json.loads((FIXTURE_ROOT / "package.valid.json").read_text(encoding="utf-8"))
    record = parse_assistant_record(data)
    assert isinstance(record, AssistantContextPackage)
    assert record.manifest.record_type == "assistant_context_manifest"


def test_parse_valid_request_fixture() -> None:
    data = json.loads((FIXTURE_ROOT / "request.valid.json").read_text(encoding="utf-8"))
    record = parse_assistant_record(data)
    assert isinstance(record, InferenceRequest)
    assert record.strategy == "DIRECT"


def test_parse_valid_response_fixture() -> None:
    data = json.loads((FIXTURE_ROOT / "response.valid.json").read_text(encoding="utf-8"))
    record = parse_assistant_record(data)
    assert isinstance(record, AssistantResponse)
    assert record.status == "COMPLETED"
    assert record.admitted_memory_ids == ("mem-valid-001",)


def test_parse_failed_response_fixture() -> None:
    data = json.loads((FIXTURE_ROOT / "response.failed.json").read_text(encoding="utf-8"))
    record = parse_assistant_record(data)
    assert isinstance(record, AssistantResponse)
    assert record.status == "FAILED"
    assert "timed out" in (record.error_message or "")


def test_turn_rejects_secret_data_class() -> None:
    data = json.loads((FIXTURE_ROOT / "turn.invalid_secret.json").read_text(encoding="utf-8"))
    with pytest.raises(ValidationError, match="cannot have SECRET data class"):
        parse_assistant_record(data)


def test_turn_rejects_malformed_fixture() -> None:
    data = json.loads((FIXTURE_ROOT / "turn.malformed.json").read_text(encoding="utf-8"))
    with pytest.raises(ValidationError):
        parse_assistant_record(data)


def test_proposal_rejects_over_budget_content() -> None:
    data = json.loads((FIXTURE_ROOT / "proposal.over_budget.json").read_text(encoding="utf-8"))
    with pytest.raises(ValidationError, match="exceeds max size"):
        parse_assistant_record(data)


def test_rejects_unsupported_schema_version() -> None:
    data = json.loads((FIXTURE_ROOT / "schema_version.unsupported.json").read_text(encoding="utf-8"))
    with pytest.raises(ValidationError, match="pattern|unsupported schema major"):
        parse_assistant_record(data)


def test_records_are_deeply_immutable() -> None:
    turn = ConversationTurn(
        session_id="session-1",
        role="user",
        text="Hello",
        producer=ProducerIdentity(producer_id="test", kind="test", version="1.0.0"),
        metadata={"extra": {"key": "val"}},
    )
    with pytest.raises(ValidationError):
        turn.text = "mutated"  # type: ignore[misc]

    with pytest.raises(TypeError, match="immutable"):
        turn.metadata["new"] = True

    with pytest.raises(TypeError, match="immutable"):
        turn.metadata["extra"]["new"] = True


def test_candidate_proposal_quota() -> None:
    producer = ProducerIdentity(producer_id="test", kind="test", version="1.0.0")
    proposals = tuple(
        MemoryProposal(
            source_turn_id="turn-1",
            topic=f"topic_{i}",
            content={"val": i},
            producer=producer,
        )
        for i in range(MAX_PROPOSALS_PER_TURN + 1)
    )
    with pytest.raises(ValidationError, match="exceeding max"):
        AssistantCandidate(text="output", proposals=proposals)


def test_turn_text_length_limit() -> None:
    huge_text = "x" * (16 * 1024 + 1)
    with pytest.raises(ValidationError):
        ConversationTurn(
            session_id="session-1",
            role="user",
            text=huge_text,
            producer=ProducerIdentity(producer_id="test", kind="test", version="1.0.0"),
        )


def test_published_schema_matches_runtime_contract() -> None:
    schema = assistant_json_schema()
    assert schema["x-rai-schema-version"] == "1.0.0"
    assert schema["discriminator"]["propertyName"] == "record_type"
    for model_name, model_cls in ASSISTANT_RECORD_TYPES.items():
        assert model_cls.__name__ in schema["$defs"]

    published = Path("schemas/rai.assistant.v1.schema.json")
    assert json.loads(published.read_text(encoding="utf-8")) == schema
