"""Cross-project conformance tests for GAIA's canonical NCSI fixtures."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rai.neural.contracts import (
    Concept,
    GenerationRequest,
    NcsiContractError,
    NcsiEvent,
    NeuralObservation,
    validate_event_stream,
)

FIXTURES = Path(__file__).parent / "fixtures" / "ncsi"


def test_accepts_canonical_gaia_fixture() -> None:
    events = json.loads((FIXTURES / "gcas.ncsi.v1.valid.json").read_text(encoding="utf-8"))

    parsed = validate_event_stream(events)

    assert len(parsed) == 4  # noqa: PLR2004
    assert [event.to_wire() for event in parsed] == events


@pytest.mark.parametrize(
    "case",
    json.loads((FIXTURES / "gcas.ncsi.v1.invalid.json").read_text(encoding="utf-8")),
    ids=lambda case: case["name"],
)
def test_rejects_canonical_gaia_invalid_fixture(case: dict[str, object]) -> None:
    with pytest.raises(NcsiContractError):
        NcsiEvent.from_wire(case["event"])


@pytest.mark.parametrize(
    "events",
    [
        [{"schema-version": "gcas.ncsi.v1", "event-type": "TokenDelta", "request-id": "r", "timestamp": 1, "payload": {"token-id": 1, "token-text": "x"}}],
        [
            {"schema-version": "gcas.ncsi.v1", "event-type": "GenerationStarted", "request-id": "r", "timestamp": 1, "payload": {"model-id": "m"}},
            {"schema-version": "gcas.ncsi.v1", "event-type": "GenerationCompleted", "request-id": "r", "timestamp": 2, "payload": {"final-text": "", "token-count": 0}},
            {"schema-version": "gcas.ncsi.v1", "event-type": "TokenDelta", "request-id": "r", "timestamp": 3, "payload": {"token-id": 1, "token-text": "x"}},
        ],
    ],
)
def test_rejects_invalid_lifecycle(events: list[dict[str, object]]) -> None:
    with pytest.raises(NcsiContractError):
        validate_event_stream(events)


def test_bounds_requests_and_observations() -> None:
    with pytest.raises(NcsiContractError):
        GenerationRequest(prompt="x", request_id="r", top_k=65)
    with pytest.raises(NcsiContractError):
        GenerationRequest(prompt="x", request_id="r", layers=(1, 1))
    with pytest.raises(NcsiContractError):
        GenerationRequest(prompt="x", request_id="r" * 257)
    with pytest.raises(NcsiContractError):
        Concept(1, "x", float("nan"))
    with pytest.raises(NcsiContractError):
        NeuralObservation(
            request_id="r",
            forward_pass_id="f",
            model_id="m",
            model_revision="mr",
            tokenizer_revision="tr",
            lens_id="l",
            lens_revision="lr",
            layer=0,
            position=(2, 1),
            concepts=(),
            readout_method="jlens-sparse",
            parameters={},
            timestamp=1,
        )
