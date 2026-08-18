"""Transport-independent contracts for the ``gcas.ncsi.v1`` protocol."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Sequence

NCSI_SCHEMA_VERSION = "gcas.ncsi.v1"
MAX_CONCEPTS = 64
MAX_DISPLAY_TEXT_LENGTH = 512
MAX_PARAMETERS = 32
MAX_PROMPT_LENGTH = 131_072
MAX_NEW_TOKENS = 4096
TOKEN_SPAN_LENGTH = 2


class NcsiErrorCode(str, Enum):
    """Stable failure codes emitted by the sidecar."""

    CANCELLED = "CANCELLED"
    TIMEOUT = "TIMEOUT"
    INVALID_REQUEST = "INVALID_REQUEST"
    MODEL_NOT_FOUND = "MODEL_NOT_FOUND"
    MODEL_LOAD_FAILED = "MODEL_LOAD_FAILED"
    MODEL_INCOMPATIBLE = "MODEL_INCOMPATIBLE"
    LENS_NOT_FOUND = "LENS_NOT_FOUND"
    LENS_INCOMPATIBLE = "LENS_INCOMPATIBLE"
    ACCELERATOR_OOM = "ACCELERATOR_OOM"
    CONCURRENCY_LIMIT = "CONCURRENCY_LIMIT"
    INFERENCE_FAILED = "INFERENCE_FAILED"
    INTERNAL_ERROR = "INTERNAL_ERROR"


class NcsiContractError(ValueError):
    """A wire value violates the NCSI schema."""


class NcsiRuntimeError(RuntimeError):
    """A typed operational failure which is safe to expose to a client."""

    def __init__(self, code: NcsiErrorCode, message: str) -> None:
        super().__init__(message)
        self.code = code


def _non_empty(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise NcsiContractError(f"{field_name} must be a non-empty string")
    return value


def _number(value: object, field_name: str) -> float | int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise NcsiContractError(f"{field_name} must be numeric")
    if not math.isfinite(value):
        raise NcsiContractError(f"{field_name} must be finite")
    return value


def _non_negative_int(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise NcsiContractError(f"{field_name} must be a non-negative integer")
    return value


def _score(value: object, field_name: str) -> float:
    number = float(_number(value, field_name))
    if not 0.0 <= number <= 1.0:
        raise NcsiContractError(f"{field_name} must be in [0, 1]")
    return number


@dataclass(frozen=True)
class Concept:
    """One bounded J-lens readout concept."""

    token_id: int | str
    display_text: str
    score: float

    def __post_init__(self) -> None:
        if isinstance(self.token_id, bool) or not isinstance(self.token_id, (int, str)):
            raise NcsiContractError("token-id must be an integer or string")
        if not isinstance(self.display_text, str):
            raise NcsiContractError("display-text must be a string")
        if len(self.display_text) > MAX_DISPLAY_TEXT_LENGTH:
            raise NcsiContractError("display-text exceeds the NCSI limit")
        object.__setattr__(self, "score", _score(self.score, "score"))

    def to_wire(self) -> dict[str, object]:
        return {
            "token-id": self.token_id,
            "display-text": self.display_text,
            "score": self.score,
        }

    @classmethod
    def from_wire(cls, value: object) -> "Concept":
        data = _mapping(value, "concept")
        return cls(data.get("token-id"), data.get("display-text"), data.get("score"))


Scalar = str | int | float | bool | None


def _mapping(value: object, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise NcsiContractError(f"{field_name} must be an object with string keys")
    return value


def _parameters(value: object) -> dict[str, Scalar]:
    data = _mapping(value, "parameters")
    if len(data) > MAX_PARAMETERS:
        raise NcsiContractError("parameters exceeds the NCSI limit")
    for item in data.values():
        if item is not None and not isinstance(item, (str, int, float, bool)):
            raise NcsiContractError("parameter values must be JSON scalars")
        if isinstance(item, float) and not math.isfinite(item):
            raise NcsiContractError("parameter values must be finite")
    return dict(data)


@dataclass(frozen=True)
class NeuralObservation:
    """Compact neural readout; raw activations never cross this record."""

    request_id: str
    forward_pass_id: str
    model_id: str
    model_revision: str
    tokenizer_revision: str
    lens_id: str
    lens_revision: str
    layer: int
    position: int | tuple[int, int]
    concepts: tuple[Concept, ...]
    readout_method: str
    parameters: Mapping[str, Scalar]
    timestamp: float
    reconstruction_error: float = 0.0
    schema_version: str = NCSI_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != NCSI_SCHEMA_VERSION:
            raise NcsiContractError("unsupported observation schema-version")
        for value, name in (
            (self.request_id, "request-id"),
            (self.forward_pass_id, "forward-pass-id"),
            (self.model_id, "model-id"),
            (self.model_revision, "model-revision"),
            (self.tokenizer_revision, "tokenizer-revision"),
            (self.lens_id, "lens-id"),
            (self.lens_revision, "lens-revision"),
            (self.readout_method, "readout-method"),
        ):
            _non_empty(value, name)
        _non_negative_int(self.layer, "layer")
        if isinstance(self.position, tuple):
            if len(self.position) != TOKEN_SPAN_LENGTH:
                raise NcsiContractError("position span must have two elements")
            start = _non_negative_int(self.position[0], "position")
            end = _non_negative_int(self.position[1], "position")
            if start > end:
                raise NcsiContractError("position span must be ordered")
        else:
            _non_negative_int(self.position, "position")
        if len(self.concepts) > MAX_CONCEPTS or not all(
            isinstance(concept, Concept) for concept in self.concepts
        ):
            raise NcsiContractError("concepts must be a bounded concept list")
        object.__setattr__(self, "parameters", _parameters(self.parameters))
        object.__setattr__(
            self, "reconstruction_error", _score(self.reconstruction_error, "reconstruction-error")
        )
        object.__setattr__(self, "timestamp", float(_number(self.timestamp, "timestamp")))

    def to_wire(self) -> dict[str, object]:
        position: int | list[int] = (
            list(self.position) if isinstance(self.position, tuple) else self.position
        )
        return {
            "schema-version": self.schema_version,
            "request-id": self.request_id,
            "forward-pass-id": self.forward_pass_id,
            "model-id": self.model_id,
            "model-revision": self.model_revision,
            "tokenizer-revision": self.tokenizer_revision,
            "lens-id": self.lens_id,
            "lens-revision": self.lens_revision,
            "layer": self.layer,
            "position": position,
            "concepts": [concept.to_wire() for concept in self.concepts],
            "readout-method": self.readout_method,
            "parameters": dict(self.parameters),
            "reconstruction-error": self.reconstruction_error,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_wire(cls, value: object) -> "NeuralObservation":
        data = _mapping(value, "observation")
        raw_position = data.get("position")
        position: object = tuple(raw_position) if isinstance(raw_position, list) else raw_position
        raw_concepts = data.get("concepts")
        if not isinstance(raw_concepts, Sequence) or isinstance(raw_concepts, (str, bytes)):
            raise NcsiContractError("concepts must be an array")
        return cls(
            schema_version=data.get("schema-version"),
            request_id=data.get("request-id"),
            forward_pass_id=data.get("forward-pass-id"),
            model_id=data.get("model-id"),
            model_revision=data.get("model-revision"),
            tokenizer_revision=data.get("tokenizer-revision"),
            lens_id=data.get("lens-id"),
            lens_revision=data.get("lens-revision"),
            layer=data.get("layer"),
            position=position,
            concepts=tuple(Concept.from_wire(item) for item in raw_concepts),
            readout_method=data.get("readout-method"),
            parameters=data.get("parameters"),
            reconstruction_error=data.get("reconstruction-error", 0.0),
            timestamp=data.get("timestamp"),
        )


class EventType(str, Enum):
    GENERATION_STARTED = "GenerationStarted"
    TOKEN_DELTA = "TokenDelta"
    NEURAL_STATE_OBSERVED = "NeuralStateObserved"
    GENERATION_COMPLETED = "GenerationCompleted"
    GENERATION_FAILED = "GenerationFailed"


@dataclass(frozen=True)
class NcsiEvent:
    """One event in an NCSI generation lifecycle."""

    event_type: EventType
    request_id: str
    timestamp: float
    payload: Mapping[str, object]
    schema_version: str = NCSI_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != NCSI_SCHEMA_VERSION:
            raise NcsiContractError("unsupported event schema-version")
        _non_empty(self.request_id, "request-id")
        object.__setattr__(self, "timestamp", float(_number(self.timestamp, "timestamp")))
        data = _mapping(self.payload, "payload")
        _validate_payload(self.event_type, self.request_id, data)
        object.__setattr__(self, "payload", dict(data))

    def to_wire(self) -> dict[str, object]:
        return {
            "schema-version": self.schema_version,
            "event-type": self.event_type.value,
            "request-id": self.request_id,
            "timestamp": self.timestamp,
            "payload": dict(self.payload),
        }

    @classmethod
    def from_wire(cls, value: object) -> "NcsiEvent":
        data = _mapping(value, "event")
        try:
            event_type = EventType(data.get("event-type"))
        except (TypeError, ValueError) as exc:
            raise NcsiContractError("unknown event-type") from exc
        return cls(
            schema_version=data.get("schema-version"),
            event_type=event_type,
            request_id=data.get("request-id"),
            timestamp=data.get("timestamp"),
            payload=data.get("payload"),
        )


def _validate_payload(event_type: EventType, request_id: str, payload: Mapping[str, object]) -> None:
    if event_type is EventType.GENERATION_STARTED:
        _non_empty(payload.get("model-id"), "model-id")
    elif event_type is EventType.TOKEN_DELTA:
        token_id = payload.get("token-id")
        if isinstance(token_id, bool) or not isinstance(token_id, (int, str)):
            raise NcsiContractError("token-id must be an integer or string")
        if not isinstance(payload.get("token-text"), str):
            raise NcsiContractError("token-text must be a string")
    elif event_type is EventType.NEURAL_STATE_OBSERVED:
        observation = NeuralObservation.from_wire(payload)
        if observation.request_id != request_id:
            raise NcsiContractError("envelope and observation request-id differ")
    elif event_type is EventType.GENERATION_COMPLETED:
        if not isinstance(payload.get("final-text"), str):
            raise NcsiContractError("final-text must be a string")
        if "token-count" in payload:
            _non_negative_int(payload["token-count"], "token-count")
    elif event_type is EventType.GENERATION_FAILED:
        _non_empty(payload.get("error-code"), "error-code")
        _non_empty(payload.get("error-message"), "error-message")


@dataclass(frozen=True)
class GenerationRequest:
    """Bounded request accepted by the neural sidecar."""

    prompt: str
    request_id: str
    model_id: str | None = None
    lens_id: str | None = None
    max_new_tokens: int = 128
    timeout_seconds: float = 120.0
    top_k: int = 8
    layers: tuple[int, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        _non_empty(self.request_id, "request-id")
        if not isinstance(self.prompt, str) or not self.prompt:
            raise NcsiContractError("prompt must be a non-empty string")
        if len(self.prompt) > MAX_PROMPT_LENGTH:
            raise NcsiContractError("prompt exceeds the sidecar limit")
        if self.model_id is not None:
            _non_empty(self.model_id, "model-id")
        if self.lens_id is not None:
            _non_empty(self.lens_id, "lens-id")
        max_tokens = _non_negative_int(self.max_new_tokens, "max-new-tokens")
        if max_tokens < 1 or max_tokens > MAX_NEW_TOKENS:
            raise NcsiContractError("max-new-tokens is outside the supported range")
        timeout = float(_number(self.timeout_seconds, "timeout-seconds"))
        if timeout <= 0:
            raise NcsiContractError("timeout-seconds must be positive")
        if not 1 <= _non_negative_int(self.top_k, "top-k") <= MAX_CONCEPTS:
            raise NcsiContractError("top-k is outside the supported range")
        if not all(_non_negative_int(layer, "layers") >= 0 for layer in self.layers):
            raise NcsiContractError("layers must contain non-negative integers")

    @classmethod
    def from_wire(cls, value: object) -> "GenerationRequest":
        data = _mapping(value, "generation request")
        raw_layers = data.get("layers", ())
        if not isinstance(raw_layers, (list, tuple)):
            raise NcsiContractError("layers must be an array")
        return cls(
            prompt=data.get("prompt"),
            request_id=data.get("request-id"),
            model_id=data.get("model-id"),
            lens_id=data.get("lens-id"),
            max_new_tokens=data.get("max-new-tokens", 128),
            timeout_seconds=data.get("timeout-seconds", 120.0),
            top_k=data.get("top-k", 8),
            layers=tuple(raw_layers),
        )


def validate_event_stream(events: Sequence[Mapping[str, object]]) -> tuple[NcsiEvent, ...]:
    """Validate schema and exactly-once lifecycle for one or more requests."""
    states: dict[str, str] = {}
    parsed: list[NcsiEvent] = []
    for raw_event in events:
        event = NcsiEvent.from_wire(raw_event)
        state = states.get(event.request_id)
        if event.event_type is EventType.GENERATION_STARTED:
            if state is not None:
                raise NcsiContractError("request already started or terminated")
            states[event.request_id] = "active"
        elif state != "active":
            raise NcsiContractError("event requires an active request")
        elif event.event_type in (EventType.GENERATION_COMPLETED, EventType.GENERATION_FAILED):
            states[event.request_id] = "terminal"
        parsed.append(event)
    return tuple(parsed)
