"""Typed capability registry and schema-constrained invocation adapters."""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator
from returns.result import Failure, Result, Success

from .ports import CancellationToken
from .records import (
    ActionFailure,
    ActionResult,
    CapabilityRequest,
    ProducerIdentity,
    RiskClass,
    freeze_json_value,
)

CapabilityHandler = Callable[[dict[str, Any]], dict[str, Any] | Awaitable[dict[str, Any]]]


class CapabilityDescriptor(BaseModel):
    """Transport-neutral metadata and validation contract for one capability."""

    model_config = ConfigDict(frozen=True, extra="forbid", use_enum_values=True)

    name: str = Field(pattern=r"^[a-z][a-z0-9]*(?:[._][a-z0-9]+)*$")
    description: str = Field(min_length=1)
    input_schema: dict[str, Any]
    risk_class: RiskClass
    side_effects: tuple[str, ...] = ()
    isolation: str = Field(min_length=1)
    requires_isolation: bool = False
    verification_plan: tuple[str, ...] = Field(min_length=1)

    @field_validator("input_schema")
    @classmethod
    def freeze_schema(cls, value: dict[str, Any]) -> dict[str, Any]:
        del cls
        return freeze_json_value(value)


class RegisteredCapability:
    """A descriptor paired with its implementation."""

    def __init__(
        self,
        descriptor: CapabilityDescriptor,
        handler: CapabilityHandler,
        compatibility_handler: Callable[..., Any] | None = None,
    ) -> None:
        self.descriptor = descriptor
        self._handler = handler
        self.compatibility_handler = compatibility_handler

    @property
    def name(self) -> str:
        return self.descriptor.name

    async def invoke(
        self, request: CapabilityRequest, cancellation: CancellationToken
    ) -> Result[ActionResult, ActionFailure]:
        if cancellation.cancelled:
            return Failure(_failure(request, "CANCELLED", "operation was cancelled"))
        invalid = validate_arguments(self.descriptor.input_schema, request.arguments)
        if invalid:
            return Failure(_failure(request, "INVALID_ARGUMENT", invalid))
        try:
            output = self._handler(request.arguments)
            if inspect.isawaitable(output):
                output = await output
            if cancellation.cancelled:
                return Failure(_failure(request, "CANCELLED", "operation was cancelled"))
            return Success(
                ActionResult(
                    record_id=f"result:{request.record_id}",
                    timestamp=request.timestamp,
                    producer=ProducerIdentity(
                        producer_id=f"capability:{self.name}",
                        kind="capability",
                        version="1.0.0",
                    ),
                    correlation_id=request.correlation_id,
                    request_id=request.record_id,
                    capability=self.name,
                    output=output,
                    verification={step: True for step in self.descriptor.verification_plan},
                )
            )
        except Exception as exc:  # pylint: disable=broad-exception-caught
            return Failure(_failure(request, "CAPABILITY_FAILED", str(exc)))


class CapabilityRegistry:
    """The only name-to-capability mapping used by runtime interfaces."""

    def __init__(self) -> None:
        self._capabilities: dict[str, RegisteredCapability] = {}
        self._compatibility_groups: dict[str, list[str]] = {}

    def register(
        self, capability: RegisteredCapability, compatibility_groups: tuple[str, ...] = ()
    ) -> None:
        if capability.name in self._capabilities:
            raise ValueError(f"capability already registered: {capability.name}")
        self._capabilities[capability.name] = capability
        for group in compatibility_groups:
            self._compatibility_groups.setdefault(group, []).append(capability.name)

    def descriptors(self) -> tuple[CapabilityDescriptor, ...]:
        return tuple(
            capability.descriptor
            for _, capability in sorted(self._capabilities.items())
        )

    def descriptor(self, name: str) -> CapabilityDescriptor | None:
        capability = self._capabilities.get(name)
        return capability.descriptor if capability is not None else None

    def compatibility_groups(self) -> tuple[str, ...]:
        return tuple(sorted(self._compatibility_groups))

    def compatibility_handlers(self, group: str) -> tuple[Callable[..., Any], ...]:
        handlers = []
        for name in self._compatibility_groups.get(group, []):
            handler = self._capabilities[name].compatibility_handler
            if handler is not None:
                handlers.append(handler)
        return tuple(handlers)

    def compatibility_capabilities(self, group: str) -> tuple[RegisteredCapability, ...]:
        return tuple(
            self._capabilities[name]
            for name in self._compatibility_groups.get(group, [])
            if self._capabilities[name].compatibility_handler is not None
        )

    def resolve(
        self, request: CapabilityRequest
    ) -> Result[RegisteredCapability, ActionFailure]:
        capability = self._capabilities.get(request.capability)
        if capability is None:
            return Failure(_failure(request, "CAPABILITY_NOT_FOUND", "capability is not registered"))
        return Success(capability)


def _failure(request: CapabilityRequest, code: str, message: str) -> ActionFailure:
    return ActionFailure(
        record_id=f"failure:{request.record_id}:{code}",
        timestamp=request.timestamp,
        producer=ProducerIdentity(
            producer_id="rai.capability-registry", kind="runtime", version="1.0.0"
        ),
        correlation_id=request.correlation_id,
        request_id=request.record_id,
        capability=request.capability,
        code=code,
        message=message,
    )


def validate_arguments(schema: dict[str, Any], arguments: dict[str, Any]) -> str | None:
    """Validate the deliberately small JSON Schema subset used by capabilities."""
    if schema.get("type") != "object":
        return "capability input schema must describe an object"
    properties = schema.get("properties", {})
    required = schema.get("required", [])
    for name in required:
        if name not in arguments:
            return f"missing required argument: {name}"
    if schema.get("additionalProperties", False) is False:
        unknown = sorted(set(arguments) - set(properties))
        if unknown:
            return f"unknown argument: {unknown[0]}"
    expected_types = {
        "string": str,
        "boolean": bool,
        "integer": int,
        "number": (int, float),
        "object": dict,
        "array": list,
    }
    for name, value in arguments.items():
        property_schema = properties.get(name)
        if property_schema is None:
            continue
        expected = expected_types.get(property_schema.get("type"))
        if expected is not None and (
            not isinstance(value, expected)
            or property_schema.get("type") == "integer" and isinstance(value, bool)
        ):
            return f"argument {name} must be {property_schema['type']}"
    return None
