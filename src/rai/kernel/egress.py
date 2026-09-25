"""Outbound egress firewall for external model and decision API communications.

Enforces privacy classification boundaries, prevents data leaks of LOCAL, SECRET,
and BLOCKED data to external/remote backends, and mandates HITL approval where required.
"""

from __future__ import annotations

import logging
from typing import Any, Mapping, Protocol

from returns.result import Failure, Result, Success

from rai.kernel.records import (
    ActionFailure,
    DataClass,
    ProducerIdentity,
)

logger = logging.getLogger(__name__)


class EgressManifest(Protocol):
    """Structural manifest contract shared by kernel and assistant contexts."""

    items: tuple[Any, ...]
    approved: bool


EGRESS_PRODUCER = ProducerIdentity(
    producer_id="rai.kernel.egress_firewall",
    kind="security-firewall",
    version="1.0.0",
)

SENSITIVE_FIELD_PATTERNS = (
    "password",
    "secret",
    "token",
    "api_key",
    "apikey",
    "auth",
    "credential",
    "private_key",
)


def make_egress_failure(
    code: str,
    message: str,
    request_id: str = "egress-firewall",
    retryable: bool = False,
    producer: ProducerIdentity | None = None,
) -> ActionFailure:
    return ActionFailure(
        request_id=request_id,
        capability="security.egress",
        code=code,
        message=message,
        retryable=retryable,
        producer=producer or EGRESS_PRODUCER,
    )


class EgressFirewall:
    """Outbound data-loss prevention firewall for external inference backends."""

    def __init__(
        self,
        profile: str = "LOCAL_PREFERRED",
        producer: ProducerIdentity = EGRESS_PRODUCER,
    ) -> None:
        self.profile = profile
        self.producer = producer

    def validate_egress(
        self,
        manifest: EgressManifest,
        destination_is_remote: bool = True,
        request_id: str = "egress-firewall",
        required_source_ids: tuple[str, ...] = (),
    ) -> Result[None, ActionFailure]:
        """Validate whether the given ContextManifest is permitted to exit to destination."""
        if not destination_is_remote:
            return Success(None)

        if self.profile == "LOCAL_ONLY":
            return Failure(
                make_egress_failure(
                    code="EGRESS_LOCAL_ONLY_VIOLATION",
                    message="Egress to remote backends is prohibited under LOCAL_ONLY profile",
                    request_id=request_id,
                    producer=self.producer,
                )
            )

        present_source_ids = {item.source_id for item in manifest.items}
        missing_source_ids = sorted(set(required_source_ids) - present_source_ids)
        if missing_source_ids:
            return Failure(
                make_egress_failure(
                    code="EGRESS_MANIFEST_INCOMPLETE",
                    message=(
                        "Egress blocked: outbound context is missing manifest entries for "
                        + ", ".join(missing_source_ids)
                    ),
                    request_id=request_id,
                    producer=self.producer,
                )
            )

        approved = bool(getattr(manifest, "approved", False))
        for item in manifest.items:
            data_class = item.data_class
            if isinstance(data_class, str):
                data_class = DataClass(data_class)

            # Strict Invariant: SECRET and BLOCKED are never permitted to leave
            if data_class in {DataClass.SECRET, DataClass.BLOCKED}:
                logger.error(
                    "Blocked outbound transmission of source %s with data class %s",
                    item.source_id,
                    data_class,
                )
                return Failure(
                    make_egress_failure(
                        code="EGRESS_DATA_CLASS_FORBIDDEN",
                        message=(
                            f"Egress blocked: source '{item.source_id}' has forbidden "
                            f"data class '{data_class.value}'"
                        ),
                        request_id=request_id,
                        producer=self.producer,
                    )
                )

            # Invariant: LOCAL-only data must not egress to remote backends
            if data_class == DataClass.LOCAL:
                logger.error(
                    "Blocked outbound transmission of LOCAL source %s to remote endpoint",
                    item.source_id,
                )
                return Failure(
                    make_egress_failure(
                        code="EGRESS_LOCAL_DATA_LEAK",
                        message=(
                            f"Egress blocked: source '{item.source_id}' is classified as "
                            "LOCAL and cannot leave the local trust boundary"
                        ),
                        request_id=request_id,
                        producer=self.producer,
                    )
                )

            # Invariant: PRIVATE data requires approval in LOCAL_PREFERRED, HYBRID_APPROVAL,
            # or REMOTE_ALLOWED
            if data_class == DataClass.PRIVATE:
                if not approved:
                    return Failure(
                        make_egress_failure(
                            code="PRIVATE_DATA_EGRESS_REQUIRES_APPROVAL",
                            message=(
                                f"Egress blocked: source '{item.source_id}' is PRIVATE and "
                                "outbound transmission has not been approved by user/broker"
                            ),
                            request_id=request_id,
                            retryable=True,
                            producer=self.producer,
                        )
                    )

        # In HYBRID_APPROVAL profile, even non-private remote transmissions need manifest approval
        if self.profile == "HYBRID_APPROVAL" and not approved:
            return Failure(
                make_egress_failure(
                    code="HYBRID_APPROVAL_REQUIRED",
                    message="HYBRID_APPROVAL profile requires explicit manifest approval before remote egress",
                    request_id=request_id,
                    retryable=True,
                    producer=self.producer,
                )
            )

        return Success(None)

    def sanitize_outbound_payload(
        self,
        payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Strip sensitive credentials and private keys from outbound remote payloads."""
        sanitized: dict[str, Any] = {}
        for key, value in payload.items():
            key_lower = str(key).lower()
            if any(pattern in key_lower for pattern in SENSITIVE_FIELD_PATTERNS):
                sanitized[key] = "[REDACTED_BY_EGRESS_FIREWALL]"
            elif isinstance(value, Mapping):
                sanitized[key] = self.sanitize_outbound_payload(value)
            elif isinstance(value, (list, tuple)):
                sanitized[key] = [
                    self.sanitize_outbound_payload(v) if isinstance(v, Mapping) else v
                    for v in value
                ]
            else:
                sanitized[key] = value
        return sanitized
