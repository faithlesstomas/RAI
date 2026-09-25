"""Unit tests for InferenceBudgetGovernor and EgressFirewall."""

from datetime import datetime, timedelta, timezone

import pytest
from returns.result import Failure, Success

from rai.inference.governor import GovernorConfig, InferenceBudgetGovernor
from rai.kernel.egress import EgressFirewall
from rai.kernel.records import (
    ContextManifest,
    ContextManifestItem,
    DataClass,
    InferenceBudget,
    ProducerIdentity,
)

TEST_PRODUCER = ProducerIdentity(
    producer_id="test.governor", kind="test", version="1.0.0"
)


def _make_budget(
    cancellation_offset_sec: int = 60,
    max_agent_turns: int = 3,
    max_tool_calls: int = 5,
) -> InferenceBudget:
    return InferenceBudget(
        producer=TEST_PRODUCER,
        max_input_tokens=5000,
        max_output_tokens=1000,
        max_agent_turns=max_agent_turns,
        max_tool_calls=max_tool_calls,
        max_images=1,
        max_audio_seconds=0.0,
        max_latency_seconds=30.0,
        max_provider_cost=1.0,
        max_ram_bytes=4 * 1024**3,
        max_vram_bytes=2 * 1024**3,
        cancellation_deadline=datetime.now(timezone.utc)
        + timedelta(seconds=cancellation_offset_sec),
    )


# --- EgressFirewall Tests ---


def test_egress_firewall_allows_local_destination() -> None:
    firewall = EgressFirewall(profile="LOCAL_ONLY")
    manifest = ContextManifest(
        producer=TEST_PRODUCER,
        destination="local-daemon",
        items=(
            ContextManifestItem(
                source_id="local_doc",
                source_type="file",
                data_class=DataClass.LOCAL,
            ),
        ),
        approved=False,
    )
    result = firewall.validate_egress(manifest, destination_is_remote=False)
    assert isinstance(result, Success)


def test_egress_firewall_blocks_remote_in_local_only() -> None:
    firewall = EgressFirewall(profile="LOCAL_ONLY")
    manifest = ContextManifest(
        producer=TEST_PRODUCER,
        destination="https://generativelanguage.googleapis.com",
        items=(),
        approved=True,
    )
    result = firewall.validate_egress(manifest, destination_is_remote=True)
    assert isinstance(result, Failure)
    assert result.failure().code == "EGRESS_LOCAL_ONLY_VIOLATION"


def test_egress_firewall_blocks_secret_and_blocked_data() -> None:
    firewall = EgressFirewall(profile="REMOTE_ALLOWED")
    for bad_dc in (DataClass.SECRET, DataClass.BLOCKED):
        manifest = ContextManifest(
            producer=TEST_PRODUCER,
            destination="https://generativelanguage.googleapis.com",
            items=(
                ContextManifestItem(
                    source_id="secret_credentials",
                    source_type="env",
                    data_class=bad_dc,
                ),
            ),
            approved=True,
        )
        result = firewall.validate_egress(manifest, destination_is_remote=True)
        assert isinstance(result, Failure)
        assert result.failure().code == "EGRESS_DATA_CLASS_FORBIDDEN"


def test_egress_firewall_blocks_local_class_from_remote_egress() -> None:
    firewall = EgressFirewall(profile="REMOTE_ALLOWED")
    manifest = ContextManifest(
        producer=TEST_PRODUCER,
        destination="https://generativelanguage.googleapis.com",
        items=(
            ContextManifestItem(
                source_id="local_file_content",
                source_type="file",
                data_class=DataClass.LOCAL,
            ),
        ),
        approved=True,
    )
    result = firewall.validate_egress(manifest, destination_is_remote=True)
    assert isinstance(result, Failure)
    assert result.failure().code == "EGRESS_LOCAL_DATA_LEAK"


def test_egress_firewall_requires_approval_for_private_data() -> None:
    firewall = EgressFirewall(profile="LOCAL_PREFERRED")
    manifest_unapproved = ContextManifest(
        producer=TEST_PRODUCER,
        destination="https://generativelanguage.googleapis.com",
        items=(
            ContextManifestItem(
                source_id="user_preference",
                source_type="memory",
                data_class=DataClass.PRIVATE,
            ),
        ),
        approved=False,
    )
    res = firewall.validate_egress(manifest_unapproved, destination_is_remote=True)
    assert isinstance(res, Failure)
    assert res.failure().code == "PRIVATE_DATA_EGRESS_REQUIRES_APPROVAL"

    manifest_approved = ContextManifest(
        producer=TEST_PRODUCER,
        destination="https://generativelanguage.googleapis.com",
        items=(
            ContextManifestItem(
                source_id="user_preference",
                source_type="memory",
                data_class=DataClass.PRIVATE,
            ),
        ),
        approved=True,
    )
    res2 = firewall.validate_egress(manifest_approved, destination_is_remote=True)
    assert isinstance(res2, Success)


def test_egress_firewall_sanitizes_sensitive_keys() -> None:
    firewall = EgressFirewall()
    payload = {
        "user_query": "Jak napisać skrypt?",
        "api_key": "secret-12345",
        "nested": {
            "auth_token": "bearer-abc",
            "safe_field": 42,
        },
        "list_items": [{"password": "pwd", "name": "admin"}],
    }
    sanitized = firewall.sanitize_outbound_payload(payload)
    assert sanitized["user_query"] == "Jak napisać skrypt?"
    assert sanitized["api_key"] == "[REDACTED_BY_EGRESS_FIREWALL]"
    assert sanitized["nested"]["auth_token"] == "[REDACTED_BY_EGRESS_FIREWALL]"
    assert sanitized["nested"]["safe_field"] == 42
    assert sanitized["list_items"][0]["password"] == "[REDACTED_BY_EGRESS_FIREWALL]"
    assert sanitized["list_items"][0]["name"] == "admin"


# --- InferenceBudgetGovernor Tests ---


def test_governor_enforces_deadline() -> None:
    governor = InferenceBudgetGovernor()
    expired_budget = _make_budget(cancellation_offset_sec=-10)  # already expired
    result = governor.check_request(expired_budget)
    assert isinstance(result, Failure)
    assert result.failure().code == "DEADLINE_EXCEEDED"


def test_governor_invariant_background_remote_tokens_zero() -> None:
    # Strict invariant: background_remote_tokens = 0
    governor = InferenceBudgetGovernor(GovernorConfig(background_remote_tokens=0))
    budget = _make_budget()

    # Foreground remote is permitted
    fg_res = governor.check_request(
        budget, is_background=False, is_remote=True, estimated_tokens=100
    )
    assert isinstance(fg_res, Success)

    # Background remote is forbidden
    bg_res = governor.check_request(
        budget, is_background=True, is_remote=True, estimated_tokens=100
    )
    assert isinstance(bg_res, Failure)
    assert bg_res.failure().code == "BACKGROUND_REMOTE_FORBIDDEN"

    # Background local is permitted
    bg_local_res = governor.check_request(
        budget, is_background=True, is_remote=False, estimated_tokens=100
    )
    assert isinstance(bg_local_res, Success)


def test_governor_enforces_per_task_turns_and_tool_calls() -> None:
    governor = InferenceBudgetGovernor()
    budget = _make_budget(max_agent_turns=2, max_tool_calls=3)

    # Exceed turns (3 > 2)
    res_turns = governor.check_request(budget, current_turns=3)
    assert isinstance(res_turns, Failure)
    assert res_turns.failure().code == "TASK_MAX_TURNS_EXCEEDED"

    # Exceed tools (4 > 3)
    res_tools = governor.check_request(budget, current_tool_calls=4)
    assert isinstance(res_tools, Failure)
    assert res_tools.failure().code == "TASK_MAX_TOOL_CALLS_EXCEEDED"


def test_governor_tracks_and_enforces_daily_limits() -> None:
    config = GovernorConfig(daily_remote_token_limit=1000, daily_cost_limit_usd=5.0)
    governor = InferenceBudgetGovernor(config=config)
    budget = _make_budget()

    # Record 900 remote tokens
    governor.record_usage(
        request_id="req-1",
        tokens_in=500,
        tokens_out=400,
        cost_usd=2.0,
        is_remote=True,
    )

    # Next request estimating 50 tokens passes
    res1 = governor.check_request(budget, is_remote=True, estimated_tokens=50)
    assert isinstance(res1, Success)

    # Next request estimating 150 tokens exceeds 1000 ceiling
    res2 = governor.check_request(budget, is_remote=True, estimated_tokens=150)
    assert isinstance(res2, Failure)
    assert res2.failure().code == "DAILY_TOKEN_LIMIT_EXCEEDED"


def test_governor_usage_summary_does_not_leak_prompts() -> None:
    governor = InferenceBudgetGovernor()
    governor.record_usage(
        request_id="req-test",
        tokens_in=100,
        tokens_out=50,
        cost_usd=0.005,
        is_remote=True,
        is_background=False,
    )
    summary = governor.get_usage_summary()
    assert summary["total_tokens"] == 150
    assert summary["remote_tokens"] == 150
    assert summary["local_tokens"] == 0
    assert summary["total_cost_usd"] == 0.005
    # Assure prompt text is not anywhere in summary
    assert "prompt" not in summary
    assert "text" not in summary
