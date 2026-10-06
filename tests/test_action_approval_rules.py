"""Tests for browser egress consent, approval rules, and interactive review."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from returns.result import Failure, Success
from typer.testing import CliRunner

from rai.actions.browser import register_browser_capabilities
from rai.actions.consent import (
    browser_consent,
    fingerprint,
    has_browser_consent,
    review_text,
)
from rai.actions.execution import SQLiteExecutionStore
from rai.actions.handles import SQLiteHandleStore
from rai.actions.service import ActionCapabilityService
from rai.cli import cli
from rai.commands.approval import TerminalApprovalBroker
from rai.configuration.approvals import matching_rule, update_rule
from rai.configuration.models import ApprovalRule
from rai.configuration.storage import ConfigurationError, save_settings
from rai.kernel.audit import InMemoryAuditLedger
from rai.kernel.capabilities import CapabilityRegistry
from rai.kernel.policy import PolicyEngine
from rai.kernel.ports import CancellationToken
from rai.kernel.records import (
    CapabilityRequest,
    DataClass,
    PolicyDecision,
    PolicyOutcome,
    ProducerIdentity,
    RiskClass,
)
from rai.kernel.transport import normalize_request


TEST_PRODUCER = ProducerIdentity(producer_id="test", kind="test", version="1.0.0")


def _make_browser_request(
    capability: str = "browser.search",
    data_class: DataClass = DataClass.PRIVATE,
    query: str = "european ai regulations",
) -> CapabilityRequest:
    return CapabilityRequest(
        producer=TEST_PRODUCER,
        actor=TEST_PRODUCER,
        capability=capability,
        arguments={"task_id": "test-task", "query": query},
        data_class=data_class,
        target_resource="https://html.duckduckgo.com/html/",
        requested_side_effects=("network",),
        isolation="host-api",
        verification_plan=("observed",),
    )


def test_consent_lifecycle_and_fingerprint() -> None:
    req1 = _make_browser_request()
    req2 = _make_browser_request(query="other query")

    fp1 = fingerprint(req1)
    fp2 = fingerprint(req2)
    assert isinstance(fp1, str) and len(fp1) == 64  # noqa: PLR2004 - sha256 hex digest length
    assert fp1 != fp2
    assert fp1 == fingerprint(req1)

    assert not has_browser_consent(req1)
    assert not has_browser_consent(req2)

    with browser_consent(req1):
        assert has_browser_consent(req1)
        assert not has_browser_consent(req2)

    assert not has_browser_consent(req1)


def test_review_text_formatting() -> None:
    req = _make_browser_request(query="test\nline")
    text = review_text(req)
    assert "Action: browser.search" in text
    assert "Class: PRIVATE" in text
    assert '"test\\nline"' in text
    assert "Only this query/target is authorized" in text


def test_matching_rule_precedence(tmp_path: Path) -> None:
    config_file = tmp_path / "config.yaml"
    settings_data = {
        "schema_version": 1,
        "agents": {
            "test": {
                "approval_rules": [
                    {
                        "id": "allow-all",
                        "capability": "browser.search",
                        "data_class": "PRIVATE",
                        "decision": "allow",
                        "query": None,
                    },
                    {
                        "id": "deny-specific",
                        "capability": "browser.search",
                        "data_class": "PRIVATE",
                        "decision": "deny",
                        "query": "secret query",
                    },
                    {
                        "id": "ask-specific",
                        "capability": "browser.search",
                        "data_class": "PRIVATE",
                        "decision": "ask",
                        "query": "prompt query",
                    },
                ]
            }
        },
    }
    save_settings(settings_data, str(config_file))

    # Match allow-all for arbitrary query
    req_normal = _make_browser_request(query="normal query")
    rule = matching_rule("test", req_normal, config_file)
    assert rule is not None
    assert rule.id == "allow-all"
    assert rule.decision == "allow"

    # Deny has highest priority
    req_deny = _make_browser_request(query="secret query")
    rule_deny = matching_rule("test", req_deny, config_file)
    assert rule_deny is not None
    assert rule_deny.id == "deny-specific"
    assert rule_deny.decision == "deny"

    # Ask has priority over allow
    req_ask = _make_browser_request(query="prompt query")
    rule_ask = matching_rule("test", req_ask, config_file)
    assert rule_ask is not None
    assert rule_ask.id == "ask-specific"
    assert rule_ask.decision == "ask"

    # Unknown profile returns None
    assert matching_rule("unknown", req_normal, config_file) is None


def test_update_rule_crud(tmp_path: Path) -> None:
    config_file = tmp_path / "config.yaml"
    save_settings({"schema_version": 1, "agents": {"default": {}}}, str(config_file))

    new_rule = ApprovalRule(
        id="rule-1",
        capability="browser.search",
        data_class="PRIVATE",
        decision="allow",
        query="ai",
    )
    update_rule("default", "rule-1", new_rule, config_file)

    req = _make_browser_request(query="ai")
    matched = matching_rule("default", req, config_file)
    assert matched is not None
    assert matched.id == "rule-1"

    # Update existing rule
    updated_rule = ApprovalRule(
        id="rule-1",
        capability="browser.search",
        data_class="PRIVATE",
        decision="deny",
        query="ai",
    )
    update_rule("default", "rule-1", updated_rule, config_file)
    assert matching_rule("default", req, config_file).decision == "deny"

    # Remove rule
    update_rule("default", "rule-1", None, config_file)
    assert matching_rule("default", req, config_file) is None

    # Error cases
    with pytest.raises(ConfigurationError):
        update_rule("nonexistent", "r", new_rule, config_file)
    with pytest.raises(ConfigurationError):
        update_rule("default", "nonexistent-rule", None, config_file)


@pytest.mark.asyncio
async def test_terminal_approval_broker_matching_rule(tmp_path: Path) -> None:
    config_file = tmp_path / "config.yaml"
    save_settings(
        {
            "schema_version": 1,
            "agents": {
                "test": {
                    "approval_rules": [
                        {
                            "id": "allow-rule",
                            "capability": "browser.search",
                            "data_class": "PRIVATE",
                            "decision": "allow",
                            "query": "known",
                        },
                        {
                            "id": "deny-rule",
                            "capability": "browser.search",
                            "data_class": "PRIVATE",
                            "decision": "deny",
                            "query": "forbidden",
                        },
                    ]
                }
            },
        },
        str(config_file),
    )

    broker = TerminalApprovalBroker(profile="test", path=config_file)
    decision = PolicyDecision(
        producer=TEST_PRODUCER,
        request_id="req-1",
        outcome=PolicyOutcome.ESCALATE,
        risk_class=RiskClass.HIGH,
        policy_version="1.0.0",
        reason_codes=("PRIVATE_DATA_EGRESS",),
        actor=TEST_PRODUCER,
        data_class=DataClass.PRIVATE,
        target_resource="https://duckduckgo.com",
        requested_side_effects=("network",),
        isolation="host-api",
        verification_plan=("observed",),
    )

    # Allow rule matches
    req_allow = _make_browser_request(query="known")
    result_allow = await broker.request_action(decision, req_allow, CancellationToken())
    assert isinstance(result_allow, Success)
    assert result_allow.unwrap() == "rule:test:allow-rule"

    # Deny rule matches
    req_deny = _make_browser_request(query="forbidden")
    result_deny = await broker.request_action(decision, req_deny, CancellationToken())
    assert isinstance(result_deny, Failure)
    assert result_deny.failure().code == "DENIED"

    # Cancelled token
    token = CancellationToken()
    token.cancel()
    result_cancel = await broker.request_action(decision, req_allow, token)
    assert isinstance(result_cancel, Failure)
    assert result_cancel.failure().code == "CANCELLED"


@pytest.mark.asyncio
async def test_terminal_approval_broker_interactive_remember(tmp_path: Path) -> None:
    config_file = tmp_path / "config.yaml"
    save_settings({"schema_version": 1, "agents": {"test": {}}}, str(config_file))

    broker = TerminalApprovalBroker(profile="test", path=config_file)
    decision = PolicyDecision(
        producer=TEST_PRODUCER,
        request_id="req-2",
        outcome=PolicyOutcome.ESCALATE,
        risk_class=RiskClass.HIGH,
        policy_version="1.0.0",
        reason_codes=("PRIVATE_DATA_EGRESS",),
        actor=TEST_PRODUCER,
        data_class=DataClass.PRIVATE,
        target_resource="https://duckduckgo.com",
        requested_side_effects=("network",),
        isolation="host-api",
        verification_plan=("observed",),
    )
    req = _make_browser_request(query="saved query")

    # Interactive choice "a" (remember allow)
    with patch("rai.commands.approval.choose", return_value="a"):
        res = await broker.request_action(decision, req, CancellationToken())
    assert isinstance(res, Success)
    assert "rule:test:search-" in res.unwrap()

    # Rule is now saved and matches automatically
    matched = matching_rule("test", req, config_file)
    assert matched is not None
    assert matched.decision == "allow"
    assert matched.query == "saved query"


@pytest.mark.asyncio
async def test_action_service_browser_private_escalation_flow(tmp_path: Path) -> None:
    class MockBackend:
        searches: list[str] = []

        async def search(self, query: str, token: CancellationToken) -> Success[tuple[dict[str, str], ...]]:
            self.searches.append(query)
            return Success(({"name": "Result", "url": "https://example.com"},))

    backend = MockBackend()
    handles = SQLiteHandleStore(tmp_path / "handles.db")
    registry = CapabilityRegistry()
    register_browser_capabilities(registry, handles, backend)

    class CustomBroker:
        async def request_action(
            self,
            decision: PolicyDecision,
            request: CapabilityRequest,
            cancellation: CancellationToken,
        ) -> Success[str]:
            return Success("approved-id")

        async def request(
            self, decision: PolicyDecision, cancellation: CancellationToken
        ) -> Success[str]:
            return Success("approved-id")

    audit = InMemoryAuditLedger()
    svc = ActionCapabilityService(registry, PolicyEngine(), audit, CustomBroker())
    svc.executions = SQLiteExecutionStore(tmp_path / "executions.db")
    svc.handles = handles

    req = normalize_request(
        registry.descriptor("browser.search"),
        {"task_id": "test-task", "query": "private search"},
        data_class=DataClass.PRIVATE,
    )

    decision, result = await svc.invoke(req)
    assert decision is not None
    assert decision.outcome == PolicyOutcome.ESCALATE
    assert isinstance(result, Success)
    assert backend.searches == ["private search"]


def test_profiles_approvals_cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    config_file = config_dir / "config.yaml"
    save_settings({"schema_version": 1, "agents": {"default": {}}}, str(config_file))
    monkeypatch.setenv("RAI_CONFIG_FILE", str(config_file))

    runner = CliRunner()

    # List initially empty
    res = runner.invoke(cli, ["profile", "approvals", "list", "default"])
    assert res.exit_code == 0
    assert res.output.strip() == "[]"

    # Set rule with query
    res_set = runner.invoke(
        cli,
        [
            "profile",
            "approvals",
            "set",
            "default",
            "search-rule-1",
            "allow",
            "--query",
            "regulations",
            "--data-class",
            "PRIVATE",
        ],
    )
    assert res_set.exit_code == 0
    assert "Saved rule search-rule-1 for profile default" in res_set.output

    # List shows the added rule
    res_list = runner.invoke(cli, ["profile", "approvals", "list", "default"])
    assert res_list.exit_code == 0
    assert "search-rule-1" in res_list.output
    assert "regulations" in res_list.output

    # Remove rule
    res_rm = runner.invoke(cli, ["profile", "approvals", "remove", "default", "search-rule-1"])
    assert res_rm.exit_code == 0
    assert "Removed rule search-rule-1" in res_rm.output

    # List is empty again
    res_list2 = runner.invoke(cli, ["profile", "approvals", "list", "default"])
    assert res_list2.exit_code == 0
    assert res_list2.output.strip() == "[]"
