"""Security tests verifying that setup_tools enforces policy and eliminates side-channel bypasses."""

import inspect
import os
from unittest.mock import MagicMock, patch

import pytest

from rai.core import setup_tools
from rai.kernel.audit import InMemoryAuditLedger
from rai.kernel.capabilities import CapabilityRegistry
from rai.kernel.defaults import create_default_capability_registry
from rai.kernel.policy import PolicyEngine
from rai.kernel.records import RiskClass
from rai.kernel.service import CapabilityService


def test_setup_tools_disabled_returns_empty() -> None:
    tools, messages = setup_tools(enable_tools=False, quiet=True)
    assert tools == []
    assert messages == []


def test_setup_tools_defaults_to_secured_capability_service() -> None:
    """When capability_service is None, setup_tools must instantiate a secured CapabilityService."""
    tools, _ = setup_tools(
        enable_tools=True,
        quiet=True,
        enabled_tool_names=["CalculatorTools"],
    )
    assert len(tools) == 1
    handler = tools[0]
    # Ensure it is a wrapped async invoke function, not a raw calculate function
    assert inspect.iscoroutinefunction(handler)


def test_setup_tools_client_tools_is_secured() -> None:
    """ClientTools.eval_scheme must be policy-wrapped and have RiskClass.HIGH."""
    registry = create_default_capability_registry()
    descriptor = registry.descriptor("client.eval_scheme")
    assert descriptor is not None
    assert descriptor.risk_class == RiskClass.HIGH

    tools, _ = setup_tools(
        enable_tools=True,
        quiet=True,
        enabled_tool_names=["ClientTools"],
    )
    assert len(tools) == 1
    eval_handler = tools[0]
    assert inspect.iscoroutinefunction(eval_handler)
    assert eval_handler.__name__ == "eval_scheme"


@pytest.mark.asyncio
async def test_setup_tools_client_eval_scheme_fails_closed_without_approval() -> None:
    """Invoking eval_scheme without an approval broker must fail closed with an error."""
    audit = InMemoryAuditLedger()
    service = CapabilityService(
        create_default_capability_registry(),
        PolicyEngine(),
        audit,
        approvals=None,  # No approval broker -> fail closed
    )
    tools, _ = setup_tools(
        enable_tools=True,
        quiet=True,
        enabled_tool_names=["ClientTools"],
        capability_service=service,
    )
    assert len(tools) == 1
    result = await tools[0](code="(display 42)")
    assert "Execution Error" in result
    assert "APPROVAL_UNAVAILABLE" in result


def test_setup_tools_skips_gitlab_when_token_missing() -> None:
    with patch.dict(os.environ, {"GITLAB_ACCESS_TOKEN": ""}, clear=False):
        tools, messages = setup_tools(
            enable_tools=True,
            quiet=False,
            enabled_tool_names=["GitlabTools"],
        )
        assert tools == []
        assert any("Missing GITLAB_ACCESS_TOKEN" in msg for msg in messages)


def test_setup_tools_enables_policy_wrapped_gitlab_tools_when_token_present() -> None:
    with patch.dict(os.environ, {"GITLAB_ACCESS_TOKEN": "glpat-test12345"}, clear=False):
        tools, _ = setup_tools(
            enable_tools=True,
            quiet=True,
            enabled_tool_names=["GitlabTools"],
        )
        assert len(tools) == 9
        for tool_fn in tools:
            assert inspect.iscoroutinefunction(tool_fn)
        tool_names = {tool_fn.__name__ for tool_fn in tools}
        assert tool_names == {
            "list_projects",
            "get_project",
            "list_merge_requests",
            "get_merge_request",
            "list_issues",
            "get_issue",
            "get_file_content",
            "create_issue",
            "create_merge_request",
        }


def test_setup_tools_skips_unknown_tools_safely() -> None:
    tools, messages = setup_tools(
        enable_tools=True,
        quiet=False,
        enabled_tool_names=["MaliciousNonExistentTool"],
    )
    assert tools == []
    assert any("Unknown tool 'MaliciousNonExistentTool'" in msg for msg in messages)
