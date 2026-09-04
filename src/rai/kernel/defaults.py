"""Default capability descriptors and compatibility implementations."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from returns.result import Failure, Result, Success

from rai.tools.desktop import get_desktop_adapter
from rai.tools.information import (
    arxiv_search,
    calculate,
    get_stock_price,
    web_search,
    wikipedia_search,
)
from rai.tools.python import run_secure_python_code
from rai.tools.security.hitl import get_approval_manager
from rai.tools.security.sandbox import get_sandbox_runner
from rai.tools.shell import run_secure_shell_command

from .capabilities import (
    CapabilityDescriptor,
    CapabilityHandler,
    CapabilityRegistry,
    RegisteredCapability,
)
from .ports import CancellationToken
from .records import ActionFailure, CapabilityRequest, PolicyDecision, ProducerIdentity, RiskClass

DEFAULT_ACTOR = ProducerIdentity(
    producer_id="rai.local-user", kind="local-user", version="1.0.0"
)


def _object_schema(
    properties: dict[str, dict[str, Any]], required: tuple[str, ...] = ()
) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": list(required),
        "additionalProperties": False,
    }


def _descriptor(  # noqa: PLR0913
    name: str,
    description: str,
    schema: dict[str, Any],
    risk: RiskClass,
    side_effects: tuple[str, ...],
    isolation: str,
    verification: tuple[str, ...],
) -> CapabilityDescriptor:
    return CapabilityDescriptor(
        name=name,
        description=description,
        input_schema=schema,
        risk_class=risk,
        side_effects=side_effects,
        isolation=isolation,
        requires_isolation=isolation == "bubblewrap-or-guix",
        verification_plan=verification,
    )


async def _shell(arguments: dict[str, Any]) -> dict[str, Any]:
    output = await run_secure_shell_command(
        arguments["command"], allow_network=arguments.get("allow_network", False)
    )
    if output.startswith(("Execution Error", "Execution Failure")):
        raise RuntimeError(output)
    return {"text": output}


async def _python(arguments: dict[str, Any]) -> dict[str, Any]:
    output = await run_secure_python_code(
        arguments["code"], allow_network=arguments.get("allow_network", False)
    )
    if output.startswith(("Execution Error", "Execution Failure")):
        raise RuntimeError(output)
    return {"text": output}


def _notification(arguments: dict[str, Any]) -> dict[str, Any]:
    output = get_desktop_adapter().send_notification(
        arguments["summary"], arguments["body"], arguments.get("app_name", "AI Assistant")
    )
    return {"text": output}


def _screenshot(arguments: dict[str, Any]) -> dict[str, Any]:
    return {"text": get_desktop_adapter().take_screenshot(arguments.get("delay", 0))}


def _weather(arguments: dict[str, Any]) -> dict[str, Any]:
    return {"text": get_desktop_adapter().weather(arguments.get("location", "current_location"))}


def _notification_compat(
    summary: str, body: str, app_name: str = "AI Assistant"
) -> str:
    return get_desktop_adapter().send_notification(summary, body, app_name)


def _screenshot_compat(delay: int = 0) -> str:
    return get_desktop_adapter().take_screenshot(delay)


def _weather_compat(location: str = "current_location") -> str:
    return get_desktop_adapter().weather(location)


def _echo(arguments: dict[str, Any]) -> dict[str, Any]:
    return {"text": arguments["text"]}


def _text_handler(
    function: Callable[[str], str], argument: str
) -> CapabilityHandler:
    def handler(arguments: dict[str, Any]) -> dict[str, Any]:
        return {"text": function(arguments[argument])}

    return handler


def create_default_capability_registry() -> CapabilityRegistry:
    """Create the canonical built-in registry without invoking any capability."""
    registry = CapabilityRegistry()
    entries = (
        (
            _descriptor(
                "run_shell_command",
                "Execute a shell command inside an isolated Linux workspace.",
                _object_schema(
                    {"command": {"type": "string"}, "allow_network": {"type": "boolean"}},
                    ("command",),
                ),
                RiskClass.HIGH,
                ("process", "filesystem", "network"),
                "bubblewrap-or-guix",
                ("sandbox-terminal-status",),
            ),
            _shell,
            run_secure_shell_command,
            ("ShellTools",),
        ),
        (
            _descriptor(
                "run_python_code",
                "Execute Python code inside an isolated Linux workspace.",
                _object_schema(
                    {"code": {"type": "string"}, "allow_network": {"type": "boolean"}},
                    ("code",),
                ),
                RiskClass.HIGH,
                ("process", "filesystem", "network"),
                "bubblewrap-or-guix",
                ("sandbox-terminal-status",),
            ),
            _python,
            run_secure_python_code,
            ("PythonTools",),
        ),
        (
            _descriptor(
                "send_desktop_notification",
                "Send a local desktop notification.",
                _object_schema(
                    {
                        "summary": {"type": "string"},
                        "body": {"type": "string"},
                        "app_name": {"type": "string"},
                    },
                    ("summary", "body"),
                ),
                RiskClass.MODERATE,
                ("desktop-notification",),
                "host-api",
                ("adapter-returned",),
            ),
            _notification,
            _notification_compat,
            ("DesktopNotificationTool",),
        ),
        (
            _descriptor(
                "take_desktop_screenshot",
                "Capture a desktop screenshot through the platform adapter.",
                _object_schema({"delay": {"type": "integer"}}),
                RiskClass.HIGH,
                ("screen-capture", "filesystem"),
                "host-api",
                ("adapter-returned",),
            ),
            _screenshot,
            _screenshot_compat,
            ("DesktopScreenshotTool",),
        ),
        (
            _descriptor(
                "get_desktop_weather",
                "Read weather information from the platform adapter.",
                _object_schema({"location": {"type": "string"}}),
                RiskClass.LOW,
                (),
                "host-api",
                ("adapter-returned",),
            ),
            _weather,
            _weather_compat,
            ("DesktopWeatherTool",),
        ),
        (
            _descriptor(
                "test.echo",
                "Return supplied text for deterministic conformance testing.",
                _object_schema({"text": {"type": "string"}}, ("text",)),
                RiskClass.LOW,
                (),
                "in-process",
                ("compare-output",),
            ),
            _echo,
            None,
            (),
        ),
        (
            _descriptor(
                "calculate",
                "Evaluate a bounded mathematical expression.",
                _object_schema({"expression": {"type": "string"}}, ("expression",)),
                RiskClass.LOW,
                (),
                "in-process",
                ("function-returned",),
            ),
            _text_handler(calculate, "expression"),
            calculate,
            ("CalculatorTools",),
        ),
        (
            _descriptor(
                "search.wikipedia",
                "Search Wikipedia.",
                _object_schema({"query": {"type": "string"}}, ("query",)),
                RiskClass.LOW,
                ("network",),
                "host-api",
                ("function-returned",),
            ),
            _text_handler(wikipedia_search, "query"),
            wikipedia_search,
            ("WikipediaTools",),
        ),
        (
            _descriptor(
                "search.web",
                "Search the public web.",
                _object_schema({"query": {"type": "string"}}, ("query",)),
                RiskClass.MODERATE,
                ("network",),
                "host-api",
                ("function-returned",),
            ),
            _text_handler(web_search, "query"),
            web_search,
            ("DuckDuckGoTools",),
        ),
        (
            _descriptor(
                "search.arxiv",
                "Search arXiv.",
                _object_schema({"query": {"type": "string"}}, ("query",)),
                RiskClass.LOW,
                ("network",),
                "host-api",
                ("function-returned",),
            ),
            _text_handler(arxiv_search, "query"),
            arxiv_search,
            ("ArxivTools",),
        ),
        (
            _descriptor(
                "finance.quote",
                "Read a public market quote.",
                _object_schema({"ticker": {"type": "string"}}, ("ticker",)),
                RiskClass.LOW,
                ("network",),
                "host-api",
                ("function-returned",),
            ),
            _text_handler(get_stock_price, "ticker"),
            get_stock_price,
            ("YFinanceTools",),
        ),
    )
    for descriptor, handler, compatibility_handler, groups in entries:
        registry.register(
            RegisteredCapability(descriptor, handler, compatibility_handler),
            compatibility_groups=groups,
        )
    return registry


def isolation_available(isolation: str) -> bool:
    if isolation != "bubblewrap-or-guix":
        return True
    return get_sandbox_runner().is_available()


class HitlApprovalBroker:
    """Compatibility adapter from the existing UI broker to the kernel port."""

    async def request(
        self, decision: PolicyDecision, cancellation: CancellationToken
    ) -> Result[str, ActionFailure]:
        if cancellation.cancelled:
            return Failure(self._failure(decision, "CANCELLED", "approval was cancelled"))
        manager = get_approval_manager()
        pending = manager.register_request(
            f"{decision.target_resource}: {','.join(decision.requested_side_effects)}",
            "CapabilityPolicy",
        )
        approved = await manager.wait_for_approval(pending)
        if cancellation.cancelled:
            return Failure(self._failure(decision, "CANCELLED", "approval was cancelled"))
        if not approved:
            return Failure(self._failure(decision, "DENIED", "approval denied"))
        return Success(pending.id)

    @staticmethod
    def _failure(decision: PolicyDecision, code: str, message: str) -> ActionFailure:
        return ActionFailure(
            record_id=f"failure:{decision.request_id}:{code}",
            timestamp=decision.timestamp,
            producer=ProducerIdentity(
                producer_id="rai.approval", kind="approval-broker", version="1.0.0"
            ),
            correlation_id=decision.correlation_id,
            request_id=decision.request_id,
            capability=decision.target_resource,
            code=code,
            message=message,
        )
